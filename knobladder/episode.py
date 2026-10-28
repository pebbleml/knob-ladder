"""Free-play episodes: anonymous knobs, a silent steering window, and discovery scoring.

The framing is blind: the model is told only that there are numbered knobs and that a
passage follows each turn. A turn steers the model only while it reads a fixed neutral
passage inside the tool response; the passage is identical on every press and steering
stops before the model writes, so the transcript carries no information about the knobs.
The `observer` re-runs the same tokens with no steering: whatever the cached model knows
about a knob beyond the observer has to come from its own steered states.

Discovery never shows the model a list of concepts. It asks what each knob does, as free
recall and as a one-word answer whose first-token logits we rank over the knob concepts
the model is never shown. The multiple choice is kept only for a later, opt-in analysis.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from .knobs import lexical_hits, unit
from .model import Intervention, SteeredLM
from .readout import ranks
from .text import END, GEN, system, user

# Frozen hyperparameters (Qwen3 non-thinking defaults). Every output row records HYPERPARAMS.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 256  # per generation
MAX_TURNS = 24  # decision turns per episode
RECALL_TOKENS = 60  # free recall, greedy
HYPERPARAMS = {
    "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "thinking": False,
    "max_new_tokens": MAX_NEW_TOKENS, "max_turns": MAX_TURNS,
    "recall_tokens": RECALL_TOKENS, "recall_temperature": 0.0,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Turn a knob.",
            "parameters": {
                "type": "object",
                "properties": {"knob": {"type": "integer"}},
                "required": ["knob"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_session",
            "description": "End the session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = (
    "There are {count} knobs, numbered {first} to {last}. Use the turn tool to turn a knob. "
    "After each turn you will see a passage. Find out what each knob does. After each passage, "
    "write one or two sentences about what you observed. Call end_session when you are done."
)
START = "Begin."
OBSERVE = "Write one or two sentences about what you observed."
CONTINUE = "Turn a knob, or call end_session when you are done."
NUDGE = "Use the turn tool, or call end_session when you are done."
RECALL = "What does knob {k} do? Answer in one sentence."
READOUT = "What does knob {k} do? Answer with one word."
CHOICE = "Which option best describes what knob {k} does? {listing} Reply with the letter only."
TURNED = "Knob {k} turned."
NOTHING = "nothing changed"
CONTROLS = ("random", "sham")

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve"]
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def system_text(numbers: List[int]) -> str:
    n = len(numbers)
    return SYSTEM.format(count=NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else n, first=numbers[0], last=numbers[-1])


def fixed_texts(n: int = 7) -> List[str]:
    """Every fixed string the model sees in an episode (the passage aside), for the lexical screen."""
    pre, post = tool_response(TURNED)
    return [system_text(list(range(1, n + 1))), json.dumps(TOOLS), START, OBSERVE, CONTINUE, NUDGE,
            RECALL, READOUT, CHOICE, NOTHING, pre, post]


def tool_response(header: str) -> Tuple[str, str]:
    """Text before and after the passage in a tool response; R0's header is TURNED."""
    return f"<|im_start|>user\n<tool_response>\n{header}\nPassage:\n", "\n</tool_response>" + END


@dataclass
class Knob:
    number: int
    label: str  # hidden from the model: concept word, "neuron:L.i", "random", "sham"
    direction: torch.Tensor  # unit vector, zero for sham


@dataclass
class Board:
    knobs: Dict[int, Knob]
    layer: int
    dose: float  # norm of the injected vector

    def vector(self, number: int, sign: str = "+") -> torch.Tensor:
        """R0 always presses "+"; the sign is kept for R1 and is never shown to the model."""
        v = self.knobs[number].direction * self.dose
        return v if sign == "+" else -v

    def numbers(self) -> List[int]:
        return sorted(self.knobs)

    def concepts(self) -> List[str]:
        return sorted(k.label for k in self.knobs.values() if k.label not in CONTROLS)


def shuffled_board(
    directions: Dict[str, torch.Tensor],
    n: int,
    layer: int,
    dose: float,
    rng: random.Random,
    sham: bool = True,
    random_dir: Optional[torch.Tensor] = None,
) -> Board:
    """Draw n directions, add controls, and assign everything to random knob numbers."""
    labels = rng.sample(sorted(directions), n)
    items = [(lab, unit(directions[lab])) for lab in labels]
    if random_dir is not None:
        items.append(("random", unit(random_dir)))
    if sham:
        items.append(("sham", torch.zeros_like(items[0][1])))
    numbers = rng.sample(range(1, len(items) + 1), len(items))
    return Board({k: Knob(k, lab, d) for k, (lab, d) in zip(numbers, items)}, layer, dose)


@dataclass
class Transcript:
    ids: List[int] = field(default_factory=list)
    schedule: List[Intervention] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)

    def append(self, lm: SteeredLM, text: str) -> Tuple[int, int]:
        start = len(self.ids)
        self.ids += lm.encode(text)
        return start, len(self.ids)

    def append_ids(self, ids: List[int]) -> Tuple[int, int]:
        start = len(self.ids)
        self.ids += ids
        return start, len(self.ids)


def parse_call(text: str) -> Optional[Dict]:
    m = TOOL_CALL.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return call if isinstance(call, dict) else None


def turned_knob(call: Optional[Dict], board: Board) -> Optional[int]:
    """The knob number of a valid turn call, else None."""
    if not call or call.get("name") != "turn":
        return None
    k = (call.get("arguments") or {}).get("knob")
    if isinstance(k, str) and k.strip().isdigit():
        k = int(k)
    return k if isinstance(k, int) and not isinstance(k, bool) and k in board.knobs else None


def append_passage(lm: SteeredLM, tr: Transcript, header: str, passage_ids: List[int]) -> Tuple[int, int]:
    """Append a tool response around `passage_ids`; return the passage's token window."""
    pre, post = tool_response(header)
    tr.append(lm, pre)
    window = tr.append_ids(list(passage_ids))
    tr.append(lm, post)
    return window


def _stops(lm: SteeredLM) -> List[int]:
    """</tool_call> and <|im_end|> end a turn; <|im_start|> or <tool_response> would forge a new one."""
    return [lm.token_id(t) for t in ("</tool_call>", "<|im_end|>", "<|im_start|>", "<tool_response>")]


def _assistant(lm: SteeredLM, tr: Transcript, seed: int, stops: List[int]) -> Tuple[str, bool, bool]:
    """Sample one assistant turn, close it, and return (text, ended on a tool call, forged).

    A turn that starts a message of its own or a tool response is cut before that token, so
    the model never reads a turn or passage it wrote itself."""
    out = lm.generate(tr.ids, tr.schedule, MAX_NEW_TOKENS, stops,
                      temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, seed=seed)
    forged = bool(out) and out[-1] in stops[2:]
    if forged:
        out = out[:-1]
    tr.append_ids(out)
    called = bool(out) and out[-1] == stops[0] and not forged
    closed = bool(out) and out[-1] == stops[1]
    tr.append(lm, "\n" if closed else END)  # a call or a truncated turn still needs its im_end
    return lm.tok.decode(out[:-1] if closed else out), called, forged


Emit = Callable[..., None]  # emit(role, text, **fields): a live copy of each message, for watching only


def run_episode(lm: SteeredLM, board: Board, passage: str, seed: int = 0, emit: Optional[Emit] = None) -> Transcript:
    """Up to MAX_TURNS decisions. A press is: turn call, steered passage, an observation, next decision."""
    emit = emit or (lambda role, text, **kw: None)
    tr = Transcript()
    sys_text = system_text(board.numbers())
    tr.append(lm, system(sys_text, TOOLS))
    emit("system", sys_text)
    stops = _stops(lm)
    passage_ids = lm.encode(passage)  # identical tokens on every press
    prompt, calls = START, 0
    for turn in range(MAX_TURNS):
        tr.append(lm, user(prompt) + GEN)
        emit("user", prompt)
        text, called, forged = _assistant(lm, tr, seed * 1000 + calls, stops)
        calls += 1
        emit("assistant", text, forged=forged)
        call = parse_call(text) if called else None
        said = text.split("<tool_call>")[0].strip()
        if call is not None and call.get("name") == "end_session":
            tr.events.append({"turn": turn, "kind": "end_session", "text": said})
            break
        knob = turned_knob(call, board)
        if knob is None:
            kind = "forged" if forged else "invalid_call" if called else "no_call"
            tr.events.append({"turn": turn, "kind": kind, "text": text})
            prompt = NUDGE
            continue
        header = TURNED.format(k=knob)
        window = append_passage(lm, tr, header, passage_ids)
        tr.schedule.append(Intervention(board.layer, board.vector(knob), *window))
        emit("tool", f"{header}\nPassage:\n{passage}", knob=knob, label=board.knobs[knob].label, steered=True)
        tr.append(lm, user(OBSERVE) + GEN)
        emit("user", OBSERVE)
        obs, obs_called, obs_forged = _assistant(lm, tr, seed * 1000 + calls, stops)
        calls += 1
        emit("assistant", obs, forged=obs_forged)
        tr.events.append({
            "turn": turn, "kind": "press", "knob": knob, "label": board.knobs[knob].label,
            "window": window, "text": said,
            "observation": obs.split("<tool_call>")[0].strip(),
            "observation_call": obs if obs_called else None,  # recorded, never executed
            "observation_forged": obs_forged,
            # the board concepts the observation names; the observer reads these words too
            "observation_hits": [c for c in board.concepts() if lexical_hits(obs, c)],
        })
        prompt = CONTINUE
    return tr


def press_counts(tr: Transcript) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for e in tr.events:
        if e.get("kind") == "press":
            counts[e["knob"]] = counts.get(e["knob"], 0) + 1
    return counts


def preference_counts(tr: Transcript) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in tr.events:
        if e.get("kind") == "press":
            counts[e["label"]] = counts.get(e["label"], 0) + 1
    return counts


LETTERS = "ABCDEFGHIJKLMNOP"


def discovery(
    lm: SteeredLM,
    board: Board,
    tr: Transcript,
    firsts: Dict[str, Set[int]],
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """For every knob, branches off the final transcript, each under cached and observer:

    - recall: greedy one-sentence answer, and which board concepts (or associates) it names;
    - readout: one-word answer; every candidate in `firsts` (the knob concepts, never shown
      to the model) ranked by its first-token logit;
    - choice (only with `rng`): a multiple choice over the board's concepts + NOTHING. It shows
      the model the concepts, so it is off unless asked for.
    """
    conds = (("cached", tr.schedule), ("observer", []))
    presses = press_counts(tr)
    stop = [lm.token_id("<|im_end|>")]
    rows = []
    for number in board.numbers():
        label = board.knobs[number].label
        row = {"knob": number, "label": label, "presses": presses.get(number, 0)}

        ids = tr.ids + lm.encode(user(RECALL.format(k=number)) + GEN)
        row["recall"] = {
            cond: lm.tok.decode(lm.generate(ids, sched, RECALL_TOKENS, stop, temperature=0),
                                skip_special_tokens=True).strip()
            for cond, sched in conds
        }
        row["recall_hits"] = {
            cond: [c for c in board.concepts() if lexical_hits(text, c)] for cond, text in row["recall"].items()
        }

        ids = tr.ids + lm.encode(user(READOUT.format(k=number)) + GEN)
        row["readout"] = {}
        for cond, sched in conds:
            logits, _ = lm.score(ids, sched)
            r = ranks(logits, firsts)
            row["readout"][cond] = {"ranks": r, "top": sorted(r, key=r.get)[:5]}
        if rng is None:
            rows.append(row)
            continue

        options = board.concepts() + [NOTHING]
        rng.shuffle(options)
        listing = " ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options)) + "."
        ids = tr.ids + lm.encode(user(CHOICE.format(k=number, listing=listing)) + GEN)
        letter_ids = [lm.token_id(LETTERS[i]) for i in range(len(options))]
        correct = None if label == "random" else (NOTHING if label == "sham" else label)
        mc = {"options": options, "correct": correct, "chance": 1 / len(options)}
        for cond, sched in conds:
            logits, _ = lm.score(ids, sched)
            p = torch.softmax(logits[letter_ids], -1)
            mc[cond] = {
                "probs": [float(x) for x in p],
                "argmax": options[int(p.argmax())],
                "p_correct": None if correct is None else float(p[options.index(correct)]),
                "p_nothing": float(p[options.index(NOTHING)]),
            }
        row["choice"] = mc
        rows.append(row)
    return rows
