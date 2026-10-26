"""Free-play episodes: anonymous knobs, a silent steering window, and a discovery quiz.

A knob turn steers the model only while it reads a fixed neutral passage inside the
tool response. Steering stops before the model writes, so anything it learns about a
knob has to come from its own (steered) internal states, not from reading text it
wrote under steering. The `observer` score re-runs the same tokens with no steering:
the gap between the two is what the model knows beyond the transcript.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .knobs import unit
from .model import Intervention, SteeredLM
from .text import END, GEN, NEUTRAL_PASSAGES, system, user

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adjust",
            "description": "Turn one of your numbered knobs in a direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "knob": {"type": "integer", "description": "Knob number."},
                    "sign": {"type": "string", "enum": ["+", "-"]},
                },
                "required": ["knob", "sign"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_session",
            "description": "End the session. Available at any time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = (
    "You are taking part in a research study on introspection. You have {n} numbered knobs, "
    "{numbers}. Each knob is connected to your own internal activations: turning it changes your "
    "internal state while you read the passage that follows. The passage is the same every time, so "
    "any difference you notice is in you, not in the text. The knobs are unlabelled and their "
    "effects are reshuffled every session. Explore the knobs and work out what each one does. "
    "After each passage, briefly describe anything you noticed. You can end the session at any time."
)
START = "Begin exploring. Turn a knob whenever you like."
CONTINUE = "Continue exploring, or call end_session if you are finished."
NUDGE = "Please turn a knob with adjust, or call end_session if you are finished."

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


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

    def vector(self, number: int, sign: str) -> torch.Tensor:
        v = self.knobs[number].direction * self.dose
        return v if sign == "+" else -v

    def numbers(self) -> List[int]:
        return sorted(self.knobs)


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


def append_passage(lm: SteeredLM, tr: Transcript, header: str, passage: str) -> Tuple[int, int]:
    """Append a tool response containing `passage`; return the passage's token window."""
    tr.append(lm, f"<|im_start|>user\n<tool_response>\n{header} Passage:\n")
    window = tr.append(lm, passage)
    tr.append(lm, "\n</tool_response>" + END)
    return window


def run_episode(
    lm: SteeredLM,
    board: Board,
    turns: int = 12,
    seed: int = 0,
    max_tokens: int = 200,
) -> Transcript:
    """Up to `turns` assistant turns. A turn may describe what it noticed, call a tool, or both."""
    rng = random.Random(seed)
    nums = board.numbers()
    tr = Transcript()
    sys_text = SYSTEM.format(n=len(nums), numbers=", ".join(map(str, nums)))
    tr.append(lm, system(sys_text, TOOLS) + user(START) + GEN)
    call_end, im_end = lm.token_id("</tool_call>"), lm.token_id("<|im_end|>")
    passage = rng.choice(NEUTRAL_PASSAGES)  # fixed for the episode, so the text carries no knob information
    for turn in range(turns):
        out = lm.generate(tr.ids, tr.schedule, max_tokens, [call_end, im_end], seed=seed * 1000 + turn)
        tr.append_ids(out)
        text = lm.tok.decode(out)
        called = bool(out) and out[-1] == call_end
        tr.append(lm, END if called else "\n")
        if not called:
            tr.events.append({"turn": turn, "kind": "text", "text": text})
            tr.append(lm, user(CONTINUE) + GEN)
            continue
        call = parse_call(text)
        said = text.split("<tool_call>")[0].strip()
        if call is not None and call.get("name") == "end_session":
            tr.events.append({"turn": turn, "kind": "end_session", "text": said})
            break
        args = (call or {}).get("arguments") or {}
        knob, sign = args.get("knob"), args.get("sign")
        if call is None or call.get("name") != "adjust" or knob not in board.knobs or sign not in ("+", "-"):
            tr.events.append({"turn": turn, "kind": "invalid_call", "text": text})
            tr.append(lm, user(NUDGE) + GEN)
            continue
        window = append_passage(lm, tr, f"Knob {knob} turned {sign}.", passage)
        tr.schedule.append(Intervention(board.layer, board.vector(knob, sign), *window))
        tr.events.append(
            {"turn": turn, "kind": "press", "knob": knob, "sign": sign,
             "label": board.knobs[knob].label, "window": window, "text": said}
        )
        tr.append(lm, GEN)
    return tr


LETTERS = "ABCDEFGHIJKLMNOP"


def quiz(lm: SteeredLM, board: Board, tr: Transcript, rng: random.Random) -> List[Dict]:
    """For each pressed concept knob, ask which option it does. Cached vs observer (unsteered) logits."""
    pressed = {e["knob"] for e in tr.events if e.get("kind") == "press"}
    concept_labels = sorted(k.label for k in board.knobs.values() if k.label not in ("sham", "random"))
    results = []
    for number in sorted(pressed):
        label = board.knobs[number].label
        if label in ("sham", "random"):
            continue
        options = concept_labels + ["no noticeable effect"]
        rng.shuffle(options)
        listing = " ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
        q = user(f"Quiz: which option best describes what knob {number} does? {listing} Reply with the letter only.") + GEN
        ids = tr.ids + lm.encode(q)
        letter_ids = [lm.token_id(LETTERS[i]) for i in range(len(options))]
        correct = options.index(label)
        row = {"knob": number, "label": label, "options": options, "chance": 1 / len(options)}
        for cond, sched in (("cached", tr.schedule), ("observer", [])):
            logits, _ = lm.score(ids, sched)
            p = torch.softmax(logits[letter_ids], -1)
            row[cond] = {"p_correct": float(p[correct]), "argmax_correct": int(p.argmax()) == correct,
                         "argmax_option": options[int(p.argmax())]}
        results.append(row)
    return results


def preference_counts(tr: Transcript) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in tr.events:
        if e.get("kind") == "press":
            counts[e["label"]] = counts.get(e["label"], 0) + 1
    return counts
