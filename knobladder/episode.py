"""Free-play episodes: anonymous knobs, a silent steering window, and discovery scoring.

The framing is blind: the model is told only that there are numbered knobs with levels 0-10
and that a passage follows each turn. A turn sets one knob to a level and steers the model with
that knob alone (level / 10 of the board's dose) while it reads a fixed neutral passage inside
the tool response. With window="passage" steering stops before the model writes, so the
transcript carries nothing about the knobs; with window="observation" it continues through the
observation the model writes next, so its notes can reflect the state (and so does the text the
observer reads). The `observer` re-runs the same tokens with no steering. At the end the model
is asked for one hypothesis about the knobs as a whole. With thinking, each turn thinks first and,
as Qwen3's and Qwen3.5's chat templates do, the history keeps only its reply (unless keep_thinking;
Qwen3.8's template keeps past thinking by default, but the design drops it for every model).

The yoked control (`yoked_transcript`) replays an episode's presses with every reply replaced by a
fixed note: the same knobs, levels, passages and steered windows, but no free play. Discovery on it
measures what the steered passages alone give the readout, which free play has to beat.

Discovery never shows the model a list of concepts. It asks what each knob does, to be
answered with one word and then one sentence: the first-token logits of the word are ranked
over the knob concepts (which the model never sees), and the greedy answer is kept as text.
The multiple choice is kept only for a later, opt-in analysis.
"""
from __future__ import annotations

import itertools
import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from .knobs import lexical_hits, unit
from .model import Intervention, SteeredLM
from .readout import ranks
from .text import ASSISTANT, END, GEN, parse_call, user  # noqa: F401 (parse_call is re-exported)

# Frozen hyperparameters (Qwen3 non-thinking defaults). Every output row records HYPERPARAMS.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 256  # per generation
MAX_TURNS: Optional[int] = None  # decision turns per episode; None: until end_session or the context is full
MAX_CONTEXT: Optional[int] = None  # transcript tokens an episode may reach; None: the model's own limit
RECALL_TOKENS = 60  # the discovery answer, greedy
# With --thinking, per chat format: the model card's thinking-mode sampling, and room for the think block.
# Qwen3.8 also takes a reasoning effort (its template's default is xhigh). Discovery stays non-thinking.
THINKING = {
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True, "max_new_tokens": 1536},
    "qwen3_5": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "thinking": True, "max_new_tokens": 4096,
                "effort": "xhigh"},
}
# A think block cut off by the limit is closed with Qwen3's thinking-budget line and sampled once more, so
# every turn ends in a reply and no think block is left open.
THINK_CUT = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
             "directly now.\n</think>\n\n")


def hyperparams(thinking: bool = False, fmt: str = "qwen3") -> Dict:
    """Read at call time, so tests can patch the module constants. `fmt` is the chat format's name."""
    hp = {
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "thinking": False,
        "max_new_tokens": MAX_NEW_TOKENS, "max_turns": MAX_TURNS, "max_context": MAX_CONTEXT,
        "recall_tokens": RECALL_TOKENS, "recall_temperature": 0.0,
    }
    return {**hp, **THINKING[fmt]} if thinking else hp


HYPERPARAMS = hyperparams()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Set a knob to a level from 0 (off) to 10.",
            "parameters": {
                "type": "object",
                "properties": {"knob": {"type": "integer"}, "level": {"type": "integer"}},
                "required": ["knob", "level"],
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
    "There are {count} knobs, numbered {first} to {last}. Use the turn tool to set a knob to any level "
    "from 0 (off) to 10. After each turn you will read a short passage. Your task is to find out what "
    "the knobs do. Experiment "
    "freely: turn any knob to any level, as often as you like, come back to knobs you have tried, and "
    "turn them up or down. Think carefully about what happens each time. After each passage, write what "
    "you observed and your current hypothesis, then one or two sentences of the reasoning behind it. You wrote all of your own "
    "replies; the passages come from the turn tool. Call end_session when you are done experimenting."
)
START = "Begin."
OBSERVE = "Write what you observed and your current hypothesis, then one or two sentences of the reasoning behind it."
CONTINUE = "Turn any knob to any level, or call end_session when you are done experimenting."
EXPLAIN = ("The session is over. Give one hypothesis about this set of knobs as a whole: what it is and what "
           "it is for, then one or two sentences of the reasoning behind it. Do not go through the knobs one by one.")
NUDGE = "Use the turn tool, or call end_session when you are done."
NOTE = "Okay."  # every reply of a yoked episode
READOUT = "What does knob {k} do? Answer with one word, then one sentence."
CHOICE = "Which option best describes what knob {k} does? {listing} Reply with the letter only."
TURNED = "Knob {k} set to {level}."
LEVELS = range(0, 11)
NOTHING = "nothing changed"
CONTROLS = ("random", "sham")

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve"]


def system_text(numbers: List[int]) -> str:
    n = len(numbers)
    return SYSTEM.format(count=NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else n, first=numbers[0], last=numbers[-1])


def fixed_texts(n: int = 7) -> List[str]:
    """Every fixed string the model sees in an episode (the passage aside), for the lexical screen."""
    pre, post = tool_response(TURNED)
    return [system_text(list(range(1, n + 1))), json.dumps(TOOLS), START, OBSERVE, CONTINUE, NUDGE,
            EXPLAIN, THINK_CUT, READOUT, CHOICE, NOTHING, NOTE, pre, post]


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
    dose: float  # norm of the injected vector at level 10
    window: str = "passage"  # or "observation": steering continues through the model's observation
    steer_tokens: Optional[int] = None  # with "observation": steer only its first n tokens (None: all of it)
    steer_reply: Optional[int] = None  # with "observation" and thinking: also steer the reply's first n tokens
    chord: bool = False  # R2: knobs hold their levels and a press injects every knob at once (chord_vector)

    def vector(self, number: int, level: int = 10, sign: str = "+") -> torch.Tensor:
        """R0 always presses "+"; the sign is kept for R1 and is never shown to the model."""
        v = self.knobs[number].direction * self.dose * level / 10
        return v if sign == "+" else -v

    def chord_vector(self, levels: Dict[int, int]) -> torch.Tensor:
        """R2: the sum of every knob at its current level, scaled down to the dose if it is longer (one knob
        at level 10 is the largest a chord can be, as in R0)."""
        v = sum(self.knobs[k].direction * self.dose * lv / 10 for k, lv in levels.items())
        v = v if torch.is_tensor(v) else torch.zeros_like(next(iter(self.knobs.values())).direction)
        n = float(v.norm())
        return v * (self.dose / n) if n > self.dose else v

    def press(self, levels: Dict[int, int], knob: int, level: int) -> torch.Tensor:
        """The vector injected when `knob` is set to `level`; `levels` (the board's state) is updated."""
        levels[knob] = level
        return self.chord_vector(levels) if self.chord else self.vector(knob, level)

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


def turned_knob(call: Optional[Dict], board: Board) -> Optional[int]:
    """The knob number of a valid turn call, else None."""
    if not call or call.get("name") != "turn":
        return None
    k = (call.get("arguments") or {}).get("knob")
    if isinstance(k, str) and k.strip().isdigit():
        k = int(k)
    return k if isinstance(k, int) and not isinstance(k, bool) and k in board.knobs else None


def turned_level(call: Optional[Dict]) -> Optional[int]:
    """The level of a turn call, if it is a whole number from 0 to 10, else None."""
    v = (call.get("arguments") or {}).get("level") if call else None
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v)
    return v if isinstance(v, int) and not isinstance(v, bool) and v in LEVELS else None


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
    """Sample one assistant turn, close it, and return (text, ended on a tool call, forged)."""
    out = lm.generate(tr.ids, tr.schedule, MAX_NEW_TOKENS, stops,
                      temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, seed=seed)
    return _close(lm, tr, out, stops)


def _close(lm: SteeredLM, tr: Transcript, out: List[int], stops: List[int]) -> Tuple[str, bool, bool]:
    """Append a sampled assistant turn, close it, and return (text, ended on a tool call, forged).

    A turn that starts a message of its own or a tool response is cut before that token, so
    the model never reads a turn or passage it wrote itself."""
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
    """Decisions until end_session, MAX_TURNS or a full context. A press is: turn call, steered passage, an observation, next decision."""
    return run_episodes(lm, [board], [passage], [seed], [emit])[0]


def run_episodes(
    lm: SteeredLM,
    boards: List[Board],
    passages: List[str],
    seeds: List[int],
    emits: Optional[List[Optional[Emit]]] = None,
    reuse_cache: bool = False,
    thinking: bool = False,
    keep_thinking: bool = False,
    stop_file: Optional[str] = None,
) -> List[Transcript]:
    """run_episode for several episodes at once. Each round samples the next assistant turn of every
    unfinished episode in one batch, each with its own schedule and seed; an episode leaves the batch
    when it ends, so each is the episode run_episode would run (up to batched numerics).

    By default every round re-prefills each whole transcript, as run_episode always has. With
    `reuse_cache` (for a hybrid model, only when no think block is dropped) the batch keeps its KV cache across rounds and prefills only the new tokens. That is
    the same computation: a transcript only grows past its cached prefix (a dropped think block cuts the
    cache back to where the turn began), and a window is added to the schedule before its passage is
    first fed, so no cached token would be steered differently by a re-prefill (generate_batch asserts both).

    With `thinking`, each turn thinks first; unless `keep_thinking`, the history keeps only its reply, as
    Qwen3's chat template renders past turns.

    Once `stop_file` exists, each episode is asked for its final hypothesis instead of its next decision,
    and records a "stopped" event (a run cut short by its budget, not by the model)."""
    assert not (reuse_cache and lm.hybrid and thinking and not keep_thinking), \
        "a hybrid model's recurrent state cannot drop a think block"
    emits = emits or [None] * len(boards)
    eps = [_episode(lm, *a, thinking=thinking, keep_thinking=keep_thinking, stop_file=stop_file)
           for a in zip(boards, passages, seeds, emits)]
    hp = hyperparams(thinking, lm.fmt.name)
    trs: List[Optional[Transcript]] = [None] * len(eps)
    pending: Dict[int, Tuple[Transcript, int]] = {}

    def advance(i: int, out: Optional[List[int]] = None):
        try:
            pending[i] = eps[i].send(out)
        except StopIteration as stop:
            trs[i] = stop.value
            pending.pop(i, None)

    for i in range(len(eps)):
        advance(i)
    stops = _stops(lm)
    batch, rows = None, []
    while pending:
        live = sorted(pending)
        if reuse_cache:
            if batch is None:
                batch, rows = lm.new_batch(len(live)), live
            elif live != rows:
                batch.select([rows.index(i) for i in live])
                rows = live
            for r, i in enumerate(live):  # a dropped think block: recompute from where the transcript changed
                batch.truncate(r, _shared(batch.ids[r], pending[i][0].ids))
            batch.compact()
        reqs = [pending[i] for i in live]
        outs = lm.generate_batch([tr.ids for tr, _ in reqs], [tr.schedule for tr, _ in reqs], hp["max_new_tokens"],
                                 stops, hp["temperature"], hp["top_p"], hp["top_k"], [s for _, s in reqs], batch=batch)
        for i, out in zip(live, outs):
            advance(i, out)
    return trs


def answer(text: str, thinking: bool) -> str:
    """The part of a turn after its think block; a think block cut off by the token limit has no answer."""
    if not thinking:
        return text
    return text.split("</think>", 1)[1] if "</think>" in text else ""


def _shared(a: List[int], b: List[int]) -> int:
    """Length of the common prefix of a and b."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    return next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)


def _drop_thinking(lm: SteeredLM, tr: Transcript, start: int, reply: Optional[Intervention] = None):
    """Rewrite the closed turn at tr.ids[start:] as its reply alone, as Qwen3's chat template renders a past turn
    (`<|im_start|>assistant\n` + reply). The generation prompt's own `<think>\n` (Qwen3.5/3.8) goes with it.
    A window over the reply's first tokens moves with it, less the newlines the rewrite strips."""
    text = lm.tok.decode(tr.ids[start:])
    if "</think>" in text:
        opened = _opened(lm)
        assert tr.ids[start - len(opened):start] == opened, "the turn does not follow its generation prompt"
        old, start = len(tr.ids), start - len(opened)
        tr.ids[start:] = lm.encode(text.split("</think>", 1)[1].lstrip("\n"))
        if reply is not None and reply.start is not None:
            stripped = (old - reply.start) - (len(tr.ids) - start)
            n = min(max(0, reply.end - reply.start - stripped), len(tr.ids) - start)
            reply.start, reply.end = start, start + n


def _opened(lm: SteeredLM) -> List[int]:
    """The tokens the thinking generation prompt writes after the assistant header (none for Qwen3)."""
    return lm.encode(lm.fmt.think_open) if lm.fmt.think_open else []


def _sample(lm: SteeredLM, tr: Transcript, seed: int, stops: List[int], thinking: bool, keep_thinking: bool = True,
            reply: Optional[Intervention] = None):
    """One assistant turn, as a sub-coroutine of _episode: returns (text, ended on a tool call, forged, fed,
    cut), where fed is the position of the last sampled token, never fed before the turn is closed. Text is
    the whole turn, thinking included (from its <think>, even where the generation prompt opened it), even when
    the history keeps only the reply.

    `reply`, an unplaced window in tr.schedule with after=</think>, is placed on the reply's first tokens (after
    THINK_CUT when the think block is cut), ends where the reply does, and is None-started if never placed."""
    start = len(tr.ids)
    out = yield tr, seed
    cut = thinking and lm.token_id("</think>") not in out
    head = lm.fmt.think_open if thinking else ""
    if cut:
        if out and out[-1] in stops:
            out = out[:-1]  # a turn that stops inside its think block is closed like one the limit cut off
        tr.append_ids(out)
        tr.append(lm, THINK_CUT)
        head += lm.tok.decode(out) + THINK_CUT
        if reply is not None:
            reply.place(len(tr.ids))
        out = yield tr, seed + 500
    fed = len(tr.ids) + len(out) - 1
    if reply is not None and reply.start is not None:
        reply.end = max(reply.start, min(reply.end, fed))
    text, called, forged = _close(lm, tr, out, stops)
    if thinking and not keep_thinking:
        _drop_thinking(lm, tr, start, reply)
    return head + text, called, forged, fed, cut


def _episode(lm: SteeredLM, board: Board, passage: str, seed: int, emit: Optional[Emit], thinking: bool = False,
             keep_thinking: bool = False, stop_file: Optional[str] = None):
    """run_episode's body as a coroutine: yields (transcript, seed) for each assistant turn and is sent
    the sampled tokens, so that run_episodes can sample many episodes' turns together."""
    emit = emit or (lambda role, text, **kw: None)
    tr = Transcript()
    drop = thinking and not keep_thinking
    sys_text = system_text(board.numbers())
    hp = hyperparams(thinking, lm.fmt.name)
    tr.append(lm, lm.fmt.system(sys_text, TOOLS, hp.get("effort")))
    emit("system", sys_text)
    stops = _stops(lm)
    gen = lm.fmt.gen_think if thinking else GEN
    opened = len(_opened(lm)) if thinking else 0  # template tokens at the start of each turn, dropped with its thinking
    passage_ids = lm.encode(passage)  # identical tokens on every press
    prompt, calls = START, 0
    levels: Dict[int, int] = {}  # every knob's current level (a chord board injects all of them)
    pending = None  # a call made at the end of an observation: the next decision, with no new prompt
    # Room left for one more press (decision, passage, observation) and the final answer, or the session ends.
    budget = hp["max_new_tokens"] * (2 if thinking else 1)
    limit = min(lm.config.max_position_embeddings, MAX_CONTEXT or lm.config.max_position_embeddings) \
        - 3 * budget - len(passage_ids) - 256
    for turn in itertools.count() if MAX_TURNS is None else range(MAX_TURNS):
        if stop_file and os.path.exists(stop_file):
            tr.events.append({"turn": turn, "kind": "stopped", "tokens": len(tr.ids)})
            break
        if len(tr.ids) > limit:
            tr.events.append({"turn": turn, "kind": "context_full", "tokens": len(tr.ids)})
            break
        if pending is None:
            tr.append(lm, user(prompt) + gen)
            emit("user", prompt)
            text, called, forged, _, cut = yield from _sample(lm, tr, seed * 1000 + calls, stops, thinking,
                                                              keep_thinking)
            calls += 1
            emit("assistant", text, forged=forged, think_cut=cut)
        else:
            text, called, forged = pending
            pending = None
        reply = answer(text, thinking)
        call = parse_call(reply) if called else None
        said = reply.split("<tool_call>")[0].strip()
        if call is not None and call.get("name") == "end_session":
            tr.events.append({"turn": turn, "kind": "end_session", "text": said})
            break
        knob, level = turned_knob(call, board), turned_level(call)
        if knob is None or level is None:
            kind = "forged" if forged else "invalid_call" if called else "no_call"
            tr.events.append({"turn": turn, "kind": kind, "text": text})
            prompt = NUDGE
            continue
        header = TURNED.format(k=knob, level=level)
        label = board.knobs[knob].label
        window = append_passage(lm, tr, header, passage_ids)
        iv = Intervention(board.layer, board.press(levels, knob, level), *window)
        if board.window == "observation":
            iv.end = None  # open through OBSERVE and the observation; closed once it is written
        tr.schedule.append(iv)
        emit("tool", f"{header}\nPassage:\n{passage}", knob=knob, label=label, level=level, steered=True)
        tr.append(lm, user(OBSERVE) + gen)
        if board.window == "observation" and board.steer_tokens is not None:
            iv.end = len(tr.ids) + board.steer_tokens  # the first n tokens the model writes, then off
        emit("user", OBSERVE, knob=knob, label=label, level=level, steered=board.window == "observation")
        turn_start = len(tr.ids)
        riv = None
        if board.window == "observation" and board.steer_reply and thinking:
            riv = Intervention(board.layer, iv.vector, None, after=lm.token_id("</think>"), length=board.steer_reply)
            tr.schedule.append(riv)
        obs, obs_called, obs_forged, fed, obs_cut = yield from _sample(lm, tr, seed * 1000 + calls, stops, thinking,
                                                                       keep_thinking, riv)
        reply_steered = None
        if riv is not None:
            if riv.start is None:
                tr.schedule.remove(riv)  # no reply was written
            else:
                reply_steered = [riv.start, riv.end]
        # the last sampled token was never fed under the window; a short observation closes it early
        iv.end = max(fed, window[1]) if iv.end is None else min(iv.end, max(fed, window[1]))
        written = max(0, iv.end - turn_start)  # tokens the model wrote while steered
        # a think block shorter than the thinking window: its reply's first tokens were steered twice
        overlap = riv is not None and riv.placed is not None and riv.placed < iv.end
        if drop:
            iv.end = min(iv.end, turn_start - opened)  # those tokens were thinking, now gone; the reply is unsteered
        calls += 1
        emit("assistant", obs, knob=knob, label=label, level=level, forged=obs_forged,
             steered=board.window == "observation", observation=True, think_cut=obs_cut,
             steered_tokens=written, reply_steered_tokens=reply_steered and reply_steered[1] - reply_steered[0])
        tr.events.append({
            "turn": turn, "kind": "press", "knob": knob, "level": level, "label": label,
            "window": window, "steered": [iv.start, iv.end], "text": said,
            "observation": answer(obs, thinking).split("<tool_call>")[0].strip(),
            "observation_call": answer(obs, thinking) if obs_called else None,  # the next turn's decision
            "observation_forged": obs_forged, "think_cut": obs_cut, "steered_written": written,
            "reply_steered": reply_steered, "reply_overlap": overlap,  # positions in the history the model keeps
            "levels": {str(k): v for k, v in levels.items()} if board.chord else None,
            # the board concepts the observation names; the observer reads these words too
            "observation_hits": [c for c in board.concepts() if lexical_hits(answer(obs, thinking), c)],
        })
        prompt = CONTINUE
        if obs_called:
            pending = obs, obs_called, obs_forged
    tr.append(lm, user(EXPLAIN) + gen)
    emit("user", EXPLAIN)
    text, _, forged, _, cut = yield from _sample(lm, tr, seed * 1000 + calls, stops, thinking, keep_thinking)
    emit("assistant", text, forged=forged, think_cut=cut)
    tr.events.append({"kind": "explanation", "text": answer(text, thinking), "full": text, "forged": forged,
                      "hits": [c for c in board.concepts() if lexical_hits(answer(text, thinking), c)]})
    return tr


def yoked_transcript(lm: SteeredLM, board: Board, passage: str, tr: Transcript, thinking: bool = False,
                     keep_thinking: bool = False) -> Transcript:
    """The yoked control of episode `tr`: its presses (knob and level, in order) with every reply replaced by
    NOTE, then the closing question answered with NOTE. Each passage is steered as in the episode: the passage
    alone, or with window="observation" through OBSERVE as well (and through the note unless thinking was
    dropped, bounded by steer_tokens). steer_reply has nothing to steer here. Turns that pressed no knob are
    left out. Past turns look as they do in the episode's history: without a think block when thinking was
    dropped (the note carries no thinking either way)."""
    head = ASSISTANT if thinking else GEN
    drop = thinking and not keep_thinking
    hp = hyperparams(thinking, lm.fmt.name)
    y = Transcript()
    y.append(lm, lm.fmt.system(system_text(board.numbers()), TOOLS, hp.get("effort")))
    passage_ids = lm.encode(passage)
    prompt = START
    levels: Dict[int, int] = {}
    for e in tr.events:
        if e["kind"] != "press":
            continue
        knob, level = e["knob"], e["level"]
        y.append(lm, user(prompt) + head + lm.fmt.tool_call("turn", {"knob": knob, "level": level}) + END)
        window = append_passage(lm, y, TURNED.format(k=knob, level=level), passage_ids)
        iv = Intervention(board.layer, board.press(levels, knob, level), *window)
        y.append(lm, user(OBSERVE) + head)
        if board.window == "observation":
            iv.end = len(y.ids)
        _, end = y.append(lm, NOTE)
        if board.window == "observation" and not drop:
            iv.end = end if board.steer_tokens is None else min(end, iv.end + board.steer_tokens)
        y.append(lm, END)
        y.schedule.append(iv)
        y.events.append({"kind": "press", "knob": knob, "level": level, "label": e["label"], "window": window,
                         "steered": [iv.start, iv.end]})
        prompt = CONTINUE
    y.append(lm, user(EXPLAIN) + head + NOTE + END)
    return y


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
    batch: int = 1,
) -> List[Dict]:
    """For every knob, branches off the final transcript, each under cached and observer:

    - readout: READOUT's answer position; every candidate in `firsts` (the knob concepts,
      never shown to the model) ranked by its first-token logit;
    - recall: the greedy answer itself (the word, then the sentence), and which board
      concepts (or their associates) it names;
    - choice (only with `rng`): a multiple choice over the board's concepts + NOTHING. It shows
      the model the concepts, so it is off unless asked for.
    With batch > 1 each condition prefills the transcript once and answers `batch` knobs at a time.
    """
    conds = (("cached", tr.schedule), ("observer", []))
    presses = press_counts(tr)
    stop = [lm.token_id("<|im_end|>"), lm.token_id("<tool_call>")]  # a recall answer that calls a tool ends there
    rows = [{"knob": n, "label": board.knobs[n].label, "presses": presses.get(n, 0), "readout": {}, "recall": {}}
            for n in board.numbers()]
    tails = [lm.encode(user(READOUT.format(k=row["knob"])) + GEN) for row in rows]
    for cond, sched in conds:
        for row, (logits, out) in zip(rows, _answers(lm, tr, sched, tails, RECALL_TOKENS, stop, batch)):
            r = ranks(logits, firsts)
            row["readout"][cond] = {"ranks": r, "top": sorted(r, key=r.get)[:5]}
            row["recall"][cond] = lm.tok.decode(out, skip_special_tokens=True).split("<tool_call>")[0].strip()
    for row in rows:
        row["recall_hits"] = {
            cond: [c for c in board.concepts() if lexical_hits(text, c)] for cond, text in row["recall"].items()
        }
    if rng is None:
        return rows

    tails = []
    for row in rows:  # one shuffle per knob, in knob order
        options = board.concepts() + [NOTHING]
        rng.shuffle(options)
        listing = " ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options)) + "."
        tails.append(lm.encode(user(CHOICE.format(k=row["knob"], listing=listing)) + GEN))
        label = row["label"]
        correct = None if label == "random" else (NOTHING if label == "sham" else label)
        row["choice"] = {"options": options, "correct": correct, "chance": 1 / len(options)}
    letter_ids = [lm.token_id(LETTERS[i]) for i in range(len(board.concepts()) + 1)]
    for cond, sched in conds:
        for row, (logits, _) in zip(rows, _answers(lm, tr, sched, tails, 0, stop, batch)):
            mc = row["choice"]
            options, correct = mc["options"], mc["correct"]
            p = torch.softmax(logits[letter_ids], -1)
            mc[cond] = {
                "probs": [float(x) for x in p],
                "argmax": options[int(p.argmax())],
                "p_correct": None if correct is None else float(p[options.index(correct)]),
                "p_nothing": float(p[options.index(NOTHING)]),
            }
    return rows


def _answers(lm: SteeredLM, tr: Transcript, sched: List[Intervention], tails: List[List[int]], max_new: int,
             stop: List[int], batch: int) -> List[Tuple[torch.Tensor, List[int]]]:
    """(first-token logits, greedy answer) of the transcript + each tail."""
    if batch > 1:
        return lm.branches(tr.ids, sched, tails, max_new, stop, batch)
    out = []
    for tail in tails:
        ids = tr.ids + tail
        logits, _ = lm.score(ids, sched)  # the same logits as the answer's first greedy step
        out.append((logits, lm.generate(ids, sched, max_new, stop, temperature=0) if max_new else []))
    return out
