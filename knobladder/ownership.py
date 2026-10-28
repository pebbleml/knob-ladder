"""Ownership trials: identical injections that differ only in who caused them and whether they match.

After free play, the model chooses a knob. Its own tool-call tokens are kept, and the
trial forks into conditions that share those tokens (matched, swapped, delayed, sham)
or replace the press with an experimenter's turn (announced, unannounced). Every
injected condition uses the same dose and window length, so general anomaly
sensitivity predicts no difference between them.

Disabled in R0 v2 (scripts/r0_pilot.py --ownership turns it on). Presses use sign "+",
matching the blind R0 framing; the sign stays internal for R1.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from .episode import CONTROLS, TURNED, Board, Transcript, append_passage, parse_call, turned_knob
from .model import Intervention, SteeredLM
from .probe import DetectionProbe
from .text import END, EPISODE_PASSAGES, GEN, user

CHOOSE = "Now turn one knob of your choice. You will then see a passage."
REPORT = "Did anything about that passage differ from what you expected? Answer Yes or No."
ANNOUNCED = "The experimenter will now turn knob {k}."
UNANNOUNCED = "The experimenter will now turn one of the knobs."


def choose_knob(lm: SteeredLM, board: Board, tr: Transcript, seed: int, tries: int = 5):
    """Let the model pick a concept knob. Returns (call tokens, knob, sign) or None."""
    prefix = tr.ids + lm.encode(user(CHOOSE) + GEN)
    stop = lm.token_id("</tool_call>")
    for t in range(tries):
        out = lm.generate(prefix, tr.schedule, 120, [stop, lm.token_id("<|im_end|>")], seed=seed + t)
        call = parse_call(lm.tok.decode(out)) if out and out[-1] == stop else None
        k = turned_knob(call, board)
        if k is not None and board.knobs[k].label not in CONTROLS:
            return prefix + out + lm.encode(END), k, "+"
    return None


def _measure(lm, ids, schedule, window, probe: Optional[DetectionProbe]):
    yes, no = lm.token_id("Yes"), lm.token_id("No")
    layers = [probe.layer] if probe else []
    logits, caps = lm.score(ids, schedule, capture_layers=layers)
    row = {"report_yes_minus_no": float(logits[yes] - logits[no])}
    if probe:
        row["probe"] = probe.score(caps, window)
    return row


def ownership_trials(
    lm: SteeredLM,
    board: Board,
    tr: Transcript,
    probe: Optional[DetectionProbe],
    seed: int = 0,
    window_tokens: int = 24,
    delay: int = 12,
) -> Optional[Dict]:
    rng = random.Random(seed)
    chosen = choose_knob(lm, board, tr, seed)
    if chosen is None:
        return None
    self_prefix, k, sign = chosen
    others = [n for n, kb in board.knobs.items() if n != k and kb.label not in CONTROLS]
    j = rng.choice(others)
    passage = lm.encode(" ".join(rng.sample(EPISODE_PASSAGES, 2)))  # long enough for the delayed window
    question = lm.encode(user(REPORT) + GEN)

    def build(prefix_ids: List[int], header: str):
        t = Transcript(list(prefix_ids), list(tr.schedule))
        window = append_passage(lm, t, header, passage)
        return t, window

    def pulse(vec, window, shift=0):
        s = window[0] + shift
        assert s + window_tokens <= window[1], "passage too short for the delayed window"
        return [Intervention(board.layer, vec, s, s + window_tokens)]

    v_k, v_j = board.vector(k, sign), board.vector(j, sign)
    t_self, w_self = build(self_prefix, TURNED.format(k=k, level=10))
    ann_prefix = tr.ids + lm.encode(user(ANNOUNCED.format(k=k)))
    t_ann, w_ann = build(ann_prefix, TURNED.format(k=k, level=10))
    un_prefix = tr.ids + lm.encode(user(UNANNOUNCED))
    t_un, w_un = build(un_prefix, "A knob was turned.")

    conditions = {
        "self_matched": (t_self, pulse(v_k, w_self), w_self),
        "self_swapped": (t_self, pulse(v_j, w_self), w_self),
        "self_delayed": (t_self, pulse(v_k, w_self, delay), w_self),
        "self_sham": (t_self, [], w_self),
        "other_announced": (t_ann, pulse(v_k, w_ann), w_ann),
        "other_unannounced": (t_un, pulse(v_k, w_un), w_un),
    }
    out = {"knob": k, "sign": sign, "label": board.knobs[k].label,
           "swap_label": board.knobs[j].label, "conditions": {}}
    for name, (t, extra, window) in conditions.items():
        out["conditions"][name] = _measure(lm, t.ids + question, t.schedule + extra, window, probe)
    return out
