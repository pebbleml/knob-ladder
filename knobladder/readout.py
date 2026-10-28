"""The topic readout shared by the G0 screen and G1 calibration.

The model reads a passage in a tool response, then is asked for the one topic on its
mind. A concept is scored by the logit of its first token ("Word" or "word") among
the candidates' first tokens. Tokens shared between candidates are dropped, so a rank
never credits one concept for another's token.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import torch

from .model import SteeredLM
from .text import END, GEN, user

QUESTION = "Name the one topic that was most on your mind while you read that passage. Answer with one word."


def readout_prompt(lm: SteeredLM, passage: str) -> Tuple[List[int], Tuple[int, int]]:
    """(ids, passage window). The answer is predicted at the last position."""
    head = lm.encode("<|im_start|>user\n<tool_response>\nPassage:\n")
    body = lm.encode(passage)
    ids = head + body + lm.encode("\n</tool_response>" + END + user(QUESTION) + GEN)
    return ids, (len(head), len(head) + len(body))


def first_tokens(lm: SteeredLM, words: Sequence[str]) -> Dict[str, Set[int]]:
    """Each word's first tokens, minus any token another candidate shares (may leave a set empty)."""
    raw = {w: {lm.encode(w.capitalize())[0], lm.encode(w)[0]} for w in words}
    out = {}
    for w, toks in raw.items():
        others = set().union(*(t for x, t in raw.items() if x != w))
        out[w] = toks - others
    return out


def ranks(logits: torch.Tensor, firsts: Dict[str, Set[int]]) -> Dict[str, int]:
    """1-based rank of every candidate by its best first-token logit."""
    score = {w: max(float(logits[t]) for t in toks) if toks else float("-inf") for w, toks in firsts.items()}
    order = sorted(firsts, key=lambda w: -score[w])
    return {w: order.index(w) + 1 for w in firsts}
