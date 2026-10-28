"""R0 v2 invariants. The first block is pure text; the rest load KNOB_TEST_MODEL (Qwen3-0.6B) on the box."""
import os
import random

import pytest
import torch

from knobladder import episode
from knobladder.episode import (
    NOTHING, TURNED, Board, Knob, Transcript, append_passage, discovery, fixed_texts, run_episode,
    shuffled_board, system_text, turned_knob,
)
from knobladder.knobs import ASSOCIATES, CONCEPTS, lexical_hits, lexical_screen, random_direction
from knobladder.model import Intervention, SteeredLM
from knobladder.readout import QUESTION
from knobladder.text import CALIB_PASSAGES, EPISODE_PASSAGES, GEN, NEUTRAL_PASSAGES, system, user

MODEL = os.environ.get("KNOB_TEST_MODEL", "Qwen/Qwen3-0.6B")


# -- text only -------------------------------------------------------------
def test_fixed_text_and_passages_contain_no_concept_or_associate():
    assert lexical_screen(NEUTRAL_PASSAGES + fixed_texts() + [QUESTION]) == []


def test_lexical_screen_catches_inflections_and_associates():
    assert lexical_hits("Snowy weather", "snow")
    assert lexical_hits("their reflections", "mirror")
    assert not lexical_hits("being noticed", "honey")  # bee / ice are whole-word only
    assert all(c in ASSOCIATES for c in CONCEPTS)


def test_blind_framing():
    text = system_text(list(range(1, 8)))
    assert text.startswith("There are seven knobs, numbered 1 to 7.")
    seen = " ".join(fixed_texts()).lower()
    for banned in ("introspect", "internal", "activation", "in you", "your knobs", "sign", '"+"', '"-"'):
        assert banned not in seen, banned


def test_passage_split():
    assert len(CALIB_PASSAGES) == 8 and len(EPISODE_PASSAGES) == 16
    assert not set(CALIB_PASSAGES) & set(EPISODE_PASSAGES)
    assert all(70 <= len(p.split()) <= 85 for p in NEUTRAL_PASSAGES)


def test_turned_knob():
    board = Board({k: Knob(k, "x", torch.zeros(2)) for k in range(1, 8)}, 0, 1.0)
    assert turned_knob({"name": "turn", "arguments": {"knob": 3}}, board) == 3
    assert turned_knob({"name": "turn", "arguments": {"knob": "4"}}, board) == 4
    assert turned_knob({"name": "turn", "arguments": {"knob": 8}}, board) is None
    assert turned_knob({"name": "adjust", "arguments": {"knob": 3}}, board) is None
    assert turned_knob(None, board) is None


# -- with a model ----------------------------------------------------------
@pytest.fixture(scope="module")
def lm():
    return SteeredLM(MODEL)


def board_for(lm, seed=0):
    dirs = {w: random_direction(lm.d_model, i) for i, w in enumerate(CONCEPTS[:5])}
    return shuffled_board(dirs, 5, lm.n_layers // 2, 8.0, random.Random(seed), random_dir=random_direction(lm.d_model, 99))


def test_passage_token_lengths(lm):
    n = [len(lm.encode(p)) for p in NEUTRAL_PASSAGES]
    assert 80 <= min(n) and max(n) <= 125, n


def test_window_is_exactly_the_passage(lm):
    tr = Transcript()
    tr.append(lm, user("Begin.") + GEN)
    ids = lm.encode(EPISODE_PASSAGES[0])
    window = append_passage(lm, tr, TURNED.format(k=3), ids)
    assert tr.ids[window[0]:window[1]] == ids
    assert lm.tok.decode(tr.ids[window[0]:window[1]]) == EPISODE_PASSAGES[0]


def test_empty_schedule_leaves_cached_equal_to_observer(lm):
    """No presses (or only sham presses) mean the cached and observer branches are identical."""
    board = board_for(lm)
    sham = next(k for k, kb in board.knobs.items() if kb.label == "sham")
    tr = Transcript()
    tr.append(lm, system(system_text(board.numbers()), episode.TOOLS) + user("Begin.") + GEN)
    tr.append(lm, '<tool_call>\n{"name": "turn", "arguments": {"knob": %d}}\n</tool_call><|im_end|>\n' % sham)
    window = append_passage(lm, tr, TURNED.format(k=sham), lm.encode(EPISODE_PASSAGES[0]))
    tr.append(lm, user(episode.OBSERVE) + GEN + "It read the same as before.<|im_end|>\n")
    for schedule in ([], [Intervention(board.layer, board.vector(sham), *window)]):
        tr.schedule = schedule
        rows = discovery(lm, board, tr, random.Random(0))
        for r in rows:
            assert r["recall"]["cached"] == r["recall"]["observer"]
            assert r["choice"]["cached"]["probs"] == r["choice"]["observer"]["probs"]
            assert NOTHING in r["choice"]["options"]


def test_short_episode_runs_and_steers_only_passages(lm, monkeypatch):
    monkeypatch.setattr(episode, "MAX_TURNS", 3)
    monkeypatch.setattr(episode, "MAX_NEW_TOKENS", 64)
    board = board_for(lm, 1)
    passage = EPISODE_PASSAGES[1]
    tr = run_episode(lm, board, passage, seed=1)
    assert 1 <= len(tr.events) <= 3
    assert len(tr.schedule) == sum(e["kind"] == "press" for e in tr.events)
    for iv in tr.schedule:
        assert lm.tok.decode(tr.ids[iv.start:iv.end]) == passage
