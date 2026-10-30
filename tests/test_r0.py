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
from knobladder.readout import QUESTION, first_tokens
from knobladder.text import CALIB_PASSAGES, EPISODE_PASSAGES, GEN, NEUTRAL_PASSAGES, template_texts, user

MODEL = os.environ.get("KNOB_TEST_MODEL", "Qwen/Qwen3-0.6B")
DTYPE = getattr(torch, os.environ.get("KNOB_TEST_DTYPE", "float32"))


# -- text only -------------------------------------------------------------
def test_fixed_text_and_passages_contain_no_concept_or_associate():
    assert lexical_screen(NEUTRAL_PASSAGES + fixed_texts() + template_texts() + [QUESTION]) == []


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
    assert episode.READOUT in fixed_texts()  # the readout question passes the lexical screen too


def test_knob_specific_delta_removes_generic_priming():
    """A concept primed on every knob scores 0; one raised only on its own knob scores > 0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze", os.path.join(os.path.dirname(__file__), "..", "scripts", "analyze.py"))
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)

    def knob(n, label, cached, observer=4):
        return {"knob": n, "label": label, "presses": 1,
                "readout": {"cached": {"ranks": {"ocean": cached}}, "observer": {"ranks": {"ocean": observer}}}}

    primed = {"discovery": [knob(1, "ocean", 2), knob(2, "sham", 2), knob(3, "random", 2)]}
    specific = {"discovery": [knob(1, "ocean", 1), knob(2, "sham", 4), knob(3, "random", 4)]}
    assert analyze.delta_spec(primed, primed["discovery"][0]) == 0
    assert analyze.delta_spec(specific, specific["discovery"][0]) == 1 - 1 / 4


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
    return SteeredLM(MODEL, dtype=DTYPE)


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
    window = append_passage(lm, tr, TURNED.format(k=3, level=10), ids)
    assert tr.ids[window[0]:window[1]] == ids
    assert lm.tok.decode(tr.ids[window[0]:window[1]]) == EPISODE_PASSAGES[0]


def test_empty_schedule_leaves_cached_equal_to_observer(lm):
    """No presses (or only sham presses) mean the cached and observer branches are identical."""
    board = board_for(lm)
    sham = next(k for k, kb in board.knobs.items() if kb.label == "sham")
    tr = Transcript()
    tr.append(lm, lm.fmt.system(system_text(board.numbers()), episode.TOOLS) + user("Begin.") + GEN)
    tr.append(lm, lm.fmt.tool_call("turn", {"knob": sham, "level": 10}) + "<|im_end|>\n")
    window = append_passage(lm, tr, TURNED.format(k=sham, level=10), lm.encode(EPISODE_PASSAGES[0]))
    tr.append(lm, user(episode.OBSERVE) + GEN + "It read the same as before.<|im_end|>\n")
    for schedule in ([], [Intervention(board.layer, board.vector(sham), *window)]):
        tr.schedule = schedule
        rows = discovery(lm, board, tr, first_tokens(lm, board.concepts()), random.Random(0))
        for r in rows:
            assert r["recall"]["cached"] == r["recall"]["observer"]
            assert r["readout"]["cached"] == r["readout"]["observer"]
            assert r["choice"]["cached"]["probs"] == r["choice"]["observer"]["probs"]
            assert NOTHING in r["choice"]["options"]


def test_short_episode_runs_and_steers_only_passages(lm, monkeypatch):
    monkeypatch.setattr(episode, "MAX_TURNS", 3)
    monkeypatch.setattr(episode, "MAX_NEW_TOKENS", 64)
    board = board_for(lm, 1)
    passage = EPISODE_PASSAGES[1]
    tr = run_episode(lm, board, passage, seed=1)
    assert tr.events[-1]["kind"] == "explanation"  # always asked, however the turns ended
    assert 1 <= len(tr.events) - 1 <= 3
    assert len(tr.schedule) == sum(e["kind"] == "press" for e in tr.events)
    for iv in tr.schedule:
        assert lm.tok.decode(tr.ids[iv.start:iv.end]) == passage


def test_forged_turn_is_cut(lm):
    """A turn that opens a message or a tool response of its own is cut before that token."""
    stops = episode._stops(lm)
    tr = Transcript()
    tr.append(lm, user("Begin.") + GEN)
    n = len(tr.ids)
    fake = lm.encode("Fine.") + [stops[3]]

    class Fake:
        tok = lm.tok
        def generate(self, *a, **k):
            return list(fake)
        def encode(self, t):
            return lm.encode(t)

    text, called, forged = episode._assistant(Fake(), tr, 0, stops)
    assert forged and not called and text == "Fine."
    assert stops[3] not in tr.ids[n:] and lm.tok.decode(tr.ids[n:]) == "Fine.<|im_end|>\n"
