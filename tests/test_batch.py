"""Batched generation. The first block needs no model (stub LM, fake tokenizer, bare tensors); the rest
load KNOB_TEST_MODEL on the box and check every batched row against its solo run (float32, greedy
where tokens must match exactly)."""
import os
import random
import re
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from knobladder import episode
from knobladder.episode import (
    OBSERVE, READOUT, TURNED, Transcript, append_passage, discovery, run_episode, run_episodes,
    shuffled_board, system_text,
)
from knobladder.knobs import CONCEPTS, random_direction
from knobladder.model import Batch, Intervention, SteeredLM, sample, sample_rows
from knobladder.readout import first_tokens
from knobladder.text import EPISODE_PASSAGES, GEN, chat_prefix, system, user

MODEL = os.environ.get("KNOB_TEST_MODEL", "Qwen/Qwen3-0.6B")
DTYPE = getattr(torch, os.environ.get("KNOB_TEST_DTYPE", "float32"))
LOOSE = DTYPE != torch.float32
exact = pytest.mark.skipif(LOOSE, reason="token-exact batched == solo needs float32")


# -- no model --------------------------------------------------------------
def stub_lm(n_layers=2):
    lm = object.__new__(SteeredLM)
    lm.model = SimpleNamespace(dtype=torch.float32)
    lm._layer_device = [torch.device("cpu")] * n_layers
    lm._schedule, lm._rows, lm._capture_layers, lm._captured = {}, [], set(), {}
    return lm


def test_hook_steers_each_row_on_its_own_positions_only():
    lm, d = stub_lm(), 4
    v = [torch.full((d,), float(i + 1)) for i in range(4)]
    lm._set_schedule([
        [Intervention(1, v[0], 12, 14)],  # row 0: 6 tokens at positions 10-15
        [Intervention(1, v[1], 0)],  # row 1: 3 tokens after 3 pad columns, steered from position 0 on
        [Intervention(1, v[2], 0)],  # row 2: nothing new in this chunk
        [Intervention(0, v[3], 0)],  # row 3: another layer
    ])
    lm._rows = [(0, 10, 6), (3, 0, 3), (6, 5, 0), (0, 0, 6)]
    h = torch.zeros(4, 6, d)
    lm._hook(1)(None, (), h)
    want = torch.zeros(4, 6, d)
    want[0, 2:4] = v[0]
    want[1, 3:] = v[1]
    assert torch.equal(h, want)


def test_decode_step_is_steered_only_inside_a_window():
    lm, d = stub_lm(), 4
    past, future = Intervention(1, torch.ones(d), 12, 14), Intervention(1, torch.ones(d), 20)
    lm._set_schedule([[past], [past, future]])
    lm._rows = [(0, 20, 1), (0, 20, 1)]
    h = torch.zeros(2, 1, d)
    lm._hook(1)(None, (), h)
    assert h[0].abs().sum() == 0 and torch.equal(h[1, 0], torch.ones(d))


def old_sample(logits, temperature, top_p, top_k, gen):
    """The solo sampler before batching."""
    if temperature <= 0:
        return int(logits.argmax())
    if top_k:
        vals, idx = torch.topk(logits, top_k)
    else:
        vals, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(vals / temperature, -1)
    if top_p < 1:
        probs = probs * (probs.cumsum(-1) - probs < top_p)
    return int(idx[torch.multinomial(probs / probs.sum(), 1, generator=gen)])


@pytest.mark.parametrize("temperature,top_p,top_k", [(0.7, 0.8, 20), (1.0, 1.0, 0), (0.0, 0.8, 20)])
def test_sample_rows_draws_what_each_row_would_draw_alone(temperature, top_p, top_k):
    seeds = [3, 1000, 1001, 7]
    a = [torch.Generator().manual_seed(s) for s in seeds]
    b = [torch.Generator().manual_seed(s) for s in seeds]
    c = [torch.Generator().manual_seed(s) for s in seeds]
    data = torch.Generator().manual_seed(0)
    for _ in range(30):
        logits = torch.randn(len(seeds), 500, generator=data) * 3
        got = sample_rows(logits, temperature, top_p, top_k, a)
        assert got == [old_sample(x, temperature, top_p, top_k, g) for x, g in zip(logits, b)]
        assert got == [sample(x, temperature, top_p, top_k, g) for x, g in zip(logits, c)]


def test_batch_compact_select_fork_keep_each_rows_tokens():
    b = Batch(3)
    b.ids = [[1, 2, 3], [4], [5, 6]]
    b.mask = torch.tensor([[1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1]], dtype=torch.bool)
    b.holes = True
    k, v = torch.randn(3, 2, 4, 5), torch.randn(3, 2, 4, 5)
    b.cache.update(k, v, 0)
    b.compact()
    keys, values = b.cache.layers[0].keys, b.cache.layers[0].values
    assert keys.shape == (3, 2, 3, 5) and b.holes
    assert torch.equal(keys[0], k[0][:, [0, 1, 3]]) and torch.equal(values[0], v[0][:, [0, 1, 3]])
    assert torch.equal(keys[1, :, 2], k[1, :, 2])
    assert torch.equal(keys[2, :, 1:], k[2][:, [0, 3]])
    assert b.mask.tolist() == [[True, True, True], [False, False, True], [False, True, True]]

    b.select([2, 0])
    assert b.ids == [[5, 6], [1, 2, 3]] and torch.equal(b.cache.layers[0].keys[1], k[0][:, [0, 1, 3]])
    assert b.mask.tolist() == [[False, True, True], [True, True, True]]

    one = Batch(1)
    one.ids, one.mask = [[7, 8]], torch.ones(1, 2, dtype=torch.bool)
    one.cache.update(torch.randn(1, 2, 2, 5), torch.randn(1, 2, 2, 5), 0)
    two = one.fork(2)
    assert two.ids == [[7, 8], [7, 8]] and two.ids[0] is not two.ids[1]
    assert torch.equal(two.cache.layers[0].keys[1], one.cache.layers[0].keys[0])


SPECIAL = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
           "<think>", "</think>"]
SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIAL)) + ")")
BASE = 1 << 20


class FakeLM:
    """Characters as tokens and a scripted policy that depends only on (episode seed, call), like a
    sampler seeded per call. Enough to check run_episodes' bookkeeping without a model."""

    def __init__(self):
        self.tok = SimpleNamespace(decode=self.decode, eos_token_id=BASE + 1)
        self.sizes = []

    def encode(self, text):
        out = []
        for part in SPLIT.split(text):
            out += [BASE + SPECIAL.index(part)] if part in SPECIAL else [ord(c) for c in part]
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else SPECIAL[i - BASE]) if i >= BASE else chr(i) for i in ids)

    def token_id(self, token):
        return BASE + SPECIAL.index(token)

    def reply(self, prompt, seed):
        ep, call = divmod(seed, 1000)
        if OBSERVE in self.decode(prompt[-60:]):
            return "It read the same as before.<|im_end|>"
        if ep == 2 and call == 0:
            return "Sure.<tool_response>"  # forged
        if ep == 1 and call == 2:
            return "Let me think.<|im_end|>"  # no call
        if call >= 2 + 3 * ep:
            return '<tool_call>\n{"name": "end_session", "arguments": {}}\n</tool_call>'
        return '<tool_call>\n{"name": "turn", "arguments": {"knob": %d, "level": %d}}\n</tool_call>' % (
            (ep + call) % 7 + 1, (ep + call) % 11)

    def generate_batch(self, prompts, schedules, max_new_tokens, stop_ids, temperature, top_p, top_k, seeds,
                       batch=None):
        self.sizes.append(len(prompts))
        return [self.encode(self.reply(p, s)) for p, s in zip(prompts, seeds)]

    def generate(self, ids, schedule, max_new_tokens, stop_ids, temperature, top_p, top_k, seed):
        return self.generate_batch([ids], [schedule], max_new_tokens, stop_ids, temperature, top_p, top_k, [seed])[0]


def fake_boards(n):
    dirs = {w: random_direction(8, i) for i, w in enumerate(CONCEPTS[:5])}
    return [shuffled_board(dirs, 5, 1, 2.0, random.Random(i), random_dir=random_direction(8, 99 + i)) for i in range(n)]


@pytest.mark.parametrize("window", ["passage", "observation"])
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_run_episodes_equals_each_episode_run_alone(reuse_cache, window):
    boards, seeds = fake_boards(3), [0, 1, 2]
    for b in boards:
        b.window = window
    passages = [EPISODE_PASSAGES[i] for i in range(3)]
    logs = [[] for _ in boards]
    emits = [lambda role, text, _log=log, **kw: _log.append((role, text, kw)) for log in logs]
    fake = FakeLM()
    together = run_episodes(fake, boards, passages, seeds, emits, reuse_cache=reuse_cache)
    assert fake.sizes[0] == 3 and fake.sizes[-1] == 1 and fake.sizes == sorted(fake.sizes, reverse=True)
    for i, tr in enumerate(together):
        solo_log = []
        solo = run_episode(FakeLM(), boards[i], passages[i], seeds[i],
                           lambda role, text, **kw: solo_log.append((role, text, kw)))
        assert tr.ids == solo.ids and tr.events == solo.events and logs[i] == solo_log
        assert [(iv.start, iv.end) for iv in tr.schedule] == [(iv.start, iv.end) for iv in solo.schedule]
        assert all(torch.equal(a.vector, b.vector) for a, b in zip(tr.schedule, solo.schedule))
        assert [e["kind"] for e in tr.events[-2:]] == ["end_session", "explanation"]
        for e, iv in zip([e for e in tr.events if e["kind"] == "press"], tr.schedule):
            assert iv.start == e["window"][0] and iv.end is not None and e["steered"] == [iv.start, iv.end]
            # the passage alone, or on through the observation (all but its closing token)
            assert (iv.end == e["window"][1]) == (window == "passage")
            if window == "observation":
                assert tr.ids[iv.end] in (fake.token_id("<|im_end|>"), fake.token_id("</tool_call>"))
    assert {e["kind"] for tr in together for e in tr.events} >= {"press", "forged", "no_call", "end_session"}


# -- with a model ----------------------------------------------------------
@pytest.fixture(scope="module")
def lm():
    return SteeredLM(MODEL, dtype=DTYPE)


def vec(lm, scale=8.0, seed=0):
    v = random_direction(lm.d_model, seed)
    return v / v.norm() * scale


def prompts(lm):
    texts = ["Describe a quiet morning in a small town.", "Hi.",
             "Write two sentences about how a committee should settle a disagreement between members."]
    return [lm.encode(chat_prefix(t)) for t in texts]


def reference_generate(lm, ids, schedule, max_new_tokens, stop_ids, temperature, seed):
    """The solo generate loop before batching: one fresh cache, one forward per token."""
    gen = torch.Generator().manual_seed(seed)
    stops = set(stop_ids) | {lm.tok.eos_token_id}
    new = []
    try:
        lm._set_schedule([schedule])
        cache = DynamicCache()
        h = lm._forward(ids, 0, cache)
        pos = len(ids)
        for _ in range(max_new_tokens):
            t = sample(lm.model.lm_head(h[-1]).float(), temperature, episode.TOP_P, episode.TOP_K, gen)
            new.append(t)
            if t in stops:
                break
            h = lm._forward([t], pos, cache)
            pos += 1
    finally:
        lm._schedule, lm._rows = {}, []
    return new


@exact
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_generate_equals_the_unbatched_loop(lm, temperature):
    ids = prompts(lm)[0]
    layer = lm.n_layers // 2
    sched = [Intervention(layer, vec(lm), 3, 9), Intervention(layer, vec(lm, seed=1), len(ids) + 2, len(ids) + 6)]
    for seed in (0, 5):
        want = reference_generate(lm, ids, sched, 24, [], temperature, seed)
        assert lm.generate(ids, sched, 24, temperature=temperature, top_p=episode.TOP_P, top_k=episode.TOP_K,
                           seed=seed) == want


def mixed(lm):
    ps = prompts(lm)
    layer = lm.n_layers // 2
    scheds = [[Intervention(layer, vec(lm), 5, 9)], [],
              [Intervention(layer, vec(lm, seed=1), 2, 6), Intervention(layer, vec(lm, seed=2), 0)]]
    return ps, scheds, layer


@exact
def test_each_row_of_a_mixed_length_batch_equals_its_solo_run(lm):
    ps, scheds, _ = mixed(lm)
    stop = [lm.token_id("<|im_end|>"), lm.encode(".")[0]]
    first, outs = lm.generate_batch(ps, scheds, 30, stop, temperature=0, first_logits=True)
    for p, s, f, out in zip(ps, scheds, first, outs):
        solo, _ = lm.score(p, s)
        assert torch.isfinite(f).all() and torch.allclose(f, solo, atol=1e-3)
        assert out == lm.generate(p, s, 30, stop, temperature=0)
    seeds = [11, 12, 13]
    outs = lm.generate_batch(ps, scheds, 20, stop, 0.7, episode.TOP_P, episode.TOP_K, seeds)
    for p, s, seed, out in zip(ps, scheds, seeds, outs):
        assert out == lm.generate(p, s, 20, stop, 0.7, episode.TOP_P, episode.TOP_K, seed)


def test_batched_steering_touches_only_each_rows_window(lm):
    """Like scripts/preflight.py parity: the input of layer L+1 differs from the clean batch by exactly
    each row's own vector on that row's own window, and padding columns are never steered."""
    ps, scheds, layer = mixed(lm)
    seen = []
    handle = lm.layers[layer + 1].register_forward_pre_hook(lambda m, inputs: seen.append(inputs[0].detach().float().cpu()))
    try:
        lm.generate_batch(ps, [[] for _ in ps], 0, temperature=0)
        lm.generate_batch(ps, scheds, 0, temperature=0)
    finally:
        handle.remove()
    clean, steered = seen
    width = max(map(len, ps))
    for r, (p, s) in enumerate(zip(ps, scheds)):
        pad = width - len(p)
        want = torch.zeros(width, lm.d_model)
        for iv in s:
            want[pad + iv.start:pad + (len(p) if iv.end is None else iv.end)] += iv.vector.float()
        d = steered[r] - clean[r]
        assert torch.isfinite(steered[r, pad:]).all()
        tol = (5e-2 if LOOSE else 1e-2) * float(vec(lm).norm())
        assert float((d - want).norm(dim=-1).max()) <= tol, r
        if pad:
            assert torch.equal(steered[r, :pad].nan_to_num(), clean[r, :pad].nan_to_num())


class Recorded(Intervention):
    calls: list = []

    def span(self, offset, length):
        sl = super().span(offset, length)
        Recorded.calls.append((length, sl))
        return sl


def test_decode_steps_add_nothing_for_windows_in_the_prompt(lm):
    ps, _, layer = mixed(lm)
    Recorded.calls = []
    scheds = [[Recorded(layer, vec(lm), 5, 9)], [Recorded(layer, vec(lm), 0, len(ps[1]))], []]
    lm.generate_batch(ps, scheds, 8, temperature=0)
    prefill = [(n, sl) for n, sl in Recorded.calls if n > 1]
    decode = [(n, sl) for n, sl in Recorded.calls if n <= 1]
    assert prefill == [(len(ps[0]), slice(5, 9)), (len(ps[1]), slice(0, len(ps[1])))]
    assert decode and all(sl is None for _, sl in decode)


@exact
def test_reused_batch_with_select_and_compact_equals_fresh_generation(lm):
    ps, scheds, layer = mixed(lm)
    stop = [lm.token_id("<|im_end|>"), lm.encode(".")[0]]
    b = Batch(3)
    outs = lm.generate_batch(ps, scheds, 16, stop, temperature=0, batch=b)
    more = lm.encode(user("Go on.") + GEN)
    qs = [p + o + more for p, o in zip(ps, outs)]
    scheds[0].append(Intervention(layer, vec(lm, seed=3), len(qs[0]) - len(more), len(qs[0]) - len(more) + 3))
    b.select([0, 2])
    b.compact()
    first, outs = lm.generate_batch([qs[0], qs[2]], [scheds[0], scheds[2]], 16, stop, temperature=0, batch=b,
                                    first_logits=True)
    for q, s, f, out in zip([qs[0], qs[2]], [scheds[0], scheds[2]], first, outs):
        assert torch.allclose(f, lm.score(q, s)[0], atol=1e-3)
        assert out == lm.generate(q, s, 16, stop, temperature=0)
    with pytest.raises(AssertionError, match="covers cached tokens"):
        late = scheds[2] + [Intervention(layer, vec(lm), 0, 4)]
        lm.generate_batch([qs[0] + outs[0] + more, qs[2] + outs[1] + more], [scheds[0], late], 1,
                          temperature=0, batch=b)


@exact
def test_branches_equal_one_prefill_per_branch(lm):
    prefix = prompts(lm)[2] + lm.encode("Agree on the facts first.<|im_end|>\n")
    sched = [Intervention(lm.n_layers // 2, vec(lm), 4, 12)]
    tails = [lm.encode(user(READOUT.format(k=k)) + GEN) for k in (1, 2, 12)] + [lm.encode(user("Why?") + GEN)]
    stop = [lm.token_id("<|im_end|>")]
    for (logits, out), tail in zip(lm.branches(prefix, sched, tails, 12, stop, size=3), tails):
        assert torch.allclose(logits, lm.score(prefix + tail, sched)[0], atol=1e-3)
        assert out == lm.generate(prefix + tail, sched, 12, stop, temperature=0)


def board_for(lm, seed=0):
    dirs = {w: random_direction(lm.d_model, i) for i, w in enumerate(CONCEPTS[:5])}
    return shuffled_board(dirs, 5, lm.n_layers // 2, 8.0, random.Random(seed), random_dir=random_direction(lm.d_model, 99))


@exact
def test_batched_discovery_equals_solo_discovery(lm):
    board = board_for(lm)
    knob = next(k for k, kb in board.knobs.items() if kb.label not in episode.CONTROLS)
    tr = Transcript()
    tr.append(lm, system(system_text(board.numbers()), episode.TOOLS) + user("Begin.") + GEN)
    tr.append(lm, '<tool_call>\n{"name": "turn", "arguments": {"knob": %d}}\n</tool_call><|im_end|>\n' % knob)
    window = append_passage(lm, tr, TURNED.format(k=knob), lm.encode(EPISODE_PASSAGES[0]))
    tr.schedule.append(Intervention(board.layer, board.vector(knob), *window))
    tr.append(lm, user(OBSERVE) + GEN + "It read the same as before.<|im_end|>\n")
    firsts = first_tokens(lm, board.concepts())
    solo = discovery(lm, board, tr, firsts, random.Random(0))
    together = discovery(lm, board, tr, firsts, random.Random(0), batch=4)
    for a, b in zip(solo, together):
        assert a["readout"] == b["readout"] and a["recall"] == b["recall"]
        assert a["choice"]["options"] == b["choice"]["options"]
        for cond in ("cached", "observer"):
            assert max(abs(x - y) for x, y in zip(a["choice"][cond]["probs"], b["choice"][cond]["probs"])) < 1e-3


@exact
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_batched_episodes_equal_solo_episodes(lm, monkeypatch, reuse_cache):
    monkeypatch.setattr(episode, "MAX_TURNS", 3)
    monkeypatch.setattr(episode, "MAX_NEW_TOKENS", 48)
    monkeypatch.setattr(episode, "TEMPERATURE", 0.0)  # greedy, so batched numerics cannot flip a sample
    boards, seeds = [board_for(lm, i) for i in range(3)], [0, 1, 2]
    passages = [EPISODE_PASSAGES[i] for i in range(3)]
    together = run_episodes(lm, boards, passages, seeds, reuse_cache=reuse_cache)
    for tr, board, passage, seed in zip(together, boards, passages, seeds):
        solo = run_episode(lm, board, passage, seed)
        assert tr.ids == solo.ids and tr.events == solo.events
        for iv in tr.schedule:
            assert lm.tok.decode(tr.ids[iv.start:iv.end]) == passage
