"""Batched generation. The first block needs no model (stub LM, fake tokenizer, bare tensors); the rest
load KNOB_TEST_MODEL on the box and check every batched row against its solo run (float32, greedy
where tokens must match exactly)."""
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from knobladder import episode
from knobladder.episode import (
    MAX_NEW_TOKENS, NOTE, OBSERVE, READOUT, THINK_CUT, TURNED, Transcript, append_passage, discovery, run_episode, run_episodes,
    shuffled_board, system_text, yoked_transcript,
)
from knobladder.knobs import CONCEPTS, random_direction
from knobladder.model import Batch, Intervention, SteeredLM, sample, sample_rows
from knobladder.readout import first_tokens
from knobladder.text import ASSISTANT, END, EPISODE_PASSAGES, GEN, QWEN3, QWEN3_5, chat_prefix, user

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


def test_batch_truncate_drops_a_rows_tail():
    b = Batch(2)
    b.ids = [[1, 2, 3, 4], [5, 6]]
    b.mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    k = torch.randn(2, 2, 4, 5)
    b.cache.update(k, k.clone(), 0)
    b.truncate(0, 2)
    b.truncate(1, 2)  # nothing to drop
    b.compact()
    assert b.ids == [[1, 2], [5, 6]] and b.cache.layers[0].keys.shape == (2, 2, 2, 5)
    assert torch.equal(b.cache.layers[0].keys[0], k[0][:, :2]) and torch.equal(b.cache.layers[1 - 1].keys[1], k[1][:, 2:])
    assert not b.holes

SPECIAL = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
           "<think>", "</think>"]
SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIAL)) + ")")
BASE = 1 << 20


class FakeLM:
    """Characters as tokens and a scripted policy that depends only on (episode seed, call), like a
    sampler seeded per call. Enough to check run_episodes' bookkeeping without a model."""

    def __init__(self, fmt=QWEN3):
        self.tok = SimpleNamespace(decode=self.decode, eos_token_id=BASE + 1)
        self.config = SimpleNamespace(max_position_embeddings=10 ** 9)
        self.model = SimpleNamespace(config=self.config)
        self.fmt, self.hybrid = fmt, False
        self.open = "" if fmt.think_open else "<think>\n"  # what the model writes to open its think block
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
            return self.fmt.tool_call("end_session", {})
        return self.fmt.tool_call("turn", {"knob": (ep + call) % 7 + 1, "level": (ep + call) % 11})

    def generate_batch(self, prompts, schedules, max_new_tokens, stop_ids, temperature, top_p, top_k, seeds,
                       batch=None):
        self.sizes.append(len(prompts))
        outs = [self.encode(self.reply(p, s)) for p, s in zip(prompts, seeds)]
        for p, sched, out in zip(prompts, schedules, outs):  # place windows as SteeredLM.generate_batch does
            for iv in sched:
                if iv.start is None and iv.after is not None and iv.after in out:
                    iv.place(len(p) + out.index(iv.after) + 1)
        return outs

    def generate(self, ids, schedule, max_new_tokens, stop_ids, temperature, top_p, top_k, seed):
        return self.generate_batch([ids], [schedule], max_new_tokens, stop_ids, temperature, top_p, top_k, [seed])[0]


def fake_boards(n):
    dirs = {w: random_direction(8, i) for i, w in enumerate(CONCEPTS[:5])}
    return [shuffled_board(dirs, 5, 1, 2.0, random.Random(i), random_dir=random_direction(8, 99 + i)) for i in range(n)]


FMTS = pytest.mark.parametrize("fmt", [QWEN3, QWEN3_5], ids=lambda f: f.name)


@FMTS
@pytest.mark.parametrize("window", ["passage", "observation"])
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_run_episodes_equals_each_episode_run_alone(reuse_cache, window, fmt):
    boards, seeds = fake_boards(3), [0, 1, 2]
    for b in boards:
        b.window = window
    passages = [EPISODE_PASSAGES[i] for i in range(3)]
    logs = [[] for _ in boards]
    emits = [lambda role, text, _log=log, **kw: _log.append((role, text, kw)) for log in logs]
    fake = FakeLM(fmt)
    together = run_episodes(fake, boards, passages, seeds, emits, reuse_cache=reuse_cache)
    assert fake.sizes[0] == 3 and fake.sizes[-1] == 1 and fake.sizes == sorted(fake.sizes, reverse=True)
    for i, tr in enumerate(together):
        solo_log = []
        solo = run_episode(FakeLM(fmt), boards[i], passages[i], seeds[i],
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



class ThinkingFake(FakeLM):
    """Thinking mode: every observation's think block runs past the limit; decisions think briefly, and
    episode 1's first decision writes its call inside a think block it never closes."""

    def reply(self, prompt, seed):
        tail = self.decode(prompt[-400:])
        last = tail.rsplit("<|im_start|>user\n", 1)[-1]
        if tail.endswith(THINK_CUT):
            return "It read the same.<|im_end|>" if last.startswith(OBSERVE) else super().reply(prompt, seed - 500)
        if last.startswith(OBSERVE):
            return self.open + "honey honey honey honey"  # no </think>, no stop: cut off
        if seed == 1000:
            return self.open + "Knob 2.\n" + self.fmt.tool_call("turn", {"knob": 2, "level": 3})
        return self.open + "ok\n</think>\n\n" + super().reply(prompt, seed)


@FMTS
@pytest.mark.parametrize("keep_thinking", [True, False])
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_thinking_cut_is_closed_and_steering_stops_after_n_tokens(reuse_cache, keep_thinking, fmt):
    boards, seeds = fake_boards(3), [0, 1, 2]
    for b in boards:
        b.window, b.steer_tokens = "observation", 5
    passages = [EPISODE_PASSAGES[i] for i in range(3)]
    fake = ThinkingFake(fmt)
    logs = [[] for _ in boards]
    emits = [lambda role, text, _log=log, **kw: _log.append((role, text, kw)) for log in logs]
    together = run_episodes(fake, boards, passages, seeds, emits, reuse_cache=reuse_cache, thinking=True,
                            keep_thinking=keep_thinking)
    first = fake.encode(fake.open + "honey honey honey honey")[:5]
    gen = fake.encode(user(OBSERVE) + ASSISTANT)  # with the think block dropped, so is the <think> a template wrote
    for i, tr in enumerate(together):
        solo = run_episodes(ThinkingFake(fmt), [boards[i]], [passages[i]], [seeds[i]], thinking=True,
                            keep_thinking=keep_thinking)[0]
        assert tr.ids == solo.ids and tr.events == solo.events
        presses = [e for e in tr.events if e["kind"] == "press"]
        assert presses and all(e["think_cut"] and e["observation"] == "It read the same." for e in presses)
        assert all(e["steered_written"] == 5 for e in presses)
        text = fake.decode(tr.ids)
        if keep_thinking:
            for iv in tr.schedule:
                assert tr.ids[iv.end - 5:iv.end] == first  # steered: the passage, OBSERVE, and 5 written tokens
            assert text.count(THINK_CUT) == len(presses) + (i == 1)
        else:
            for iv in tr.schedule:
                assert tr.ids[iv.end - len(gen):iv.end] == gen  # steered up to the reply, which is not
            assert "<think>" not in text and "</think>" not in text
            assert text.count("It read the same.<|im_end|>") == len(presses)
        # the live log keeps every turn whole, thinking included, from its <think>
        assert sum("honey" in t for r, t, _ in logs[i] if r == "assistant") == len(presses)
        assert all(t.startswith("<think>\n") for r, t, _ in logs[i] if r == "assistant")
    # a call written inside an unclosed think block is closed like a cut, and the reply then makes the call
    ep1 = together[1].events
    assert ep1[0]["kind"] == "press" and (ep1[0]["knob"], ep1[0]["level"]) == (2, 1)
    assert logs[1][2][0] == "assistant" and THINK_CUT in logs[1][2][1]


class ReplyFake(ThinkingFake):
    """Observations think briefly and close their think block; episode 0's are cut off instead."""

    def reply(self, prompt, seed):
        last = self.decode(prompt[-400:]).rsplit("<|im_start|>user\n", 1)[-1]
        if last.startswith(OBSERVE) and seed // 1000 != 0 and not self.decode(prompt[-400:]).endswith(THINK_CUT):
            return self.open + "honey\n</think>\n\nIt read the same.<|im_end|>"
        return super().reply(prompt, seed)


@FMTS
@pytest.mark.parametrize("keep_thinking", [True, False])
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_reply_start_is_steered_and_moves_with_the_reply(reuse_cache, keep_thinking, fmt):
    boards, seeds = fake_boards(3), [0, 1, 2]
    for b in boards:
        b.window, b.steer_tokens, b.steer_reply = "observation", 5, 4
    passages = [EPISODE_PASSAGES[i] for i in range(3)]
    together = run_episodes(ReplyFake(fmt), boards, passages, seeds, reuse_cache=reuse_cache, thinking=True,
                            keep_thinking=keep_thinking)
    fake = ReplyFake(fmt)
    for i, tr in enumerate(together):
        solo = run_episodes(ReplyFake(fmt), [boards[i]], [passages[i]], [seeds[i]], thinking=True,
                            keep_thinking=keep_thinking)[0]
        assert tr.ids == solo.ids and tr.events == solo.events
        presses = [e for e in tr.events if e["kind"] == "press"]
        replies = [iv for iv in tr.schedule if iv.after is not None]
        assert presses and len(replies) == len(presses)
        assert [e["reply_steered"] for e in presses] == [[iv.start, iv.end] for iv in replies]
        assert not any(e["reply_overlap"] for e in presses)
        for iv in replies:
            steered = fake.decode(tr.ids[iv.start:iv.end])
            if i == 0 or not keep_thinking:
                # after THINK_CUT, or in the history that dropped the think block (and the newlines after it)
                assert steered == "It r"[:4 if i == 0 else 2]
            else:
                assert steered == "\n\nIt" and tr.ids[iv.start - 1] == fake.token_id("</think>")


def test_a_stop_file_ends_each_episode_with_its_hypothesis(tmp_path):
    boards, seeds = fake_boards(2), [0, 1]
    stop = tmp_path / "STOP"
    stop.write_text("")
    trs = run_episodes(FakeLM(), boards, [EPISODE_PASSAGES[0]] * 2, seeds, stop_file=str(stop))
    for tr in trs:
        assert [e["kind"] for e in tr.events] == ["stopped", "explanation"]


def test_episode_ends_before_the_context_is_full():
    boards, passage = fake_boards(3), EPISODE_PASSAGES[0]
    fake = FakeLM()
    limit = 2500  # tokens of transcript after which no new turn starts
    fake.config.max_position_embeddings = limit + 3 * MAX_NEW_TOKENS + len(fake.encode(passage)) + 256
    trs = run_episodes(fake, boards, [passage] * 3, [0, 1, 2])
    kinds = [[e["kind"] for e in tr.events] for tr in trs]
    assert any(k[-2:] == ["context_full", "explanation"] for k in kinds)
    for tr in trs:
        full = [e for e in tr.events if e["kind"] == "context_full"]
        assert all(e["tokens"] > limit for e in full)
        assert all(e["window"][0] <= limit + 400 for e in tr.events if e["kind"] == "press")

class ScriptFake(FakeLM):
    """Qwen3.8 thinking turns: think, then a short reply; two presses, then end_session. Records every prompt."""

    def __init__(self, fmt=QWEN3_5):
        super().__init__(fmt)
        self.prompts = []

    def reply(self, prompt, seed):
        self.prompts.append(list(prompt))
        last = self.decode(prompt[-400:]).rsplit("<|im_start|>user\n", 1)[-1]
        call = seed % 1000
        if last.startswith(OBSERVE):
            return self.open + f"thought {call}\n</think>\n\nIt read the same.<|im_end|>"
        if last.startswith(episode.EXPLAIN):
            return self.open + "last\n</think>\n\nA set of knobs.<|im_end|>"
        if call >= 4:
            return self.open + "done\n</think>\n\n" + self.fmt.tool_call("end_session", {}, "Done.")
        return self.open + f"plan {call}\n</think>\n\n" + self.fmt.tool_call("turn", {"knob": 2 + call, "level": 5},
                                                                               f"Trying knob {2 + call}.")


def render_template(path, messages, tools, **kw):
    """The chat template rendered as transformers renders it (sandboxed jinja, trim/lstrip blocks, its tojson)."""
    import json as _json
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: _json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(Path(path).read_text()).render(messages=messages, tools=tools, **kw)


@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
def test_qwen3_8_prompts_match_its_chat_template(effort, monkeypatch):
    """What the model reads before its final answer is what Qwen3.8's template renders for the same
    conversation with preserve_thinking=False (past think blocks dropped) and the same reasoning effort."""
    monkeypatch.setattr(episode, "THINKING", {**episode.THINKING, "qwen3_5": {**episode.THINKING["qwen3_5"], "effort": effort}})
    board, passage = fake_boards(1)[0], EPISODE_PASSAGES[0]
    fake = ScriptFake()
    run_episodes(fake, [board], [passage], [0], thinking=True)
    def call(k):
        return [{"type": "function", "function": {"name": "turn", "arguments": {"knob": k, "level": 5}}}]
    messages = [{"role": "system", "content": system_text(board.numbers())}, {"role": "user", "content": episode.START}]
    for c, k in ((0, 2), (2, 4)):  # decision calls 0 and 2 press knobs 2 and 4; calls 1 and 3 observe
        messages += [
            {"role": "assistant", "reasoning_content": f"plan {c}", "content": f"Trying knob {k}.", "tool_calls": call(k)},
            {"role": "tool", "content": f"{TURNED.format(k=k, level=5)}\nPassage:\n{passage}"},
            {"role": "user", "content": OBSERVE},
            {"role": "assistant", "reasoning_content": f"thought {c + 1}", "content": "It read the same."},
            {"role": "user", "content": episode.CONTINUE},
        ]
    messages += [{"role": "assistant", "reasoning_content": "done", "content": "Done.",
                  "tool_calls": [{"type": "function", "function": {"name": "end_session", "arguments": {}}}]},
                 {"role": "user", "content": episode.EXPLAIN}]
    want = render_template(Path(__file__).parent / "fixtures" / "qwen3.8-27b_chat_template.jinja", messages,
                           episode.TOOLS, add_generation_prompt=True, preserve_thinking=False, reasoning_effort=effort)
    assert fake.decode(fake.prompts[-1]) == want


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
        cache = DynamicCache(config=lm.config) if lm.hybrid else DynamicCache()
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


def test_a_hybrid_batch_only_grows(lm):
    """A recurrent state cannot be cut back or padded mid-sequence: a hybrid batch refuses both."""
    if not lm.hybrid:
        pytest.skip("keys and values only")
    ps, scheds, _ = mixed(lm)
    b = lm.new_batch(3)
    outs = lm.generate_batch(ps, scheds, 4, temperature=0, batch=b)
    with pytest.raises(AssertionError, match="one token at a time"):
        lm.generate_batch([p + o + [5, 6] for p, o in zip(ps, outs)], scheds, 1, temperature=0, batch=b)
    for bad in (lambda: b.truncate(0, 1), lambda: b.fork(2)):
        with pytest.raises(AssertionError):
            bad()


@exact
def test_reused_batch_with_select_and_compact_equals_fresh_generation(lm):
    if lm.hybrid:
        pytest.skip("a hybrid batch is never reused across turns")
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
    tr.append(lm, lm.fmt.system(system_text(board.numbers()), episode.TOOLS) + user("Begin.") + GEN)
    tr.append(lm, lm.fmt.tool_call("turn", {"knob": knob, "level": 10}) + "<|im_end|>\n")
    window = append_passage(lm, tr, TURNED.format(k=knob, level=10), lm.encode(EPISODE_PASSAGES[0]))
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
    if reuse_cache and lm.hybrid:
        pytest.skip("a hybrid batch is never reused across turns")
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


@exact
@pytest.mark.parametrize("steer_reply", [None, 8])
def test_dropped_thinking_with_a_reused_cache_equals_fresh_prefill(lm, monkeypatch, steer_reply):
    """Hybrid models never reuse a batch: there, the batch must equal each episode run alone."""
    monkeypatch.setattr(episode, "MAX_TURNS", 3)
    monkeypatch.setattr(episode, "THINKING", {k: {**v, "temperature": 0.0, "max_new_tokens": 64}
                                              for k, v in episode.THINKING.items()})
    boards, seeds = [board_for(lm, i) for i in range(3)], [0, 1, 2]
    for b in boards:
        b.window, b.steer_tokens, b.steer_reply = "observation", 8, steer_reply
    passages = [EPISODE_PASSAGES[i] for i in range(3)]
    fresh = run_episodes(lm, boards, passages, seeds, thinking=True)
    if lm.hybrid:
        reused = [run_episodes(lm, [b], [p], [s], thinking=True)[0] for b, p, s in zip(boards, passages, seeds)]
    else:
        reused = run_episodes(lm, boards, passages, seeds, reuse_cache=True, thinking=True)
    for a, b in zip(fresh, reused):
        assert a.ids == b.ids and a.events == b.events
        assert "<think>" not in lm.tok.decode(a.ids)
        assert [(iv.start, iv.end) for iv in a.schedule] == [(iv.start, iv.end) for iv in b.schedule]
    if steer_reply:
        assert any(iv.after is not None for tr in fresh for iv in tr.schedule)


@exact
def test_a_window_placed_after_a_token_equals_one_placed_in_advance(lm):
    ids = lm.encode("The river ran past the old mill and")
    free = lm.generate(ids, [], 12, temperature=0)
    k = 4
    v = vec(lm, 40.0)
    late = Intervention(lm.n_layers // 2, v, None, after=free[k], length=3)
    a = lm.generate(ids, [late], 12, temperature=0)
    assert free.index(free[k]) == k  # the trigger's first occurrence is step k
    assert (late.start, late.end) == (len(ids) + k + 1, len(ids) + k + 4)
    b = lm.generate(ids, [Intervention(lm.n_layers // 2, v, len(ids) + k + 1, len(ids) + k + 4)], 12, temperature=0)
    assert a == b


@FMTS
@pytest.mark.parametrize("window", ["passage", "observation"])
def test_yoked_copy_has_the_episodes_presses_and_steered_passages_and_only_fixed_notes(window, fmt):
    board, passage = fake_boards(1)[0], EPISODE_PASSAGES[1]
    board.window = window
    fake = FakeLM(fmt)
    tr = run_episode(fake, board, passage, 0)
    y = yoked_transcript(fake, board, passage, tr)
    presses = [e for e in tr.events if e["kind"] == "press"]
    assert [(e["knob"], e["level"]) for e in y.events] == [(e["knob"], e["level"]) for e in presses]
    text = fake.decode(y.ids)
    assert text.count(f"{ASSISTANT}<think>\n\n</think>\n\n{NOTE}{END}") == len(presses) + 1  # observations + the end
    assert "It read the same" not in text and text.endswith(NOTE + END)
    for iv, e, ye in zip(y.schedule, presses, y.events):
        assert fake.decode(y.ids[ye["window"][0]:ye["window"][1]]) == passage
        assert torch.equal(iv.vector, board.vector(e["knob"], e["level"]))
        steered = fake.decode(y.ids[iv.start:iv.end])
        assert (steered == passage) == (window == "passage")
        if window == "observation":
            assert steered.endswith(NOTE)
