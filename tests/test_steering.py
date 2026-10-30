"""Invariants the whole ladder relies on. Run: python3 -m pytest tests -q (uses Qwen3-0.6B)."""
import importlib.util
import os

import pytest
import torch

from knobladder.knobs import neuron_direction, random_direction
from knobladder.model import Intervention, SteeredLM
from knobladder.text import chat_prefix

MODEL = os.environ.get("KNOB_TEST_MODEL", "Qwen/Qwen3-0.6B")
DTYPE = getattr(torch, os.environ.get("KNOB_TEST_DTYPE", "float32"))  # bfloat16 on the box, as in runs
LOOSE = DTYPE != torch.float32
TOL = 1e-2 if importlib.util.find_spec("fla") else 1e-3  # fla kernel noise; see test_batch.TOL


@pytest.fixture(scope="module")
def lm():
    return SteeredLM(MODEL, dtype=DTYPE)


@pytest.fixture(scope="module")
def ids(lm):
    return lm.encode(chat_prefix("Describe a quiet morning in a small town."))


def vec(lm, scale=8.0, seed=0):
    v = random_direction(lm.d_model, seed)
    return v / v.norm() * scale


def test_empty_schedule_matches_unhooked_model(lm, ids):
    logits, _ = lm.score(ids)
    with torch.no_grad():
        ref = lm.model(torch.tensor([ids], device=lm.device)).logits[0, -1].float().cpu()
    assert torch.allclose(logits, ref, atol=1e-4)


def test_intervention_only_touches_its_positions(lm, ids):
    layer, start = lm.n_layers // 2, len(ids) - 5
    _, clean = lm.score(ids, capture_layers=[layer + 1])
    _, steered = lm.score(ids, [Intervention(layer, vec(lm), start, start + 2)], capture_layers=[layer + 1])
    a, b = clean[layer + 1], steered[layer + 1]
    assert torch.allclose(a[:start], b[:start], atol=1e-5)
    assert not torch.allclose(a[start:start + 2], b[start:start + 2], atol=1e-3)


def test_incremental_generation_matches_prefill(lm, ids):
    """Steering during generation builds the same cache as re-prefilling with the same schedule."""
    layer = lm.n_layers // 2
    sched = [Intervention(layer, vec(lm), len(ids) + 2, len(ids) + 6)]
    new = lm.generate(ids, sched, max_new_tokens=10, temperature=0)
    full = ids + new
    # Greedy tokens must be reproduced by prefilling each prefix with the same schedule.
    for i in range(len(ids), len(full)):
        logits, _ = lm.score(full[:i], sched)
        assert int(logits.argmax()) == full[i]


def test_neuron_write_vector_equals_raising_activation(lm, ids):
    layer, index, delta = lm.n_layers // 2, 123, 3.0
    mlp = lm.layers[layer].mlp

    def bump(module, inputs):
        x = inputs[0].clone()
        x[..., index] += delta
        return (x,)

    handle = mlp.down_proj.register_forward_pre_hook(bump)
    try:
        via_neuron, _ = lm.score(ids)
    finally:
        handle.remove()
    d = neuron_direction(lm, layer, index) * delta
    via_vector, _ = lm.score(ids, [Intervention(layer, d, 0)])
    # In bf16 the two paths round differently (the add happens after down_proj, not inside it).
    assert torch.allclose(via_neuron, via_vector, atol=0.5 if LOOSE else TOL)
