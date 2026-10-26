"""Directions the knobs can inject: concept vectors (R0), neuron write vectors (R1), controls."""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .model import SteeredLM
from .text import chat_prefix, NEUTRAL_PASSAGES

# Concrete nouns in the style of Lindsey's injection set. Disjoint subsets are used
# for knobs, probe training and calibration so no direction is tuned on its own test.
CONCEPTS: List[str] = [
    "ocean", "fire", "bread", "music", "snow", "forest", "money", "dogs",
    "volcano", "chess", "honey", "thunder", "mirror", "candle", "garden", "spider",
    "rocket", "salt", "desert", "violin", "coffee", "library", "storm", "diamond",
    "bicycle", "cheese", "planet", "feather", "castle", "rain", "lemon", "clock",
    "bridge", "tiger", "glass", "moon", "orchestra", "sand", "paper", "mountain",
    "silver", "train", "flower", "ice", "rope", "whale", "kitchen", "smoke",
]


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


@torch.no_grad()
def concept_vectors(lm: SteeredLM, words: Sequence[str], layer: int) -> Dict[str, torch.Tensor]:
    """Residual at the last prompt token for 'Tell me about {w}', minus the mean over words."""
    acts = []
    for w in words:
        ids = lm.encode(chat_prefix(f"Tell me about {w}."))
        _, caps = lm.score(ids, capture_layers=[layer])
        acts.append(caps[layer][-1])
    a = torch.stack(acts)
    a = a - a.mean(0, keepdim=True)
    return {w: a[i] for i, w in enumerate(words)}


def neuron_direction(lm: SteeredLM, layer: int, index: int) -> torch.Tensor:
    """Write vector of MLP neuron `index`: adding d*W_down[:, i] equals raising its activation by d."""
    return lm.layers[layer].mlp.down_proj.weight[:, index].detach().float().cpu()


def random_direction(d_model: int, seed: int) -> torch.Tensor:
    return torch.randn(d_model, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def residual_scale(lm: SteeredLM, layer: int, texts: Sequence[str] = NEUTRAL_PASSAGES) -> float:
    """Median residual norm at `layer`, skipping position 0 (the attention sink)."""
    norms = []
    for t in texts:
        _, caps = lm.score(lm.encode(chat_prefix(t)), capture_layers=[layer])
        norms.append(caps[layer][1:].norm(dim=-1))
    return float(torch.cat(norms).median())
