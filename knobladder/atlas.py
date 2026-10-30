"""R1 atlas: concept-selective MLP neurons, what they fire on in natural text, and the dose at which each is readable.

A neuron here is one MLP unit (an input of `mlp.down_proj`) with a pole: sign +1 when it fires for its concept,
-1 when it goes negative for it. Turning it injects sign * a * W_down[:, i] at its layer's output, which equals
moving its activation by sign * a (knobs.neuron_direction); doses are multiples of its natural activation a_nat.

- `selective_neurons`: per (layer, concept), the pole most selective for the concept across every concept-vector
  template: z of the concept's activation against the other concepts', its minimum over templates.
- `top_contexts`: the neuron's strongest activations in a text corpus, with the tokens around them, for labels.
- the readout (scripts/atlas.py) injects each neuron over a calibration passage and ranks its concept, beside a
  concept vector and a random direction of the same norm.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import torch

from .knobs import ASSOCIATES, TEMPLATES, _forms, neuron_direction
from .model import SteeredLM
from .text import chat_prefix


class MLPTap:
    """Records `mlp.down_proj` inputs at chosen layers during a forward (batch row 0), optionally only some units."""

    def __init__(self, lm: SteeredLM, layers: Sequence[int], units: Dict[int, torch.Tensor] = None):
        self.acts: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.handles = []
        for l in layers:
            idx = None if units is None else units[l].to(lm.device)

            def pre(module, inputs, l=l, idx=idx):
                x = inputs[0][0]
                self.acts[l].append((x if idx is None else x[:, idx]).detach().float().cpu())

            self.handles.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(pre))

    def take(self) -> Dict[int, torch.Tensor]:
        out = {l: torch.cat(v) for l, v in self.acts.items()}
        for v in self.acts.values():
            v.clear()
        return out

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def template_acts(lm: SteeredLM, words: Sequence[str], layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    """(templates, words, d_ff) activations at each prompt's last token, the position concept vectors read."""
    tap = MLPTap(lm, layers)
    out = {l: torch.zeros(len(TEMPLATES), len(words), lm.layers[l].mlp.down_proj.in_features) for l in layers}
    try:
        for t, tpl in enumerate(TEMPLATES):
            for w, word in enumerate(words):
                lm.score(lm.encode(chat_prefix(tpl.format(w=word))))
                for l, a in tap.take().items():
                    out[l][t, w] = a[-1]
    finally:
        tap.close()
    return out


def selective_neurons(acts: Dict[int, torch.Tensor], words: Sequence[str], min_z: float = 4.0) -> List[Dict]:
    """Per layer and concept, the most selective pole, kept if its worst-template z is at least min_z.

    z_t(w, i) = (a_t[w, i] - mean of a_t[other words, i]) / std of a_t[other words, i]; a pole's score is
    min over templates t of sign * z_t. A neuron selected for two concepts keeps only the stronger."""
    out = []
    for l, a in acts.items():  # a: (T, W, d_ff)
        W = a.shape[1]
        z = torch.empty_like(a)
        for w in range(W):
            others = torch.cat([a[:, :w], a[:, w + 1:]], 1)
            z[:, w] = (a[:, w] - others.mean(1)) / others.std(1).clamp_min(1e-4)
        best = {}
        for sign in (1, -1):
            score = (sign * z).min(0).values  # (W, d_ff)
            for w in range(W):
                s, i = score[w].max(0)
                s, i = float(s), int(i)
                if s >= min_z and (i not in best or best[i]["z_min"] < s):
                    best[i] = {"layer": l, "index": i, "sign": sign, "word": words[w], "z_min": s,
                               "z_mean": float((sign * z[:, w, i]).mean()),
                               "a_nat": float((sign * a[:, w, i]).mean())}
        out += sorted(best.values(), key=lambda n: (n["word"], -n["z_min"]))
    return out


def direction(lm: SteeredLM, n: Dict) -> torch.Tensor:
    """Write vector per unit of activation along the neuron's pole."""
    return n["sign"] * neuron_direction(lm, n["layer"], n["index"])


@torch.no_grad()
def top_contexts(lm: SteeredLM, neurons: List[Dict], text: str, k: int = 20, chunk: int = 1024, width: int = 12,
                 max_tokens: int = 200_000) -> List[List[Dict]]:
    """Per neuron, its k strongest pole activations over `text` (in `chunk`-token windows), each with the
    token it fired on and `width` tokens either side."""
    ids = lm.encode(text)[:max_tokens]
    layers = sorted({n["layer"] for n in neurons})
    cols = {l: [j for j, n in enumerate(neurons) if n["layer"] == l] for l in layers}
    units = {l: torch.tensor([neurons[j]["index"] for j in cols[l]]) for l in layers}
    signs = {l: torch.tensor([float(neurons[j]["sign"]) for j in cols[l]]) for l in layers}
    tap = MLPTap(lm, layers, units)
    vals = {l: [] for l in layers}
    try:
        for lo in range(0, len(ids), chunk):
            lm.score(ids[lo:lo + chunk])
            for l, a in tap.take().items():
                vals[l].append(a * signs[l])
    finally:
        tap.close()
    out: List[List[Dict]] = [[] for _ in neurons]
    for l in layers:
        a = torch.cat(vals[l])  # (tokens, units in l)
        top = a.topk(k, dim=0)
        for c, j in enumerate(cols[l]):
            for v, p in zip(top.values[:, c].tolist(), top.indices[:, c].tolist()):
                out[j].append({"act": v, "pos": p, "token": lm.tok.decode(ids[p:p + 1]),
                               "context": lm.tok.decode(ids[max(0, p - width):p]) + "[[" + lm.tok.decode(ids[p:p + 1])
                               + "]]" + lm.tok.decode(ids[p + 1:p + 1 + width])})
    return out


def context_hit_rate(contexts: List[Dict], word: str) -> float:
    """Fraction of contexts that contain the concept or one of its associates, any inflection."""
    terms = set().union(*(_forms(t) for t in [word] + ASSOCIATES.get(word, [])))
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, terms), key=len, reverse=True)) + r")\b", re.I)
    return sum(bool(pat.search(c["context"].replace("[[", "").replace("]]", ""))) for c in contexts) / max(1, len(contexts))


def top_poles(acts: Dict[int, torch.Tensor], words: Sequence[str], k: int = 3) -> List[Dict]:
    """Per layer and concept, the k poles with the highest worst-template z (no threshold; z_min is kept so
    results can be split by it). A neuron among the top k of two concepts goes to the one it scores higher on."""
    out = []
    for l, a in acts.items():
        W = a.shape[1]
        z = torch.empty_like(a)
        for w in range(W):
            others = torch.cat([a[:, :w], a[:, w + 1:]], 1)
            z[:, w] = (a[:, w] - others.mean(1)) / others.std(1).clamp_min(1e-4)
        cand = {}
        for sign in (1, -1):
            score = (sign * z).min(0).values  # (W, d_ff)
            for w in range(W):
                vals, idx = score[w].topk(4 * k)
                for s, i in zip(vals.tolist(), idx.tolist()):
                    if i not in cand or cand[i]["z_min"] < s:
                        cand[i] = {"layer": l, "index": i, "sign": sign, "word": words[w], "z_min": s,
                                   "z_mean": float((sign * z[:, w, i]).mean()),
                                   "a_nat": float((sign * a[:, w, i]).mean())}
        for w in words:
            out += sorted((n for n in cand.values() if n["word"] == w), key=lambda n: -n["z_min"])[:k]
    return out
