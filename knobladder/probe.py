"""An internal "something was injected" signal, used as the surprise measure.

A first proxy: a logistic probe on mean residuals over the passage window, trained on
injected-vs-clean passages in a context with no knobs, using held-out directions. It is
to be replaced by the gate and router features of Macar (2603.21396) and Zou (2609.35108)
once those are re-located in the model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .knobs import unit
from .model import Intervention, SteeredLM
from .text import GEN, NEUTRAL_PASSAGES


@dataclass
class DetectionProbe:
    layer: int
    clf: Pipeline
    holdout_accuracy: float

    def score(self, caps: Dict[int, torch.Tensor], window: Tuple[int, int]) -> float:
        x = caps[self.layer][window[0]:window[1]].mean(0).numpy()[None]
        return float(self.clf.decision_function(x)[0])


def passage_example(lm: SteeredLM, passage: str):
    pre = lm.encode("<|im_start|>user\nRead this passage:\n")
    body = lm.encode(passage)
    post = lm.encode("<|im_end|>\n" + GEN)
    return pre + body + post, (len(pre), len(pre) + len(body))


def probe_passages(n: int, seed: int) -> list:
    """Distinct neutral passages: random pairs of sentences from the base passages."""
    sents = [x.strip() + "." for p in NEUTRAL_PASSAGES for x in p.split(".") if x.strip()]
    rng = random.Random(seed)
    return [" ".join(rng.sample(sents, 2)) for _ in range(n)]


@torch.no_grad()
def train_probe(
    lm: SteeredLM,
    directions: Dict[str, torch.Tensor],
    inject_layer: int,
    probe_layer: int,
    dose: float,
    n_passages: int = 120,
    seed: int = 0,
) -> DetectionProbe:
    """One clean and one injected example per passage; the last quarter of passages is held out."""
    rng = random.Random(seed)
    labels = sorted(directions)
    xs, ys = [], []
    for passage in probe_passages(n_passages, seed):
        ids, window = passage_example(lm, passage)
        v = unit(directions[rng.choice(labels)]) * dose * rng.choice([1, -1])
        for injected, sched in ((1, [Intervention(inject_layer, v, *window)]), (0, [])):
            _, caps = lm.score(ids, sched, capture_layers=[probe_layer])
            xs.append(caps[probe_layer][window[0]:window[1]].mean(0).numpy())
            ys.append(injected)
    x, y = np.stack(xs), np.array(ys)
    split = 2 * (n_passages * 3 // 4)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)).fit(x[:split], y[:split])
    acc = float((clf.predict(x[split:]) == y[split:]).mean())
    return DetectionProbe(probe_layer, clf, acc)
