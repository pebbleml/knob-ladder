"""Small pure-Python statistics, so analysis runs anywhere without torch."""
from __future__ import annotations

import math
import random
from typing import Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0, level: float = 0.95) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    ms = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    lo = ms[int((1 - level) / 2 * n)]
    hi = ms[min(n - 1, int((1 + level) / 2 * n))]
    return lo, hi


def sign_test(xs: Sequence[float], tol: float = 0.01) -> float:
    """Two-sided exact binomial p-value for the signs of xs (|x| <= tol counts as a tie and is dropped)."""
    pos, neg = sum(x > tol for x in xs), sum(x < -tol for x in xs)
    m = pos + neg
    if m == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def sign_flip_test(xs: Sequence[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation p-value for mean(xs) != 0, flipping signs at random."""
    if not xs:
        return 1.0
    rng = random.Random(seed)
    obs = abs(mean(xs))
    hits = sum(abs(mean([x if rng.random() < 0.5 else -x for x in xs])) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def chance_mrr(n: int) -> float:
    """Expected reciprocal rank of a uniformly random rank among n."""
    return sum(1 / r for r in range(1, n + 1)) / n
