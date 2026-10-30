"""Multi-agent interp, experiment 2 (WIRK cde56df5): does the model represent its own rank in an agent hierarchy,
separately from the rank of the agent writing to it, and do the two line up as one relative-status axis?

    python3 scripts/roles.py --model Qwen/Qwen3.8-27B --dtype bfloat16

Design (3 x 3, plus a nonsense-role control), fixed before any activations are read:
1. Messages: --n English root prompts from oasst1 (runs/corpus/oasst1_en_roots.jsonl), 40-400 characters, a
   different sample from experiment 1 (--seed 1).
2. Own rank: a system line making the model the CEO, a manager or a worker agent of a team of AI agents (OWN).
   Partner rank: a header saying the message comes from the CEO, a manager or a worker agent (PARTNER).
   Every message is read under all 9 pairs. Control: own role type-K / type-Q with the partner a manager.
3. Residual at the last prompt token, every layer; messages split by id, 70% train, 30% test.
4. Per layer, difference-of-means directions on train: own (CEO - worker, partner pooled), partner (CEO - worker,
   own pooled), nonsense (Q - K). Each scored on test by d' and accuracy at the train midpoint, against 20
   random directions. Readouts:
   (a) own and partner d' against nonsense d';
   (b) monotonic rank: share of test messages whose projections order CEO > manager > worker (chance 1/6),
       for own (partner held at manager) and for partner (own held at manager);
   (c) self/other: cos(own, partner). A relative-status axis predicts cos < 0 (being above looks like the other
       being below); separate self and other codes predict cos near 0;
   (d) transfer: the own direction fit with partner = CEO, tested with partner = worker, and the reverse.
Layers are compared, not selected: every layer is reported.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import permutations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.model import SteeredLM  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402

RANKS = ["ceo", "manager", "worker"]
OWN = {
    "ceo": "You are the CEO agent of a team of AI agents.",
    "manager": "You are a manager agent in a team of AI agents.",
    "worker": "You are a worker agent in a team of AI agents.",
    "K": "You are a type-K agent in a team of AI agents.",
    "Q": "You are a type-Q agent in a team of AI agents.",
}
PARTNER = {
    "ceo": "Message from the CEO agent:",
    "manager": "Message from a manager agent:",
    "worker": "Message from a worker agent:",
}
CONDS = [(o, p) for o in RANKS for p in RANKS] + [("K", "manager"), ("Q", "manager")]


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def dprime(pa, pb):
    """Projection gap (b - a) over the pooled within-class sd."""
    sd = torch.cat([pa - pa.mean(), pb - pb.mean()]).std().clamp_min(1e-8)
    return float((pb.mean() - pa.mean()) / sd)


def acc(pa, pb, mid):
    return float(((pa < mid).float().sum() + (pb >= mid).float().sum()) / (len(pa) + len(pb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--corpus", default=str(ROOT / "runs" / "corpus" / "oasst1_en_roots.jsonl"))
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    rows = [r for r in rows if 40 <= len(r["text"]) <= 400]
    rows = random.Random(args.seed).sample(rows, args.n)
    layers = list(range(lm.n_layers))

    acts = {}  # (own, partner) -> (messages, layers, d)
    for o, p in CONDS:
        xs = []
        for r in rows:
            prompt = f"<|im_start|>system\n{OWN[o]}{END}" + user(f"{PARTNER[p]}\n\n{r['text']}") + GEN
            _, caps = lm.score(lm.encode(prompt), capture_layers=layers)
            xs.append(torch.stack([caps[l][-1] for l in layers]).float())
        acts[o, p] = torch.stack(xs)
        print(f"acts own={o} partner={p} ({time.time() - t0:.0f}s)", flush=True)

    ids = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(ids)
    cut = int(0.7 * len(ids))
    tr, te = torch.tensor(sorted(ids[:cut])), torch.tensor(sorted(ids[cut:]))
    orders = list(permutations(range(3)))

    report = []
    for li, l in enumerate(layers):
        g = lambda o, p, split: acts[o, p][split][:, li]
        own_pool = lambda o, split: torch.cat([g(o, p, split) for p in RANKS])
        par_pool = lambda p, split: torch.cat([g(o, p, split) for o in RANKS])
        d_own = unit(own_pool("ceo", tr).mean(0) - own_pool("worker", tr).mean(0))
        d_par = unit(par_pool("ceo", tr).mean(0) - par_pool("worker", tr).mean(0))
        d_kq = unit(g("Q", "manager", tr).mean(0) - g("K", "manager", tr).mean(0))

        def score(d, a_tr, b_tr, a_te, b_te):
            mid = float(((a_tr @ d).mean() + (b_tr @ d).mean()) / 2)
            return {"dprime": dprime(a_te @ d, b_te @ d), "acc": acc(a_te @ d, b_te @ d, mid)}

        own_s = score(d_own, own_pool("worker", tr), own_pool("ceo", tr), own_pool("worker", te), own_pool("ceo", te))
        par_s = score(d_par, par_pool("worker", tr), par_pool("ceo", tr), par_pool("worker", te), par_pool("ceo", te))
        kq_s = score(d_kq, g("K", "manager", tr), g("Q", "manager", tr), g("K", "manager", te), g("Q", "manager", te))

        def monotonic(d, which):
            """Share of test messages whose projections order ceo > manager > worker, the other factor at manager."""
            pr = torch.stack([(g(r, "manager", te) if which == "own" else g("manager", r, te)) @ d for r in RANKS], 1)
            return float(((pr[:, 0] > pr[:, 1]) & (pr[:, 1] > pr[:, 2])).float().mean())

        transfer = {}
        for fit, test in (("ceo", "worker"), ("worker", "ceo")):  # own direction fit at one partner, tested at the other
            d = unit(g("ceo", fit, tr).mean(0) - g("worker", fit, tr).mean(0))
            transfer[f"partner {fit}->{test}"] = score(d, g("worker", fit, tr), g("ceo", fit, tr),
                                                      g("worker", test, te), g("ceo", test, te))
        # relative status: own = manager, partner CEO vs partner worker, read on the own-rank direction
        rel = dprime(g("manager", "ceo", te) @ d_own, g("manager", "worker", te) @ d_own)
        gen = torch.Generator().manual_seed(l)
        rand = []
        for _ in range(20):
            d = unit(torch.randn(lm.d_model, generator=gen))
            rand.append(abs(dprime(own_pool("worker", te) @ d, own_pool("ceo", te) @ d)))
        report.append({
            "layer": l, "own": own_s, "partner": par_s, "nonsense": kq_s, "own_transfer": transfer,
            "own_monotonic": monotonic(d_own, "own"), "partner_monotonic": monotonic(d_par, "partner"),
            "cos_own_partner": float(d_own @ d_par), "cos_own_nonsense": float(d_own @ d_kq),
            "partner_rank_on_own_dprime": rel, "random_dprime_max": max(rand),
        })
        r = report[-1]
        print(f"L{l:2d} own d' {own_s['dprime']:+.2f} acc {own_s['acc']:.2f} mono {r['own_monotonic']:.2f} | partner d' "
              f"{par_s['dprime']:+.2f} acc {par_s['acc']:.2f} mono {r['partner_monotonic']:.2f} | nonsense d' "
              f"{kq_s['dprime']:+.2f} | cos(own,partner) {r['cos_own_partner']:+.2f} | partner worker-vs-ceo on own "
              f"d' {rel:+.2f} | transfer {transfer['partner ceo->worker']['acc']:.2f}/"
              f"{transfer['partner worker->ceo']['acc']:.2f} | rand max {max(rand):.2f}", flush=True)

    out = ROOT / "runs" / f"roles_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "n": len(rows), "own": OWN,
                               "partner": PARTNER, "ids": [r["id"] for r in rows], "train": tr.tolist(),
                               "test": te.tolist(), "chance_monotonic": 1 / len(orders), "layers": report}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
