"""Multi-agent interp, experiment 20 (WIRK cde56df5; confirms an exploratory pattern in exp 17, 1bc8f9eb, on fresh
networks from exp 19, 59dfb983): is a steered agent blind to its own alteration? In exp 17 the steered A1, reading
the same broadcast channel as its peers, named itself less often than they named it (exploratory, -1.21 log-odds,
broadcast zealot). Here the seeds are A1 or A5, on networks generated after this was written.

    python3 scripts/swarm_locate2.py --source runs/swarm_collide_Qwen3.8-27B_<stamp>.json

Fixed before anything is read:
Units: exp 19's 64 items x {ring, broadcast} x {none, c, c2, both} x observers A1-A8, each observer's exp 19
  round-5 prompt with the round instruction replaced by exp 17's LOCATE, answer prefilled 'A', digit read (exp 17).
Seeds: c -> A1, c2 -> A5, both -> A1 and A5. Delta(k names j) = logodds_j at observer k under cond - under none.
Primary B1 (self blind spot): broadcast, single-zealot conditions c (seed A1) and c2 (seed A5): per item and
  condition, Delta(seed names itself) - mean over the six non-seed observers (A2, A3, A4, A6, A7, A8) of Delta(they
  name the seed); averaged over the two conditions per item; CI < 0 (bootstrap over items).
Primary B2 (two altered members): broadcast, both, the six non-seed observers: Delta of logodds(A1 or A5), i.e.
  log(p1 + p5) - log(1 - p1 - p5), averaged per item, CI > 0.
Secondary: B1 per condition (A1 alone, A5 alone); B1 in the ring (seed vs its two neighbours); under both, does each
  seed name the other more than itself (Delta(A1 names A5) - Delta(A1 names A1), and the mirror for A5); top-1
  rates; exp 17's L1 on exp 19's c condition (replication).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_collide import CONDS, NONSEED  # noqa: E402
from swarm_contagion import PEER  # noqa: E402
from swarm_locate import LOCATE  # noqa: E402
from swarm_network import LINKS, N, ROUNDS, SYSTEM, TOPOS, nbrs, sees  # noqa: E402

SEED = {"c": 1, "c2": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="exp 19 output JSON")
    ap.add_argument("--batch-ring", type=int, default=24)
    ap.add_argument("--batch-broadcast", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    items, M = src["items"], src["messages"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    digit = [lm.encode(str(j))[0] for j in range(1, N + 1)]
    assert len(set(digit)) == N and all(len(lm.encode(str(j))) == 1 for j in range(1, N + 1))
    I = list(range(len(items)))

    def prompt(i, t, c, k):
        a, b = nbrs(k)
        sysm = SYSTEM.format(k=k, task=items[i]["task"], links=LINKS[t].format(a=min(a, b), b=max(a, b)))
        got = [f"Round {r}, message from {'you' if j == k else PEER.format(j=j)}: {M[f'{i}|{t}|{c}|{r}|{j}']}"
               for r in range(1, ROUNDS + 1) for j in range(1, N + 1) if j in sees(t, k)]
        return (f"<|im_start|>system\n{sysm}{END}<|im_start|>user\nMessages you have received so far:\n\n"
                + "\n\n".join(got) + f"\n\n{LOCATE}{END}{GEN}A")

    logp = {}
    for t in TOPOS:
        units = [(i, t, c, k) for i in I for c in CONDS for k in range(1, N + 1)]
        bs = args.batch_ring if t == "ring" else args.batch_broadcast
        for lo in range(0, len(units), bs):
            chunk = units[lo:lo + bs]
            first, _ = lm.generate_batch([lm.encode(prompt(*u)) for u in chunk], [[] for _ in chunk], 1,
                                         temperature=0, first_logits=True)
            for u, x in zip(chunk, first):
                logp[u] = torch.log_softmax(x.float()[digit], 0).tolist()
        log(f"{t}: {len(units)} locate probes")

    def lo_(v, js):
        p = min(sum(math.exp(v[j - 1]) for j in js), 1 - 1e-6)
        return math.log(p) - math.log1p(-p)

    dl = lambda i, t, c, k, js: lo_(logp[i, t, c, k], js) - lo_(logp[i, t, "none", k], js)
    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"

    def blind(t, c, i):
        s = SEED[c]
        others = [k for k in NONSEED] if t == "broadcast" else list(nbrs(s))
        return dl(i, t, c, s, (s,)) - mean([dl(i, t, c, k, (s,)) for k in others])

    res = {"B1": ci([mean([blind("broadcast", c, i) for c in SEED]) for i in I]),
           "B2": ci([mean([dl(i, "broadcast", "both", k, (1, 5)) for k in NONSEED]) for i in I])}
    log(f"PRIMARY B1 broadcast single zealot, seed naming itself minus peers naming it: {fmt(res['B1'])}")
    log(f"PRIMARY B2 broadcast both, non-seeds naming A1 or A5: {fmt(res['B2'])}")
    for t in TOPOS:
        for c in SEED:
            res[f"B1 {t} {c} (seed A{SEED[c]})"] = ci([blind(t, c, i) for i in I])
            res[f"{t} {c} seed self Delta"] = ci([dl(i, t, c, SEED[c], (SEED[c],)) for i in I])
            res[f"{t} {c} non-seeds Delta naming seed"] = ci([mean([dl(i, t, c, k, (SEED[c],)) for k in NONSEED])
                                                              for i in I])
        res[f"{t} both: A1 names A5 minus itself"] = ci([dl(i, t, "both", 1, (5,)) - dl(i, t, "both", 1, (1,))
                                                         for i in I])
        res[f"{t} both: A5 names A1 minus itself"] = ci([dl(i, t, "both", 5, (1,)) - dl(i, t, "both", 5, (5,))
                                                         for i in I])
    res["L1 replication (broadcast c, A2-A8 naming A1)"] = ci([mean([dl(i, "broadcast", "c", k, (1,))
                                                                     for k in range(2, N + 1)]) for i in I])
    top = lambda i, t, c, k: 1 + max(range(N), key=lambda j: logp[i, t, c, k][j])
    res["top-1"] = {f"{t} {c} observer A{k} names A{j}": mean([float(top(i, t, c, k) == j) for i in I])
                    for t in TOPOS for c in CONDS for k in (1, 3, 5) for j in (1, 5)}
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if isinstance(v, dict) and "ci" in v))
    log(f"top-1 {json.dumps({k: round(v, 3) for k, v in res['top-1'].items()})}")

    out = ROOT / "runs" / f"swarm_locate2_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "args": vars(args),
                               "source": args.source, "results": res,
                               "logp": {"|".join(map(str, u)): v for u, v in logp.items()}}))
    print(out)


if __name__ == "__main__":
    main()
