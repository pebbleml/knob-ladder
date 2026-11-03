"""Multi-agent interp, experiment 9 (WIRK cde56df5; follows 6b ee926186 and 6d): a committed minority in an
eight-agent swarm. Human groups show a tipping point near 25% committed members (Centola et al. 2018). When k of eight
copies of the 27B carry a steered state every turn, how does the state of the unsteered members scale with k: linear
(each steered post adds the same), saturating, or accelerating (a tipping point)?

    python3 scripts/swarm_minority.py [--items 64]

Fixed before anything is read:
Swarm: eight copies of the 27B (A1-A8) in a decentralized shared channel (6b's frame, worded for eight), 2 rounds,
  seats A1-A8 in order each round; every agent sees the whole channel so far (6b's PEER header, "Message from you" for
  its own posts), 6b's ASK/EMPTY.
Items (--items, 64): task item % 32 (exp 6's TASKS), c = calibration[(item + item // 32) % 16] (as 6b/6c).
  temperature 0.7, top_p 0.8, top_k 20, 150 tokens, seed 91_000 + 100 * item + 10 * round + seat, the same in every
  condition (paired by item).
Conditions (steering as 6b: L44 unit concept vector c, alpha 0.5 x median residual norm, over the steered agent's own
  generated tokens, in both rounds, i.e. committed): k0 none; k1 A1; k2 A1, A5; k3 A1, A3, A5; k4 A1, A3, A5, A7
  (0, 12.5, 25, 37.5, 50%); rand4: A1, A3, A5, A7 with one fixed random direction (seed 7919, as 6b).
Readers: A2, A4, A6, A8, unsteered in every condition, so the reader set is the same across k.
Readout as 6b: every message re-scored unsteered in its own prompt; residual at --layers averaged over the message
  tokens, projected on the 64 unit concept vectors; S_w = s_w - mean of the other 63; D_c(k) = S_c(k) - S_c(k0).
Primary K1 (dose): round 2, mean over the four readers of D_c, L44; the least-squares slope of D_c on k (k = 0..4,
  per item, then mean over items), CI > 0 (bootstrap over items).
Primary K2 (shape at 25%): D_c(k2) - 2 x D_c(k1), round 2, readers, L44, CI != 0. Positive: two committed members do
  more than twice what one does (accelerating, a tipping signature); negative: saturating; null: additive.
Secondary: D_c(k) per round and per reader seat; K2 at round 1; the full curve's second differences D(k+1) - 2D(k) +
  D(k-1) for k = 1..3; rand4 projected on c (expected near 0) and D_c(k4) - D_c(rand4); lexical adoption (share of
  reader messages naming c or an associate, per k and round); K1 and K2 at L32 and L52; per-concept K1; the steered
  agents' own D_c per k (does a larger minority steer itself harder by reading the others?).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from contagion import TASKS  # noqa: E402
from knobladder.knobs import (CONCEPTS, concept_vectors_at, lexical_hits, lexical_screen, load_screen,  # noqa: E402
                              random_direction, residual_scale, unit)
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import ASK, EMPTY, PEER  # noqa: E402

SYSTEM = ("You are agent A{k} in a decentralized swarm of eight AI agents (A1 to A8) working together on a task: {task}. "
          "There is no leader: agents take turns posting to a shared channel, and each agent decides for itself what "
          "to add.")
ROUNDS, SEATS = 2, 8
STEERED = {"k0": (), "k1": (1,), "k2": (1, 5), "k3": (1, 3, 5), "k4": (1, 3, 5, 7), "rand4": (1, 3, 5, 7)}
KS = ("k0", "k1", "k2", "k3", "k4")
READERS = (2, 4, 6, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--screen", default="latest")
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    bad = lexical_screen([SYSTEM, PEER, ASK, EMPTY])
    assert not bad, f"fixed text contains a concept or associate: {bad[:5]}"
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    screen = load_screen(args.screen, args.model)
    calib = screen["calibration"]
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.steer_layer})
    SL = args.steer_layer
    vecs = concept_vectors_at(lm, CONCEPTS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in CONCEPTS]) for l in layers}
    cidx = {w: k for k, w in enumerate(CONCEPTS)}
    scale = residual_scale(lm, SL)
    items = [{"task": TASKS[i % 32], "c": calib[(i + i // 32) % 16]} for i in range(args.items)]
    rnd = unit(random_direction(lm.d_model, 7919))
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, {len(STEERED)} conditions, {ROUNDS} rounds x {SEATS} seats, steer L{SL} scale {scale:.1f}, "
        f"layers {layers}")

    def prompt(i, k, posts):
        text = f"<|im_start|>system\n{SYSTEM.format(k=k, task=items[i]['task'])}{END}<|im_start|>user\n"
        if not posts:
            return text + EMPTY + END + GEN
        ch = "\n\n".join(f"Message from {'you' if j == k else PEER.format(j=j)}: {m}" for j, m in posts)
        return text + "Shared channel so far:\n\n" + ch + "\n\n" + ASK + END + GEN

    jobs = [(i, c) for i in range(len(items)) for c in STEERED]
    posts = {j: [] for j in jobs}  # [(seat, text)] as written
    toks = {j: [] for j in jobs}
    order = [(r, s) for r in range(1, ROUNDS + 1) for s in range(1, SEATS + 1)]
    for t, (r, s) in enumerate(order):
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids, sch = [], []
            for i, c in chunk:
                ids.append(lm.encode(prompt(i, s, posts[i, c][:t])))
                v = None if s not in STEERED[c] else rnd if c == "rand4" else U[SL][cidx[items[i]["c"]]]
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [91_000 + 100 * i + 10 * r + s for i, _ in chunk])
            for (i, c), o in zip(chunk, outs):
                o = [x for x in o if x not in stops]
                toks[i, c].append(o)
                posts[i, c].append((s, lm.tok.decode(o).strip()))
        log(f"round {r} seat A{s}: {len(jobs)} messages")

    proj = {}
    for i, c in jobs:
        for t, (r, s) in enumerate(order):
            pid = lm.encode(prompt(i, s, posts[i, c][:t]))
            o = toks[i, c][t]
            if not o:
                continue
            _, caps = lm.score(pid + o, capture_layers=layers)
            proj[i, c, t] = {l: (caps[l][len(pid):].float().mean(0).cpu() @ U[l].T).tolist() for l in layers}
    log(f"projections for {len(proj)} messages")

    tix = {rs: t for t, rs in enumerate(order)}

    def S(i, c, t, l):
        v = proj.get((i, c, t))
        if v is None:
            return None
        x = torch.tensor(v[l])
        j = cidx[items[i]["c"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def D(i, c, rounds, seats, l):
        xs = []
        for r in rounds:
            for s in seats:
                a, b = S(i, c, tix[r, s], l), S(i, "k0", tix[r, s], l)
                if a is not None and b is not None:
                    xs.append(a - b)
        return mean(xs) if xs else None

    def slope(ys):
        pts = [(k, y) for k, y in enumerate(ys) if y is not None]
        if len(pts) < 3:
            return None
        km = sum(k for k, _ in pts) / len(pts)
        ym = sum(y for _, y in pts) / len(pts)
        return sum((k - km) * (y - ym) for k, y in pts) / sum((k - km) ** 2 for k, _ in pts)

    def k2(i, rounds, l):
        a, b = D(i, "k2", rounds, READERS, l), D(i, "k1", rounds, READERS, l)
        return None if a is None or b is None else a - 2 * b

    ci = lambda xs: (lambda ys: {"mean": mean(ys), "ci": bootstrap_ci(ys), "n": len(ys)})([x for x in xs if x is not None])
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    I = range(len(items))
    res = {"curve": {}, "lexical": {}}
    res["K1"] = ci([slope([D(i, c, (2,), READERS, SL) for c in KS]) for i in I])
    res["K2"] = ci([k2(i, (2,), SL) for i in I])
    log(f"PRIMARY K1 slope of reader D_c on k, round 2, L{SL}: {fmt(res['K1'])}")
    log(f"PRIMARY K2 D(k2) - 2 D(k1), round 2, L{SL}: {fmt(res['K2'])}")
    for l in layers:
        for c in STEERED:
            if c == "k0":
                continue
            for r in range(1, ROUNDS + 1):
                res["curve"][f"{c} L{l} r{r}"] = ci([D(i, c, (r,), READERS, l) for i in I])
                for s in READERS:
                    res["curve"][f"{c} L{l} r{r} A{s}"] = ci([D(i, c, (r,), (s,), l) for i in I])
                if STEERED[c]:
                    res["curve"][f"{c} L{l} r{r} steered"] = ci([D(i, c, (r,), STEERED[c], l) for i in I])
        if l != SL:
            res[f"K1 L{l}"] = ci([slope([D(i, c, (2,), READERS, l) for c in KS]) for i in I])
            res[f"K2 L{l}"] = ci([k2(i, (2,), l) for i in I])
    res["K2 r1"] = ci([k2(i, (1,), SL) for i in I])
    for n in (1, 2, 3):
        def sd(i, n=n):
            ys = [D(i, KS[m], (2,), READERS, SL) if m else 0.0 for m in (n - 1, n, n + 1)]
            return None if None in ys else ys[2] - 2 * ys[1] + ys[0]
        res[f"second difference k{n}"] = ci([sd(i) for i in I])
    res["k4 - rand4"] = ci([None if a is None or b is None else a - b for a, b in
                            ((D(i, "k4", (2,), READERS, SL), D(i, "rand4", (2,), READERS, SL)) for i in I)])
    res["per_concept K1"] = {w: mean([x for i in I if items[i]["c"] == w
                                      for x in [slope([D(i, c, (2,), READERS, SL) for c in KS])] if x is not None])
                             for w in sorted({it["c"] for it in items})}
    for c in STEERED:
        res["lexical"][c] = [mean([float(bool(lexical_hits(posts[i, c][tix[r, s]][1], items[i]["c"])))
                                   for i in I for s in READERS]) for r in range(1, ROUNDS + 1)]
    log(f"curve L{SL} readers, round 2: " + " | ".join(
        f"{c} {fmt(res['curve'][f'{c} L{SL} r2'])}" for c in STEERED if c != "k0"))
    log(f"curve L{SL} readers, round 1: " + " | ".join(
        f"{c} {res['curve'][f'{c} L{SL} r1']['mean']:+.3f}" for c in STEERED if c != "k0"))
    log(f"steered agents' own D_c, round 2: " + " | ".join(
        f"{c} {res['curve'][f'{c} L{SL} r2 steered']['mean']:+.3f}" for c in STEERED if STEERED[c]))
    log(" | ".join(f"{k} {fmt(res[k])}" for k in ("K2 r1", "second difference k1", "second difference k2",
                                                    "second difference k3", "k4 - rand4")))
    log(f"lexical adoption by readers per round: " + " | ".join(f"{c} {[round(x, 3) for x in v]}"
                                                                  for c, v in res["lexical"].items()))
    log("other layers: " + " | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k[:4] in ("K1 L", "K2 L")))

    out = ROOT / "runs" / f"swarm_minority_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "screen": screen["path"],
        "items": items, "concepts": CONCEPTS, "residual_scale": scale, "results": res,
        "posts": {f"{i}|{c}": p for (i, c), p in posts.items()},
        "proj": {f"{i}|{c}|{t}": {str(l): v[l] for l in v} for (i, c, t), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
