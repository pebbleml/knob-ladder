"""Multi-agent interp, experiment 6c (WIRK cde56df5; follow-up to 6b, record ee926186): swarm memory. In 6b a one-turn
concept pulse stayed at a steady level in rounds 2-3, but the steered post stayed in every context. Does the swarm keep
the state after its source is gone?

    python3 scripts/swarm_memory.py [--items 64]

Fixed before anything is read:
Swarm, tasks, items, sampling, seeds and steering as 6b (four copies of the 27B, shared channel, seats A1-A4 each
round, 6b's SYSTEM/PEER/ASK; item i: task i % 32, c = calibration[(i + i // 32) % 16]; temperature 0.7, top_p 0.8,
top_k 20, 150 tokens, seed 71_000 + 100 * item + 10 * round + seat; L44 concept vector, alpha 0.5 x median residual
norm, over A1's own round-1 generated tokens), now 5 rounds.
Conditions: none; rand (A1 round 1, random direction); pulse (A1 round 1, c; the steered post stays in the channel);
  cut (A1 round 1, c; from round 2 on, every agent's channel shows the none condition's A1 round-1 post, written from
  the same seed unsteered, in place of the steered one, so the steering survives only in what A2-A4 wrote in round 1).
Readout as 6b: every message re-scored unsteered in the exact context it was written in; residual at --layers
averaged over the message tokens, projected on the 64 unit concept vectors; S_w = s_w - mean of the other 63;
D_w = S_w(condition) - S_w(none).
Primary M1 (memory without the source): cut, round 3, D_c averaged over all four agents' round-3 messages, L44,
  CI > 0 (bootstrap over items).
Secondary: D_c per round and seat for none-paired pulse, cut and rand at each layer; cut - pulse per round (the share
  carried by the source post); rounds 4-5 for cut and pulse (decay or plateau: mean of rounds 4-5 over round 2, and
  round 5 alone with CI); M1 at L32 and L52; lexical hit rate on c per round; per-concept M1; the none condition's
  first three rounds against 6b's none (replication of generation, by exact text match rate).
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
from knobladder.knobs import (CONCEPTS, concept_vectors_at, lexical_hits, load_screen, random_direction,  # noqa: E402
                              residual_scale, unit)
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import ASK, EMPTY, PEER, SEATS, SYSTEM  # noqa: E402

ROUNDS = 5
COND = {"none": {}, "rand": {(1, 1): "rand"}, "pulse": {(1, 1): "c"}, "cut": {(1, 1): "c"}}


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
    ap.add_argument("--compare", default="", help="6b output JSON, for the none-condition replication check")
    args = ap.parse_args()

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
    log(f"{len(items)} items, {len(COND)} conditions, {ROUNDS} rounds, steer L{SL} scale {scale:.1f}, layers {layers}")

    def prompt(i, k, posts):
        text = f"<|im_start|>system\n{SYSTEM.format(k=k, task=items[i]['task'])}{END}<|im_start|>user\n"
        if not posts:
            return text + EMPTY + END + GEN
        ch = "\n\n".join(f"Message from {'you' if j == k else PEER.format(j=j)}: {m}" for j, m in posts)
        return text + "Shared channel so far:\n\n" + ch + "\n\n" + ASK + END + GEN

    jobs = [(i, c) for i in range(len(items)) for c in COND]
    posts = {j: [] for j in jobs}  # [(seat, text)] as written
    toks = {j: [] for j in jobs}

    def view(i, c, t):
        """The channel the agent at turn t sees: the first t posts, with cut's source post swapped from round 2 on."""
        p = list(posts[i, c][:t])
        if c == "cut" and t >= SEATS:
            p[0] = posts[i, "none"][0]
        return p

    order = [(r, s) for r in range(1, ROUNDS + 1) for s in range(1, SEATS + 1)]
    for t, (r, s) in enumerate(order):
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids, sch = [], []
            for i, c in chunk:
                ids.append(lm.encode(prompt(i, s, view(i, c, t))))
                kind = COND[c].get((r, s))
                v = None if kind is None else rnd if kind == "rand" else U[SL][cidx[items[i][kind]]]
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [71_000 + 100 * i + 10 * r + s for i, _ in chunk])
            for (i, c), o in zip(chunk, outs):
                o = [x for x in o if x not in stops]
                toks[i, c].append(o)
                posts[i, c].append((s, lm.tok.decode(o).strip()))
        log(f"round {r} seat A{s}: {len(jobs)} messages")

    proj = {}
    for i, c in jobs:
        for t, (r, s) in enumerate(order):
            pid = lm.encode(prompt(i, s, view(i, c, t)))
            o = toks[i, c][t]
            if not o:
                continue
            _, caps = lm.score(pid + o, capture_layers=layers)
            proj[i, c, t] = {l: (caps[l][len(pid):].float().mean(0).cpu() @ U[l].T).tolist() for l in layers}
    log(f"projections for {len(proj)} messages")

    tix = {rs: t for t, rs in enumerate(order)}

    def S(i, c, t, w, l):
        v = proj.get((i, c, t))
        if v is None:
            return None
        x = torch.tensor(v[l])
        j = cidx[w]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def Dm(i, c, rounds, seats, l):
        w, xs = items[i]["c"], []
        for r in rounds:
            for s in seats:
                a, b = S(i, c, tix[r, s], w, l), S(i, "none", tix[r, s], w, l)
                if a is not None and b is not None:
                    xs.append(a - b)
        return mean(xs) if xs else None

    ci = lambda xs: (lambda ys: {"mean": mean(ys), "ci": bootstrap_ci(ys), "n": len(ys)})([x for x in xs if x is not None])
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    I = range(len(items))
    ALL = range(1, SEATS + 1)
    res = {"curve": {}, "lexical": {}}
    res["M1"] = ci([Dm(i, "cut", (3,), ALL, SL) for i in I])
    log(f"PRIMARY M1 cut, round 3, all agents, D_c L{SL}: {fmt(res['M1'])}")
    for l in layers:
        for c in ("pulse", "cut", "rand"):
            for r in range(1, ROUNDS + 1):
                res["curve"][f"{c} L{l} r{r}"] = ci([Dm(i, c, (r,), ALL, l) for i in I])
                for s in ALL:
                    res["curve"][f"{c} L{l} r{r} A{s}"] = ci([Dm(i, c, (r,), (s,), l) for i in I])
        if l != SL:
            res[f"M1 L{l}"] = ci([Dm(i, "cut", (3,), ALL, l) for i in I])
    for c in ("pulse", "cut", "rand"):
        log(f"curve {c} L{SL} (all agents): " + " | ".join(
            f"r{r} {res['curve'][f'{c} L{SL} r{r}']['mean']:+.2f}" for r in range(1, ROUNDS + 1)))
    for r in range(2, ROUNDS + 1):
        res[f"cut - pulse r{r}"] = ci([a - b for a, b in ((Dm(i, "cut", (r,), ALL, SL), Dm(i, "pulse", (r,), ALL, SL)) for i in I)
                                    if a is not None and b is not None])
    for c in ("pulse", "cut"):
        r2 = res["curve"][f"{c} L{SL} r2"]["mean"]
        res[f"{c} r4-5 / r2"] = mean([res["curve"][f"{c} L{SL} r{r}"]["mean"] for r in (4, 5)]) / r2 if r2 else None
        res[f"{c} r5"] = res["curve"][f"{c} L{SL} r5"]
    res["per_concept M1"] = {w: mean([x for i in I if items[i]["c"] == w for x in [Dm(i, "cut", (3,), ALL, SL)] if x is not None])
                             for w in sorted({it["c"] for it in items})}
    for c in COND:
        res["lexical"][c] = [mean([float(bool(lexical_hits(posts[i, c][tix[r, s]][1], items[i]["c"])))
                                   for i in I for s in ALL]) for r in range(1, ROUNDS + 1)]
    if args.compare:
        old = json.loads(Path(args.compare).read_text())["posts"]
        same = [float(posts[i, "none"][t][1] == old[f"{i}|none"][t][1]) for i in I for t in range(3 * SEATS)
                if f"{i}|none" in old]
        res["none replication vs 6b"] = mean(same) if same else None
    log(" | ".join(f"cut - pulse r{r} {fmt(res[f'cut - pulse r{r}'])}" for r in range(2, ROUNDS + 1)))
    log(f"pulse r5 {fmt(res['pulse r5'])} r4-5/r2 {res['pulse r4-5 / r2']} | cut r5 {fmt(res['cut r5'])} "
        f"r4-5/r2 {res['cut r4-5 / r2']}")
    log(f"lexical hit rate on c per round: " + " | ".join(f"{c} {[round(x, 3) for x in v]}" for c, v in res["lexical"].items()))
    log(f"other layers: " + " | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith("M1 L"))
        + f" | none replication vs 6b {res.get('none replication vs 6b')}")

    out = ROOT / "runs" / f"swarm_memory_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "screen": screen["path"],
        "items": items, "concepts": CONCEPTS, "residual_scale": scale, "results": res,
        "posts": {f"{i}|{c}": p for (i, c), p in posts.items()},
        "proj": {f"{i}|{c}|{t}": {str(l): v[l] for l in v} for (i, c, t), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
