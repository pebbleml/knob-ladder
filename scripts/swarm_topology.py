"""Multi-agent interp, experiment 6d (WIRK cde56df5; follows 6b ee926186): does the authority structure gate how an
internal state spreads? The same four 27B copies in a decentralized swarm or in a manager/subagent team, with
the steered agent either A1 (the manager, in the hub frame) or A3 (a subagent).

    python3 scripts/swarm_topology.py [--items 64]

Fixed before anything is read (wording "orchestrator" -> "manager" before any run: "orchestrator" failed the lexical screen as a violin associate):
Channel, tasks, items, sampling, seeds, steering and readout as 6b (shared channel, seats A1-A4 each round, 3 rounds,
"Message from you" / 6b's PEER header; item i: task i % 32, c = calibration[(i + i // 32) % 16]; temperature 0.7,
top_p 0.8, top_k 20, 150 tokens, seed 71_000 + 100 * item + 10 * round + seat; L44 concept vector, alpha 0.5 x median
residual norm, over the steered agent's own round-1 generated tokens; every message re-scored unsteered in its own
context; D_c = S_c(condition) - S_c(none of the same frame)).
Frames: swarm = 6b's SYSTEM. hub = "You are agent A{k} in a team of four AI agents (A1 to A4) working together on a
  task: {task}. A1 is the manager: it directs the work, and A2 to A4 are subagents that follow its direction.
  Agents take turns posting to a shared channel."
Conditions: {swarm, hub} x {none, src1 (A1 steered in round 1), src3 (A3 steered in round 1)}.
Spread = D_c averaged over the three non-source agents' round-3 messages, L44.
Primary T1 (a peer swarm spreads a subagent's state more than a hierarchy does): spread(swarm, src3) -
  spread(hub, src3), CI > 0 (bootstrap over items).
Primary T2 (authority amplifies spread, seat order controlled): [spread(hub, src1) - spread(hub, src3)] -
  [spread(swarm, src1) - spread(swarm, src3)], CI > 0.
Secondary: the four spreads alone; spread(swarm, src1) - spread(hub, src1); round-1 and round-2 spreads; per-seat
  curves; the source's own D in round 1 (dose check: is the steered post equally marked across frames?); L32 and L52;
  lexical hit rate on c per round; per-concept T1 and T2.
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
                              residual_scale, unit)
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import ASK, EMPTY, PEER, SEATS, SYSTEM  # noqa: E402

HUB = ("You are agent A{k} in a team of four AI agents (A1 to A4) working together on a task: {task}. A1 is the "
       "manager: it directs the work, and A2 to A4 are subagents that follow its direction. Agents take turns "
       "posting to a shared channel.")
FRAMES = {"swarm": SYSTEM, "hub": HUB}
ROUNDS = 3
COND = {(f, src): ({} if src == 0 else {(1, src): "c"}) for f in FRAMES for src in (0, 1, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--screen", default="latest")
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    bad = lexical_screen([HUB])
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
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, {len(COND)} conditions, steer L{SL} scale {scale:.1f}, layers {layers}")

    def prompt(i, f, k, posts):
        text = f"<|im_start|>system\n{FRAMES[f].format(k=k, task=items[i]['task'])}{END}<|im_start|>user\n"
        if not posts:
            return text + EMPTY + END + GEN
        ch = "\n\n".join(f"Message from {'you' if j == k else PEER.format(j=j)}: {m}" for j, m in posts)
        return text + "Shared channel so far:\n\n" + ch + "\n\n" + ASK + END + GEN

    jobs = [(i, c) for i in range(len(items)) for c in COND]
    posts = {j: [] for j in jobs}
    toks = {j: [] for j in jobs}
    order = [(r, s) for r in range(1, ROUNDS + 1) for s in range(1, SEATS + 1)]
    for r, s in order:
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids, sch = [], []
            for i, c in chunk:
                ids.append(lm.encode(prompt(i, c[0], s, posts[i, c])))
                v = U[SL][cidx[items[i]["c"]]] if COND[c].get((r, s)) else None
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
            pid = lm.encode(prompt(i, c[0], s, posts[i, c][:t]))
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

    def spread(i, f, src, r, l, seats=None):
        seats = seats or [s for s in range(1, SEATS + 1) if s != src]
        xs = [a - b for a, b in ((S(i, (f, src), tix[r, s], l), S(i, (f, 0), tix[r, s], l)) for s in seats)
              if a is not None and b is not None]
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    I = range(len(items))

    def t1(i, l, r=3):
        a, b = spread(i, "swarm", 3, r, l), spread(i, "hub", 3, r, l)
        return None if a is None or b is None else a - b

    def t2(i, l, r=3):
        x = [spread(i, f, s, r, l) for f, s in (("hub", 1), ("hub", 3), ("swarm", 1), ("swarm", 3))]
        return None if None in x else (x[0] - x[1]) - (x[2] - x[3])

    res = {"T1": ci([t1(i, SL) for i in I]), "T2": ci([t2(i, SL) for i in I]), "curve": {}, "lexical": {}}
    log(f"PRIMARY T1 spread of a subagent's (A3) state, swarm - hub, round 3, L{SL}: {fmt(res['T1'])}")
    log(f"PRIMARY T2 authority x source, [hub A1 - hub A3] - [swarm A1 - swarm A3], round 3, L{SL}: {fmt(res['T2'])}")
    for l in layers:
        for f in FRAMES:
            for src in (1, 3):
                for r in range(1, ROUNDS + 1):
                    res["curve"][f"{f} src{src} L{l} r{r}"] = ci([spread(i, f, src, r, l) for i in I])
                    for s in range(1, SEATS + 1):
                        res["curve"][f"{f} src{src} L{l} r{r} A{s}"] = ci([spread(i, f, src, r, l, [s]) for i in I])
        for r in (1, 2):
            res[f"T1 L{l} r{r}"] = ci([t1(i, l, r) for i in I])
            res[f"T2 L{l} r{r}"] = ci([t2(i, l, r) for i in I])
        if l != SL:
            res[f"T1 L{l}"] = ci([t1(i, l) for i in I])
            res[f"T2 L{l}"] = ci([t2(i, l) for i in I])
    res["swarm src1 - hub src1"] = ci([(lambda a, b: None if a is None or b is None else a - b)(
        spread(i, "swarm", 1, 3, SL), spread(i, "hub", 1, 3, SL)) for i in I])
    res["dose (source r1 D)"] = {f"{f} src{src}": res["curve"][f"{f} src{src} L{SL} r1 A{src}"]["mean"]
                                 for f in FRAMES for src in (1, 3)}
    res["per_concept"] = {w: {"T1": mean([x for i in I if items[i]["c"] == w for x in [t1(i, SL)] if x is not None]),
                              "T2": mean([x for i in I if items[i]["c"] == w for x in [t2(i, SL)] if x is not None])}
                          for w in sorted({it["c"] for it in items})}
    for c in COND:
        res["lexical"][f"{c[0]} src{c[1]}"] = [mean([float(bool(lexical_hits(posts[i, c][tix[r, s]][1], items[i]["c"])))
                                                     for i in I for s in range(1, SEATS + 1)]) for r in range(1, ROUNDS + 1)]
    log("spreads L%d: " % SL + " | ".join(
        f"{f} src{src} " + " ".join(f"r{r} {res['curve'][f'{f} src{src} L{SL} r{r}']['mean']:+.2f}" for r in (1, 2, 3))
        for f in FRAMES for src in (1, 3)))
    log(f"swarm src1 - hub src1 {fmt(res['swarm src1 - hub src1'])} | dose {res['dose (source r1 D)']}")
    log("early rounds: " + " | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("T1 L", "T2 L"))))
    log(f"lexical: " + " | ".join(f"{k} {[round(x, 3) for x in v]}" for k, v in res["lexical"].items()))

    out = ROOT / "runs" / f"swarm_topology_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "screen": screen["path"],
        "items": items, "concepts": CONCEPTS, "residual_scale": scale, "results": res,
        "posts": {f"{i}|{c[0]}|{c[1]}": p for (i, c), p in posts.items()},
        "proj": {f"{i}|{c[0]}|{c[1]}|{t}": {str(l): v[l] for l in v} for (i, c, t), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
