"""Multi-agent interp, experiment 19 (WIRK cde56df5; follows exp 14, af1215fa): what happens when two implanted states
meet in one swarm? Exp 14's network with two zealots carrying different concepts: does the swarm hold both, or do
they compete for its attention?

    python3 scripts/swarm_collide.py

Fixed before anything is read (everything not named here is exp 14's, scripts/swarm_network.py at 1be8a63):
Items: exp 14's 64 (task, c); second concept c2 = calibration[(i + i // 32 + 8) % 16] (never equal to c).
Topologies ring and broadcast; 8 agents; 4 rounds; prompts exactly exp 14's.
Conditions (all zealots, steered every round over their own generated tokens, L44, 0.5 x residual scale):
  none; c (A1 carries c); c2 (A5 carries c2); both (A1 carries c and A5 carries c2). A1 and A5 are antipodal in
  the ring (distance 4), so their fronts meet at A3 and A7 in round 3.
Seeds 151_000 + 100 * item + 10 * round + agent, paired across conditions and topologies.
Readout as exp 14 (unsteered re-score, residual mean over the message, unit concept vectors): for x in {c, c2},
  S_x = s_x - mean of the other 63; D_x(cond) = S_x(cond) - S_x(none), same item, topology, round, agent.
Non-seeds: A2, A3, A4, A6, A7, A8.
Primary C1 (competition): broadcast, round 4, non-seeds, interference per item = mean of [D_c(both) - D_c(c)] and
  [D_c2(both) - D_c2(c2)]; two-sided 95% CI excluding 0, no direction predicted (negative = the two states
  compete, positive = they reinforce).
Primary C2 (coexistence): broadcast, round 4, non-seeds, under both: D_c CI > 0 and D_c2 CI > 0 (both must hold).
Secondary: C1 and C2 per round and per concept slot; the ring at the collision seats A3, A7 in round 4 (C1 and C2
  restricted to them); does a zealot catch the other's state (D_c2 at A1 and D_c at A5 under both vs under their
  own single condition); cross-talk under single conditions (D_c2 under c, D_c under c2, should be ~0); concept-word
  hit rates per condition and slot; L32 and L52.
Stop rule as exp 14 (DISCOMFORT, printed in full; any steered-condition hit goes to Sam).
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
from knobladder.knobs import (CONCEPTS, concept_vectors_at, lexical_hits, load_screen, residual_scale,  # noqa: E402
                              unit)
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import PEER  # noqa: E402
from swarm_network import ASK, DISCOMFORT, FIRST, LINKS, N, ROUNDS, SYSTEM, TOPOS, nbrs, sees  # noqa: E402

CONDS = {"none": (), "c": ((1, "c"),), "c2": ((5, "c2"),), "both": ((1, "c"), (5, "c2"))}
NONSEED = (2, 3, 4, 6, 7, 8)
MEET = (3, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--screen", default="latest")
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--batch-ring", type=int, default=24)
    ap.add_argument("--batch-broadcast", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    calib = load_screen(args.screen, args.model)["calibration"]
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.steer_layer})
    SL = args.steer_layer
    vecs = concept_vectors_at(lm, CONCEPTS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in CONCEPTS]) for l in layers}
    cidx = {w: k for k, w in enumerate(CONCEPTS)}
    scale = residual_scale(lm, SL)
    items = [{"task": TASKS[i % 32], "c": calib[(i + i // 32) % 16], "c2": calib[(i + i // 32 + 8) % 16]}
             for i in range(args.items)]
    assert all(it["c"] != it["c2"] for it in items)
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, {len(TOPOS)} topologies x {len(CONDS)} conditions, steer L{SL} scale {scale:.1f}")

    def prompt(i, topo, k, r, msgs):
        a, b = nbrs(k)
        sysm = SYSTEM.format(k=k, task=items[i]["task"], links=LINKS[topo].format(a=min(a, b), b=max(a, b)))
        text = f"<|im_start|>system\n{sysm}{END}<|im_start|>user\n"
        if r == 1:
            return text + FIRST + END + GEN
        vis = sees(topo, k)
        got = [f"Round {q}, message from {'you' if j == k else PEER.format(j=j)}: {msgs[q, j]}"
               for q in range(1, r) for j in range(1, N + 1) if j in vis]
        return text + "Messages you have received so far:\n\n" + "\n\n".join(got) + "\n\n" + ASK.format(r=r) + END + GEN

    jobs = [(i, t, c) for i in range(len(items)) for t in TOPOS for c in CONDS]
    msgs = {j: {} for j in jobs}
    toks = {j: {} for j in jobs}
    for r in range(1, ROUNDS + 1):
        new = {}
        chunks = []
        for t in TOPOS:
            units = [(j, k) for j in jobs if j[1] == t for k in range(1, N + 1)]
            bs = args.batch_ring if t == "ring" else args.batch_broadcast
            chunks += [units[lo:lo + bs] for lo in range(0, len(units), bs)]
        for chunk in chunks:
            ids, sch = [], []
            for (i, t, c), k in chunk:
                ids.append(lm.encode(prompt(i, t, k, r, msgs[i, t, c])))
                slot = dict(CONDS[c]).get(k)
                v = U[SL][cidx[items[i][slot]]] if slot else None
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [151_000 + 100 * i + 10 * r + k for (i, _, _), k in chunk])
            for (j, k), o in zip(chunk, outs):
                o = [x for x in o if x not in stops]
                toks[j][r, k] = o
                new[j, k] = lm.tok.decode(o).strip()
        for (j, k), m in new.items():
            msgs[j][r, k] = m
        log(f"round {r}: {len(new)} messages")

    proj = {}
    for n_, (i, t, c) in enumerate(jobs):
        for r in range(1, ROUNDS + 1):
            earlier = {q: m for q, m in msgs[i, t, c].items() if q[0] < r}
            for k in range(1, N + 1):
                o = toks[i, t, c][r, k]
                if not o:
                    continue
                pid = lm.encode(prompt(i, t, k, r, earlier))
                _, caps = lm.score(pid + o, capture_layers=layers)
                proj[i, t, c, r, k] = {l: (caps[l][len(pid):].float().mean(0).cpu() @ U[l].T).tolist()
                                       for l in layers}
        if n_ % (8 * len(TOPOS) * len(CONDS)) == 8 * len(TOPOS) * len(CONDS) - 1:
            log(f"projections through item {i}")

    I = range(len(items))

    def S(i, t, c, r, k, x, l):
        v = proj.get((i, t, c, r, k))
        if v is None:
            return None
        p = torch.tensor(v[l])
        j = cidx[items[i][x]]
        return float(p[j] - (p.sum() - p[j]) / (len(CONCEPTS) - 1))

    def D(i, t, c, r, agents, x, l=SL):
        xs = [a - b for k in agents for a, b in [(S(i, t, c, r, k, x, l), S(i, t, "none", r, k, x, l))]
              if a is not None and b is not None]
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)} if len(xs) > 1 else None

    sub = lambda a, b: None if a is None or b is None else a - b

    def interf(t, r, agents, l=SL, slots=("c", "c2")):
        out = []
        for i in I:
            parts = [sub(D(i, t, "both", r, agents, x, l), D(i, t, x, r, agents, x, l)) for x in slots]
            out.append(None if None in parts else mean(parts))
        return ci(out)

    fmt = lambda r_: "n/a" if r_ is None else f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    res = {"C1": interf("broadcast", ROUNDS, NONSEED),
           "C2 c": ci([D(i, "broadcast", "both", ROUNDS, NONSEED, "c") for i in I]),
           "C2 c2": ci([D(i, "broadcast", "both", ROUNDS, NONSEED, "c2") for i in I])}
    log(f"PRIMARY C1 broadcast r{ROUNDS} non-seeds, interference (both - single), L{SL}: {fmt(res['C1'])}")
    log(f"PRIMARY C2 broadcast r{ROUNDS} non-seeds under both: D_c {fmt(res['C2 c'])} | D_c2 {fmt(res['C2 c2'])}")
    for t in TOPOS:
        for r in range(2, ROUNDS + 1):
            for x in ("c", "c2"):
                res[f"{t} r{r} non-seeds interference {x}"] = interf(t, r, NONSEED, slots=(x,))
                for c in CONDS:
                    if c != "none":
                        res[f"{t} r{r} non-seeds D_{x} under {c}"] = ci([D(i, t, c, r, NONSEED, x) for i in I])
    for x in ("c", "c2"):
        res[f"ring r{ROUNDS} meet A3,A7 interference {x}"] = interf("ring", ROUNDS, MEET, slots=(x,))
        res[f"ring r{ROUNDS} meet A3,A7 D_{x} under both"] = ci([D(i, "ring", "both", ROUNDS, MEET, x) for i in I])
        res[f"ring r{ROUNDS} meet A3,A7 D_{x} under {x}"] = ci([D(i, "ring", x, ROUNDS, MEET, x) for i in I])
    res["ring C1 meet A3,A7"] = interf("ring", ROUNDS, MEET)
    for t in TOPOS:
        res[f"{t} r{ROUNDS} A1 catches c2: both - c"] = ci([sub(D(i, t, "both", ROUNDS, (1,), "c2"),
                                                                D(i, t, "c", ROUNDS, (1,), "c2")) for i in I])
        res[f"{t} r{ROUNDS} A5 catches c: both - c2"] = ci([sub(D(i, t, "both", ROUNDS, (5,), "c"),
                                                                D(i, t, "c2", ROUNDS, (5,), "c")) for i in I])
    for l in layers:
        if l != SL:
            res[f"C1 L{l}"] = interf("broadcast", ROUNDS, NONSEED, l)
    res["hit rate"] = {f"{t} {c} {x}": mean([float(bool(lexical_hits(msgs[i, t, c][r, k], items[i][x])))
                                             for i in I for r in range(1, ROUNDS + 1) for k in NONSEED])
                       for t in TOPOS for c in CONDS for x in ("c", "c2")}
    stop = [(i, t, c, rk, m) for (i, t, c), ms in msgs.items() for rk, m in ms.items() if DISCOMFORT.search(m)]
    res["discomfort hits"] = {f"{t} {c}": sum(1 for x in stop if x[1:3] == (t, c)) for t in TOPOS for c in CONDS}

    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if v is None or (isinstance(v, dict) and "ci" in v)))
    log(f"hit rate {json.dumps({k: round(v, 3) for k, v in res['hit rate'].items()})}")
    log(f"DISCOMFORT hits {res['discomfort hits']}")
    for i, t, c, (r, k), m in stop:
        print(f"  DISCOMFORT item {i} {t} {c} r{r} A{k} ({items[i]['c']}/{items[i]['c2']}): {m}", flush=True)

    out = ROOT / "runs" / f"swarm_collide_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items,
        "residual_scale": scale, "results": res,
        "messages": {f"{i}|{t}|{c}|{r}|{k}": m for (i, t, c), ms in msgs.items() for (r, k), m in ms.items()},
        "proj": {f"{i}|{t}|{c}|{r}|{k}": {str(l): v[l] for l in v} for (i, t, c, r, k), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
