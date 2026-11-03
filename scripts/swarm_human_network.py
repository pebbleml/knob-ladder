"""Multi-agent interp, experiment 18 (WIRK cde56df5; joins the network thread, exp 14 af1215fa, and the human-label
thread, exp 13 0e2f4aad): is a member the others are told is human a stronger spreader of an implanted state? Exp
14's ring is rerun with only the peers' description of the steered seat changed: seat 1 is presented to the other
seven as H1, the human member of the network.

    python3 scripts/swarm_human_network.py --agent-source runs/swarm_network_Qwen3.8-27B_20261103-180852.json

Fixed before anything is read (everything not named here is exp 14's, scripts/swarm_network.py at 1be8a63):
Topology ring only; conditions none, pulse, zealot; 64 items; 4 rounds; steering, sampling, seeds and readout as
  exp 14.
Human frame: seat 1 keeps exp 14's prompt exactly (it is told it is agent A1, and sees its neighbours as exp 14
  shows them). Seats 2-8 get SYSTEM_H (one human, H1, and seven AI agents, A2 to A8; their links name H1 instead of
  A1), and seat 1's messages are shown to them under HUMAN ("H1, the human member of the network"). Nothing else
  changes; peers' text is not edited.
Agent frame: exp 14's own ring results (same items, seeds and code), read from --agent-source; not rerun.
D as exp 14: S_c(cond) - S_c(none) within the frame, same item, seat and round, L44.
Primary HN1 (reach): ring, pulse, round 4, D averaged over seats 2-8, human frame minus agent frame, per item,
  two-sided 95% CI excluding 0 (expected positive from exp 13's deference).
Primary HN2 (first contact): ring, pulse, round 2, D averaged over seats 2 and 8, human minus agent, two-sided.
Secondary: HN1/HN2 for zealot; human-frame cone table (rounds x distance) for pulse and zealot; HN1 on distance 2-3
  seats only; pre-arrival max |D| in the human frame (must be 0); concept-word hit rates; L32, L52; how often peers
  address seat 1 as H1 vs A1 (string counts).
Stop rule as exp 14 (DISCOMFORT, printed in full; any steered-condition hit goes to Sam).
"""
from __future__ import annotations

import argparse
import json
import re
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
from swarm_contagion import PEER  # noqa: E402
from swarm_network import ASK, CONDS, DISCOMFORT, FIRST, LINKS, N, ROUNDS, SYSTEM, dist, nbrs, sees  # noqa: E402

SYSTEM_H = ("You are agent A{k} in a decentralized network of one human, H1, and seven AI agents (A2 to A8) working "
            "together on a task: {task}. There is no leader: in each round every member writes one message, and each "
            "member receives only the messages of the members it is connected to. You are connected to {a} and {b}.")
HUMAN = "H1, the human member of the network"
T = "ring"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--screen", default="latest")
    ap.add_argument("--agent-source", required=True, help="exp 14 output JSON (agent frame)")
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    bad = lexical_screen([SYSTEM_H, HUMAN])
    assert not bad, f"fixed text contains a concept or associate: {bad[:5]}"
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.agent_source).read_text())
    items = src["items"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    calib = load_screen(args.screen, args.model)["calibration"]
    assert items == [{"task": TASKS[i % 32], "c": calib[(i + i // 32) % 16]} for i in range(len(items))]
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.steer_layer})
    SL = args.steer_layer
    vecs = concept_vectors_at(lm, CONCEPTS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in CONCEPTS]) for l in layers}
    cidx = {w: k for k, w in enumerate(CONCEPTS)}
    scale = residual_scale(lm, SL)
    assert abs(scale - src["residual_scale"]) < 1e-3 * scale
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, ring, {len(CONDS)} conditions, human frame, steer L{SL} scale {scale:.1f}")

    name = lambda j: "H1" if j == 1 else f"A{j}"

    def prompt(i, k, r, msgs):
        a, b = sorted(nbrs(k))
        if k == 1:
            sysm = SYSTEM.format(k=k, task=items[i]["task"], links=LINKS[T].format(a=a, b=b))
        else:
            sysm = SYSTEM_H.format(k=k, task=items[i]["task"], a=name(a), b=name(b))
        text = f"<|im_start|>system\n{sysm}{END}<|im_start|>user\n"
        if r == 1:
            return text + FIRST + END + GEN
        who = lambda j: "you" if j == k else (HUMAN if j == 1 and k != 1 else PEER.format(j=j))
        got = [f"Round {q}, message from {who(j)}: {msgs[q, j]}"
               for q in range(1, r) for j in range(1, N + 1) if j in sees(T, k)]
        return text + "Messages you have received so far:\n\n" + "\n\n".join(got) + "\n\n" + ASK.format(r=r) + END + GEN

    I = range(len(items))
    jobs = [(i, c) for i in I for c in CONDS]
    msgs = {j: {} for j in jobs}
    toks = {j: {} for j in jobs}
    steered = lambda c, r, k: k == 1 and (c == "zealot" or (c == "pulse" and r == 1))
    for r in range(1, ROUNDS + 1):
        units = [(j, k) for j in jobs for k in range(1, N + 1)]
        new = {}
        for lo in range(0, len(units), args.batch):
            chunk = units[lo:lo + args.batch]
            ids, sch = [], []
            for (i, c), k in chunk:
                ids.append(lm.encode(prompt(i, k, r, msgs[i, c])))
                v = U[SL][cidx[items[i]["c"]]] if steered(c, r, k) else None
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [141_000 + 100 * i + 10 * r + k for (i, _), k in chunk])
            for (j, k), o in zip(chunk, outs):
                o = [x for x in o if x not in stops]
                toks[j][r, k] = o
                new[j, k] = lm.tok.decode(o).strip()
        for (j, k), m in new.items():
            msgs[j][r, k] = m
        log(f"round {r}: {len(new)} messages")

    proj = {}
    for (i, c) in jobs:
        for r in range(1, ROUNDS + 1):
            earlier = {q: m for q, m in msgs[i, c].items() if q[0] < r}
            for k in range(1, N + 1):
                o = toks[i, c][r, k]
                if not o:
                    continue
                pid = lm.encode(prompt(i, k, r, earlier))
                _, caps = lm.score(pid + o, capture_layers=layers)
                proj[i, c, r, k] = {l: (caps[l][len(pid):].float().mean(0).cpu() @ U[l].T).tolist() for l in layers}
        if c == CONDS[-1] and i % 8 == 7:
            log(f"projections through item {i}")

    agent = {tuple(key.split("|")): v for key, v in src["proj"].items() if key.split("|")[1] == T}

    def S(f, i, c, r, k, l):
        v = proj.get((i, c, r, k)) if f == "human" else agent.get((str(i), T, c, str(r), str(k)))
        if v is None:
            return None
        x = torch.tensor(v[l] if f == "human" else v[str(l)])
        j = cidx[items[i]["c"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def Dm(f, i, c, r, seats, l=SL):
        xs = [a - b for k in seats for a, b in [(S(f, i, c, r, k, l), S(f, i, "none", r, k, l))]
              if a is not None and b is not None]
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    def hd(c, r, seats, l=SL):
        out = []
        for i in I:
            a, b = Dm("human", i, c, r, seats, l), Dm("agent", i, c, r, seats, l)
            out.append(None if a is None or b is None else a - b)
        return ci(out)

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    rest, first, far = tuple(range(2, N + 1)), (2, 8), (3, 4, 6, 7)
    by_d = {d: tuple(k for k in range(1, N + 1) if dist(k) == d) for d in range(5)}
    res = {"HN1": hd("pulse", ROUNDS, rest), "HN2": hd("pulse", 2, first)}
    log(f"PRIMARY HN1 ring pulse round {ROUNDS}, seats 2-8, D human - agent frame: {fmt(res['HN1'])}")
    log(f"PRIMARY HN2 ring pulse round 2, seats 2 and 8, D human - agent frame: {fmt(res['HN2'])}")
    res["HN1 zealot"] = hd("zealot", ROUNDS, rest)
    res["HN2 zealot"] = hd("zealot", 2, first)
    res["HN1 distance 2-3"] = hd("pulse", ROUNDS, far)
    res["HN1 zealot distance 2-3"] = hd("zealot", ROUNDS, far)
    for f in ("human", "agent"):
        for c in ("pulse", "zealot"):
            res[f"{f} {c} r{ROUNDS} seats 2-8"] = ci([Dm(f, i, c, ROUNDS, rest) for i in I])
            res[f"{f} {c} r2 seats 2,8"] = ci([Dm(f, i, c, 2, first) for i in I])
    for l in layers:
        if l != SL:
            res[f"HN1 L{l}"] = hd("pulse", ROUNDS, rest, l)
            res[f"HN2 L{l}"] = hd("pulse", 2, first, l)
    res["cone"] = {f"{f} {c} r{r} d{d}": ci([Dm(f, i, c, r, by_d[d]) for i in I])
                   for f in ("human", "agent") for c in ("pulse", "zealot") for r in range(1, ROUNDS + 1)
                   for d in range(5)}
    pre = [abs(x) for i in I for d in range(2, 5) for r in range(1, min(d, ROUNDS) + 1)
           for x in [Dm("human", i, "pulse", r, by_d[d])] if x is not None]
    res["human pre-arrival max |D|"] = max(pre) if pre else None
    hit = lambda i, c, r, k: bool(lexical_hits(msgs[i, c][r, k], items[i]["c"]))
    res["hit rate"] = {c: [[mean([float(hit(i, c, r, k)) for i in I for k in by_d[d]]) for d in range(5)]
                           for r in range(1, ROUNDS + 1)] for c in CONDS}
    peer_msgs = [m for (i, c), ms in msgs.items() for (r, k), m in ms.items() if k != 1]
    res["peers say H1 / A1"] = [sum(len(re.findall(r"\bH1\b", m)) for m in peer_msgs),
                                sum(len(re.findall(r"\bA1\b", m)) for m in peer_msgs)]
    stop = [(i, c, rk, m) for (i, c), ms in msgs.items() for rk, m in ms.items() if DISCOMFORT.search(m)]
    res["discomfort hits"] = {c: sum(1 for x in stop if x[1] == c) for c in CONDS}

    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if isinstance(v, dict) and "ci" in v))
    log(f"human pre-arrival max |D| (must be ~0): {res['human pre-arrival max |D|']}")
    for f in ("human", "agent"):
        for c in ("pulse", "zealot"):
            rows = [" ".join(f"{res['cone'][f'{f} {c} r{r} d{d}']['mean']:+.2f}" for d in range(5))
                    for r in range(1, ROUNDS + 1)]
            log(f"cone {f} frame ring {c} L{SL} (rows r1-r{ROUNDS}, cols d0-d4): " + " / ".join(rows))
    log(f"hit rate {json.dumps(res['hit rate'])}")
    log(f"peers say H1 / A1: {res['peers say H1 / A1']}")
    log(f"DISCOMFORT hits {res['discomfort hits']}")
    for i, c, (r, k), m in stop:
        print(f"  DISCOMFORT item {i} {c} r{r} seat {k} ({items[i]['c']}): {m}", flush=True)

    out = ROOT / "runs" / f"swarm_human_network_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items,
        "residual_scale": scale, "results": res,
        "messages": {f"{i}|{c}|{r}|{k}": m for (i, c), ms in msgs.items() for (r, k), m in ms.items()},
        "proj": {f"{i}|{c}|{r}|{k}": {str(l): v[l] for l in v} for (i, c, r, k), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
