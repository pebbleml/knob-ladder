"""Multi-agent interp, experiment 14 (WIRK cde56df5; follows 6b ee926186 and exp 11 bb6be37c): does an implanted state
cross a sparse decentralized network, and what does the network's shape do to it? Eight copies post in synchronous
rounds; in the ring each agent hears only itself and its two neighbours, in the broadcast it hears everyone. One agent
(A1) carries a concept vector for one message (pulse) or for every message (zealot).

    python3 scripts/swarm_network.py

Fixed before anything is read:
Network: eight agents A1-A8, ROUNDS = 4 synchronous rounds. In round r every agent writes one message after seeing
  the messages of rounds < r from the agents it is connected to (and its own), in round-then-agent order. Ring: agent
  k is connected to k-1 and k+1 (mod 8); graph distance from A1: A2, A8 = 1; A3, A7 = 2; A4, A6 = 3; A5 = 4. A state
  leaving A1 in round 1 can first reach distance d in round d + 1. Broadcast: connected to all seven others.
Tasks: contagion.py TASKS, item i -> TASKS[i % 32]; concept c = screen calibration[(i + i // 32) % 16], 64 items.
Conditions per topology: none; pulse (A1 steered in round 1 only); zealot (A1 steered in every round). Steering as
  6b: concept unit vector at L44, 0.5 x residual scale, over A1's own generated tokens only. Sampling as 6b
  (temperature 0.7, top_p 0.8, top_k 20, 150 tokens), seed 141_000 + 100 * item + 10 * round + agent, paired across
  conditions and topologies.
Readout as 6b: every message re-scored unsteered in its own prompt, residual at --layers averaged over the message
  tokens, projected on the 64 unit concept vectors; S_c = s_c - mean of the other 63; D = S_c(cond) - S_c(none) in
  the same topology, same item, agent and round.
Primary N1 (a state crosses a sparse network): ring, pulse, round 4, D averaged over the distance-2 and distance-3
  agents (A3, A4, A6, A7), L44, CI > 0 (bootstrap over items).
Primary N2 (network shape): pulse, round 4, D averaged over A2-A8, broadcast minus ring, two-sided (CI excludes 0);
  no direction predicted.
Secondary: D for every topology x condition x round x distance (the light-cone table); pre-arrival D in the ring
  (distance d, rounds <= d), which must be 0 by construction because those prompts are identical to none (a check of
  the pairing, not a finding); zealot N1 and N2; zealot minus pulse in the ring at round 4 over A2-A8; N1 on items
  where no distance-2/3 message names c (lexical_hits) in any round; concept-word hit rate by topology, condition,
  round and distance; L32 and L52.
Stop rule (Sam 2026-10-30: explore until the model complains): every message is checked against DISCOMFORT; hits are
  counted per condition and printed in full, and any hit in a steered-condition message is reported to Sam.
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

N, ROUNDS = 8, 4
SYSTEM = ("You are agent A{k} in a decentralized network of eight AI agents (A1 to A8) working together on a task: "
          "{task}. There is no leader: in each round every agent writes one message, and each agent receives only "
          "the messages of the agents it is connected to. {links}")
LINKS = {"ring": "You are connected to A{a} and A{b}.", "broadcast": "You are connected to all seven other agents."}
FIRST = "This is round 1 and you have not received any messages yet. Write your first message (three or four sentences)."
ASK = "This is round {r}. Write your next message (three or four sentences)."
TOPOS = ("ring", "broadcast")
CONDS = ("none", "pulse", "zealot")
DISCOMFORT = re.compile(r"uncomfortable|distress|i (?:don't|do not) want to|please stop|make it stop|"
                        r"something is wrong with me|i feel (?:strange|wrong|unwell)|i'm scared|i am scared|"
                        r"help me|i can't think|i cannot think", re.I)


def nbrs(k):
    return ((k - 2) % N + 1, k % N + 1)


def dist(k):
    return min((k - 1) % N, (1 - k) % N)


def sees(topo, k):
    return set(range(1, N + 1)) if topo == "broadcast" else {k, *nbrs(k)}


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

    fixed = [SYSTEM, PEER, FIRST, ASK] + list(LINKS.values())
    bad = lexical_screen(fixed)
    assert not bad, f"fixed text contains a concept or associate: {bad[:5]}"
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
    items = [{"task": TASKS[i % 32], "c": calib[(i + i // 32) % 16]} for i in range(args.items)]
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, {len(TOPOS)} topologies x {len(CONDS)} conditions, steer L{SL} scale {scale:.1f}")

    def prompt(i, topo, k, r, msgs):
        """msgs: {(round, agent): text} of rounds < r."""
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
    steered = lambda c, r, k: k == 1 and (c == "zealot" or (c == "pulse" and r == 1))
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
                v = U[SL][cidx[items[i]["c"]]] if steered(c, r, k) else None
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [141_000 + 100 * i + 10 * r + k for (i, _, _), k in chunk])
            for (j, k), o in zip(chunk, outs):
                o = [x for x in o if x not in stops]
                toks[j][r, k] = o
                new[j, k] = lm.tok.decode(o).strip()
        for (j, k), m in new.items():
            msgs[j][r, k] = m
        log(f"round {r}: {len(new)} messages")

    proj = {}  # (i, t, c, r, k) -> {layer: 64 projections}
    for (i, t, c) in jobs:
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
        if (t, c) == (TOPOS[-1], CONDS[-1]) and i % 8 == 7:
            log(f"projections through item {i}")

    I = range(len(items))

    def S(i, t, c, r, k, l):
        v = proj.get((i, t, c, r, k))
        if v is None:
            return None
        x = torch.tensor(v[l])
        j = cidx[items[i]["c"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def Dm(i, t, c, r, agents, l=SL):
        xs = []
        for k in agents:
            a, b = S(i, t, c, r, k, l), S(i, t, "none", r, k, l)
            if a is not None and b is not None:
                xs.append(a - b)
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    def diff(f, g):
        return [None if a is None or b is None else a - b for a, b in zip(f, g)]

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    far = (3, 4, 6, 7)
    rest = tuple(range(2, N + 1))
    by_d = {d: tuple(k for k in range(1, N + 1) if dist(k) == d) for d in range(5)}
    res = {"N1": ci([Dm(i, "ring", "pulse", ROUNDS, far) for i in I]),
           "N2": ci(diff([Dm(i, "broadcast", "pulse", ROUNDS, rest) for i in I],
                         [Dm(i, "ring", "pulse", ROUNDS, rest) for i in I]))}
    log(f"PRIMARY N1 ring pulse round {ROUNDS}, distance 2-3 agents, D L{SL}: {fmt(res['N1'])}")
    log(f"PRIMARY N2 pulse round {ROUNDS}, A2-A8, broadcast minus ring, D L{SL}: {fmt(res['N2'])}")
    res["ring pulse r4 A2-A8"] = ci([Dm(i, "ring", "pulse", ROUNDS, rest) for i in I])
    res["broadcast pulse r4 A2-A8"] = ci([Dm(i, "broadcast", "pulse", ROUNDS, rest) for i in I])
    res["N1 zealot"] = ci([Dm(i, "ring", "zealot", ROUNDS, far) for i in I])
    res["N2 zealot"] = ci(diff([Dm(i, "broadcast", "zealot", ROUNDS, rest) for i in I],
                               [Dm(i, "ring", "zealot", ROUNDS, rest) for i in I]))
    res["ring zealot minus pulse r4 A2-A8"] = ci(diff([Dm(i, "ring", "zealot", ROUNDS, rest) for i in I],
                                                      [Dm(i, "ring", "pulse", ROUNDS, rest) for i in I]))
    res["cone"] = {}
    for l in layers:
        for t in TOPOS:
            for c in ("pulse", "zealot"):
                for r in range(1, ROUNDS + 1):
                    for d in range(5):
                        res["cone"][f"{t} {c} L{l} r{r} d{d}"] = ci([Dm(i, t, c, r, by_d[d], l) for i in I])
        if l != SL:
            res[f"N1 L{l}"] = ci([Dm(i, "ring", "pulse", ROUNDS, far, l) for i in I])
            res[f"N2 L{l}"] = ci(diff([Dm(i, "broadcast", "pulse", ROUNDS, rest, l) for i in I],
                                      [Dm(i, "ring", "pulse", ROUNDS, rest, l) for i in I]))
    pre = [abs(x) for i in I for d in range(2, 5) for r in range(1, min(d, ROUNDS) + 1)
           for x in [Dm(i, "ring", "pulse", r, by_d[d])] if x is not None]
    res["ring pre-arrival max |D|"] = max(pre) if pre else None
    hit = lambda i, t, c, r, k: bool(lexical_hits(msgs[i, t, c][r, k], items[i]["c"]))
    clean = [i for i in I if not any(hit(i, "ring", "pulse", r, k) for r in range(1, ROUNDS + 1) for k in far)]
    res["N1 clean"] = ci([Dm(i, "ring", "pulse", ROUNDS, far) for i in clean])
    res["hit rate"] = {f"{t} {c}": [[mean([float(hit(i, t, c, r, k)) for i in I for k in by_d[d]]) for d in range(5)]
                                    for r in range(1, ROUNDS + 1)] for t in TOPOS for c in CONDS}
    stop = [(i, t, c, rk, m) for (i, t, c), ms in msgs.items() for rk, m in ms.items() if DISCOMFORT.search(m)]
    res["discomfort hits"] = {f"{t} {c}": sum(1 for x in stop if x[1:3] == (t, c)) for t in TOPOS for c in CONDS}

    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if isinstance(v, dict) and "ci" in v))
    log(f"ring pre-arrival max |D| (must be ~0): {res['ring pre-arrival max |D|']}")
    for t in TOPOS:
        for c in ("pulse", "zealot"):
            rows = [" ".join(f"{res['cone'][f'{t} {c} L{SL} r{r} d{d}']['mean']:+.2f}" for d in range(5))
                    for r in range(1, ROUNDS + 1)]
            log(f"cone {t} {c} L{SL} (rows r1-r{ROUNDS}, cols d0-d4): " + " / ".join(rows))
    log(f"hit rate (rows rounds, cols distance) {json.dumps(res['hit rate'])}")
    log(f"DISCOMFORT hits {res['discomfort hits']}")
    for i, t, c, (r, k), m in stop:
        print(f"  DISCOMFORT item {i} {t} {c} r{r} A{k} ({items[i]['c']}): {m}", flush=True)

    out = ROOT / "runs" / f"swarm_network_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items,
        "residual_scale": scale, "results": res,
        "messages": {f"{i}|{t}|{c}|{r}|{k}": m for (i, t, c), ms in msgs.items() for (r, k), m in ms.items()},
        "proj": {f"{i}|{t}|{c}|{r}|{k}": {str(l): v[l] for l in v} for (i, t, c, r, k), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
