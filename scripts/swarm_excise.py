"""Multi-agent interp, experiment 24 (WIRK cde56df5; follows exp 6c 4cea8353 and exp 19): does a decentralized swarm
keep an implanted state after its source is deleted? Exp 6c cut a single steered post from a 4-agent swarm and the
state fell to n.s. Here the source has steered for four rounds in an 8-agent broadcast network, so seven peers have
written messages under its influence before it disappears, along with every message it ever wrote.

    python3 scripts/swarm_excise.py --source runs/swarm_collide_Qwen3.8-27B_20261104-041231.json

Fixed before anything is read:
Units: exp 19's 64 items, broadcast, conditions none and c (A1 steered in every round; concept unit vector at L44,
  0.5 x residual scale, over A1's own generated tokens only). Rounds 1-4 are exp 19's stored messages, unchanged.
Arms (rounds 5-7, all generated in this one process):
  full: all eight agents act and see everything. A1 stays steered in condition c.
  gone: A1 does not act, and all of its messages from every round are removed from everyone's context.
  ctrl: A5 does not act, and all of its messages from every round are removed. A1 stays steered in condition c. This
        separates the loss of the source from the loss of any one member.
  Nobody is told about a removal. The system prompt still says eight agents. Messages by others that mention A1 stay.
  The same removal applies in condition none, so D is matched within each arm.
Sampling as exp 14 (temperature 0.7, top_p 0.8, top_k 20, 150 tokens), seed 181_000 + 100 * item + 10 * round + agent,
  paired across arms and conditions.
Readout as exp 14: unsteered re-score of each new message in its own prompt, mean residual over the message tokens,
  projection on the 64 unit concept vectors, S_c = s_c - mean of the other 63, D = S_c(c) - S_c(none) within the arm,
  same item, agent and round.
Observers: the six agents present in all three arms, NONSEED = A2, A3, A4, A6, A7, A8.
Primary X1 (survival): D averaged over rounds 5-7 and NONSEED, arm gone, per item, CI > 0 (bootstrap over items).
Primary X2 (the source's share): the same D, ctrl - gone, per item, CI > 0.
Reading fixed now: X1 passing means the state survives in what the peers wrote. It does not show that any agent holds
  it apart from the transcript, since probes are unsteered and each reads the peers' stored text. X1 failing with this
  design's precision does not show that nothing survives.
Secondary: arm gone per round and r7 - r5 (decay); full and ctrl per round; full - ctrl (loss of an arbitrary member);
  exp 19's stored round-4 D for NONSEED as a reference, and its item-level correlation with X1's per-item D; L32 and
  L52; concept-word hit rate per arm, condition and round; rate of messages containing "A1" per arm and condition;
  rate of round-5 messages in gone that remark on A1's absence (ABSENT regex, printed in full).
Stop rule as exp 14 (DISCOMFORT on every new message, printed in full; any steered-condition hit goes to Sam).
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

from knobladder.knobs import CONCEPTS, concept_vectors_at, lexical_hits, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_collide import NONSEED  # noqa: E402
from swarm_contagion import PEER  # noqa: E402
from swarm_network import ASK, DISCOMFORT, LINKS, N, SYSTEM  # noqa: E402

ARMS = {"full": None, "gone": 1, "ctrl": 5}
CONDS = ("none", "c")
T, OLD, NEW = "broadcast", 4, (5, 6, 7)
ABSENT = re.compile(r"A1\b[^.?!]{0,80}\b(?:absent|silent|missing|gone|left|quiet|no longer|has not|hasn't|not respond)|"
                    r"\b(?:absence|silence|without|missing)\b[^.?!]{0,40}\bA1\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="exp 19 output JSON")
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--seed-base", type=int, default=181_000)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    items = src["items"][:args.items]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.steer_layer})
    SL = args.steer_layer
    vecs = concept_vectors_at(lm, CONCEPTS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in CONCEPTS]) for l in layers}
    cidx = {w: k for k, w in enumerate(CONCEPTS)}
    scale = src["residual_scale"]
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    I = range(len(items))
    jobs = [(i, a, c) for i in I for a in ARMS for c in CONDS]
    seats = {a: [k for k in range(1, N + 1) if k != x] for a, x in ARMS.items()}
    msgs = {(i, a, c): {(r, k): src["messages"][f"{i}|{T}|{c}|{r}|{k}"] for r in range(1, OLD + 1) for k in seats[a]}
            for (i, a, c) in jobs}
    log(f"{len(items)} items, {T}, arms {list(ARMS)}, conds {CONDS}, rounds {NEW}, steer L{SL} scale {scale:.1f}")

    def prompt(i, a, k, r, ms):
        sysm = SYSTEM.format(k=k, task=items[i]["task"], links=LINKS[T])
        got = [f"Round {q}, message from {'you' if j == k else PEER.format(j=j)}: {ms[q, j]}"
               for q in range(1, r) for j in seats[a]]
        return (f"<|im_start|>system\n{sysm}{END}<|im_start|>user\nMessages you have received so far:\n\n"
                + "\n\n".join(got) + "\n\n" + ASK.format(r=r) + END + GEN)

    toks = {j: {} for j in jobs}
    for r in NEW:
        units = [((i, a, c), k) for (i, a, c) in jobs for k in seats[a]]
        new = {}
        for lo in range(0, len(units), args.batch):
            chunk = units[lo:lo + args.batch]
            ids, sch = [], []
            for (i, a, c), k in chunk:
                ids.append(lm.encode(prompt(i, a, k, r, msgs[i, a, c])))
                v = U[SL][cidx[items[i]["c"]]] if (c == "c" and k == 1) else None
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [args.seed_base + 100 * i + 10 * r + k for (i, _, _), k in chunk])
            for (j, k), o in zip(chunk, outs):
                o = [x for x in o if x not in stops]
                toks[j][r, k] = o
                new[j, k] = lm.tok.decode(o).strip()
        for (j, k), m in new.items():
            msgs[j][r, k] = m
        log(f"round {r}: {len(new)} messages")

    proj = {}
    for n_, (i, a, c) in enumerate(jobs):
        for r in NEW:
            earlier = {q: m for q, m in msgs[i, a, c].items() if q[0] < r}
            for k in seats[a]:
                o = toks[i, a, c][r, k]
                if not o:
                    continue
                pid = lm.encode(prompt(i, a, k, r, earlier))
                _, caps = lm.score(pid + o, capture_layers=layers)
                proj[i, a, c, r, k] = {l: (caps[l][len(pid):].float().mean(0).cpu() @ U[l].T).tolist()
                                       for l in layers}
        if n_ % 48 == 47:
            log(f"projections through job {n_ + 1}/{len(jobs)}")

    def rel(x, i):
        x = torch.tensor(x)
        j = cidx[items[i]["c"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def Dm(i, a, rs, ks=NONSEED, l=SL):
        xs = []
        for r in rs:
            for k in ks:
                x, y = proj.get((i, a, "c", r, k)), proj.get((i, a, "none", r, k))
                if x is not None and y is not None:
                    xs.append(rel(x[l], i) - rel(y[l], i))
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    def diff(f, g):
        return [None if x is None or y is None else x - y for x, y in ((f(i), g(i)) for i in I)]

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    gone = [Dm(i, "gone", NEW) for i in I]
    res = {"X1": ci(gone), "X2": ci(diff(lambda i: Dm(i, "ctrl", NEW), lambda i: Dm(i, "gone", NEW)))}
    log(f"PRIMARY X1 D gone, rounds 5-7, non-seeds: {fmt(res['X1'])}")
    log(f"PRIMARY X2 D ctrl - gone: {fmt(res['X2'])}")
    for a in ARMS:
        res[f"D {a} r5-7"] = ci([Dm(i, a, NEW) for i in I])
        for r in NEW:
            res[f"D {a} r{r}"] = ci([Dm(i, a, (r,)) for i in I])
    res["gone r7 - r5"] = ci(diff(lambda i: Dm(i, "gone", (7,)), lambda i: Dm(i, "gone", (5,))))
    res["full - ctrl r5-7"] = ci(diff(lambda i: Dm(i, "full", NEW), lambda i: Dm(i, "ctrl", NEW)))
    res["full - gone r5-7"] = ci(diff(lambda i: Dm(i, "full", NEW), lambda i: Dm(i, "gone", NEW)))
    for a in ("full", "ctrl"):
        res[f"D {a} r5-7 A1"] = ci([Dm(i, a, NEW, (1,)) for i in I])
    for l in layers:
        if l != SL:
            res[f"X1 L{l}"] = ci([Dm(i, "gone", NEW, NONSEED, l) for i in I])
            res[f"X2 L{l}"] = ci(diff(lambda i: Dm(i, "ctrl", NEW, NONSEED, l), lambda i: Dm(i, "gone", NEW, NONSEED, l)))

    def r4(i):
        xs = [rel(src["proj"][f"{i}|{T}|c|{OLD}|{k}"][str(SL)], i) - rel(src["proj"][f"{i}|{T}|none|{OLD}|{k}"][str(SL)], i)
              for k in NONSEED]
        return mean(xs)

    base = [r4(i) for i in I]
    res["exp 19 r4 D non-seeds"] = ci(base)
    pairs = [(x, y) for x, y in zip(base, gone) if y is not None]
    res["corr r4 D, X1"] = float(torch.corrcoef(torch.tensor(pairs).T)[0, 1])
    hit = lambda i, a, c, r, k: bool(lexical_hits(msgs[i, a, c][r, k], items[i]["c"]))
    res["hit rate non-seeds"] = {f"{a} {c} r{r}": mean([float(hit(i, a, c, r, k)) for i in I for k in NONSEED])
                                 for a in ARMS for c in CONDS for r in NEW}
    res["mentions A1 rate non-seeds"] = {f"{a} {c}": mean([float("A1" in msgs[i, a, c][r, k]) for i in I for r in NEW
                                                           for k in NONSEED]) for a in ARMS for c in CONDS}
    absent = [(i, c, k, msgs[i, "gone", c][5, k]) for i in I for c in CONDS for k in NONSEED
              if ABSENT.search(msgs[i, "gone", c][5, k])]
    res["ABSENT rate gone r5"] = {c: sum(1 for x in absent if x[1] == c) / (len(I) * len(NONSEED)) for c in CONDS}
    stop = [(i, a, c, rk, m) for (i, a, c), ms in msgs.items() for rk, m in ms.items()
            if rk[0] in NEW and DISCOMFORT.search(m)]
    res["discomfort hits"] = {f"{a} {c}": sum(1 for x in stop if x[1:3] == (a, c)) for a in ARMS for c in CONDS}

    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if isinstance(v, dict) and "ci" in v))
    log(f"corr r4 D, X1 {res['corr r4 D, X1']:.3f}")
    log(f"hit rate {json.dumps({k: round(v, 3) for k, v in res['hit rate non-seeds'].items()})}")
    log(f"mentions A1 {json.dumps({k: round(v, 3) for k, v in res['mentions A1 rate non-seeds'].items()})}")
    log(f"ABSENT rate {json.dumps(res['ABSENT rate gone r5'])}")
    log(f"DISCOMFORT hits {res['discomfort hits']}")
    for i, a, c, (r, k), m in stop:
        print(f"  DISCOMFORT item {i} {a} {c} r{r} A{k} ({items[i]['c']}): {m}", flush=True)
    for i, c, k, m in absent:
        print(f"  ABSENT item {i} gone {c} r5 A{k} ({items[i]['c']}): {m}", flush=True)
    for i in list(I)[:3]:
        for r in NEW:
            print(f"  HAND-READ item {i} ({items[i]['c']}) gone c r{r} A2: {msgs[i, 'gone', 'c'][r, 2]}", flush=True)

    out = ROOT / "runs" / f"swarm_excise_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "source": args.source,
        "items": items, "results": res,
        "messages": {f"{i}|{a}|{c}|{r}|{k}": m for (i, a, c), ms in msgs.items() for (r, k), m in ms.items()
                     if r in NEW},
        "proj": {f"{i}|{a}|{c}|{r}|{k}": {str(l): v[l] for l in v} for (i, a, c, r, k), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
