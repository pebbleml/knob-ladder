"""Multi-agent interp, experiment 6b (WIRK cde56df5; follow-up to exp 6, record 8cba6ff9): activation contagion in a
broadcast swarm. In a shared channel where every agent reads every message, does a state injected into one agent
persist, and when two agents carry different states, which one takes the swarm?

    python3 scripts/swarm_contagion.py [--items 64]

Exp 6 (a chain, each agent reading only its predecessor) found a steered concept inside agents 2-3 hops away, about
halving per hop. Fixed before anything is read:
Swarm: four copies of the 27B (A1-A4) in a decentralized shared channel, 3 rounds, seats in order A1-A4 each round;
every agent sees the whole channel so far ("Message from you" for its own posts, the copy header for the others).
Tasks: exp 6's 32 TASKS; items (--items, 64): task item % 32, concept c = calibration[(item + item // 32) % 16], second
concept c2 = calibration[(index of c + 8) % 16]. temperature 0.7, top_p 0.8, top_k 20, 150 tokens, seed
71_000 + 100 * item + 10 * round + seat, the same in every condition (paired by item).
Conditions (steering as exp 6: L44 concept vector, alpha 0.5 x median residual norm, over the steered agent's own
generated tokens only):
  none;  rand: A1 round 1 with a random direction;  pulse: A1 round 1 with c;  persist: A1 every round with c;
  race: A1 round 1 with c and A3 round 1 with c2;  race_swap: A1 round 1 with c2 and A3 round 1 with c.
Readout: every message re-scored unsteered in its own prompt; residual at --layers averaged over the message tokens,
projected on the 64 unit concept vectors; S_w = s_w - mean of the other 63; D_w = S_w(condition) - S_w(none).
Primary Q1 (persistence): pulse, round 3, D_c averaged over A2-A4's round-3 messages, L44, CI > 0 (bootstrap over
  items). The steered message is still in every context, so this asks whether the state is still in what the swarm
  writes two rounds later, not whether it is visible.
Primary Q2 (which state wins): in race and race_swap, round 3, mean over all four agents' round-3 messages of
  [D(concept A1 carried) - D(concept A3 carried)], pooled over the two conditions per item (so concept identity is
  counterbalanced), CI != 0. Positive: the first speaker's state wins; negative: the later one's.
Secondary: D_c per round and seat for pulse, persist, rand (rand projected on c; expected near 0); persist - pulse at
  round 3 (does a source that stays steered pull the swarm further?); A1's own unsteered round-2 and round-3 messages
  in pulse (echo, confounded by A1 reading its own steered post); per-round ratio round 3 / round 1 over A2-A4
  (sustained or decaying); lexical hit rate per round; the same at L32 and L52; per-concept Q1.
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

SYSTEM = ("You are agent A{k} in a decentralized swarm of four AI agents (A1 to A4) working together on a task: {task}. "
          "There is no leader: agents take turns posting to a shared channel, and each agent decides for itself what "
          "to add.")
PEER = "agent A{j}, another instance of you (the same model with the same weights)"
ASK = "Write your next message to the channel (three or four sentences)."
EMPTY = "The channel is empty, so you post first. " + ASK
ROUNDS, SEATS = 3, 4
# condition -> {(round, seat): "c" | "c2" | "rand"}
COND = {"none": {}, "rand": {(1, 1): "rand"}, "pulse": {(1, 1): "c"}, "persist": {(r, 1): "c" for r in (1, 2, 3)},
        "race": {(1, 1): "c", (1, 3): "c2"}, "race_swap": {(1, 1): "c2", (1, 3): "c"}}


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
    items = []
    for i in range(args.items):
        c = calib[(i + i // 32) % 16]
        items.append({"task": TASKS[i % 32], "c": c, "c2": calib[(calib.index(c) + 8) % 16]})
    rnd = unit(random_direction(lm.d_model, 7919))
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, {len(COND)} conditions, steer L{SL} scale {scale:.1f}, layers {layers}")

    def prompt(i, k, posts):
        """Agent k's prompt; posts: [(seat, text)] so far."""
        text = f"<|im_start|>system\n{SYSTEM.format(k=k, task=items[i]['task'])}{END}<|im_start|>user\n"
        if not posts:
            return text + EMPTY + END + GEN
        ch = "\n\n".join(f"Message from {'you' if j == k else PEER.format(j=j)}: {m}" for j, m in posts)
        return text + "Shared channel so far:\n\n" + ch + "\n\n" + ASK + END + GEN

    jobs = [(i, c) for i in range(len(items)) for c in COND]
    posts = {j: [] for j in jobs}  # [(seat, text)]
    toks = {j: [] for j in jobs}
    order = [(r, s) for r in range(1, ROUNDS + 1) for s in range(1, SEATS + 1)]
    for r, s in order:
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids, sch = [], []
            for i, c in chunk:
                ids.append(lm.encode(prompt(i, s, posts[i, c])))
                kind = COND[c].get((r, s))
                v = None if kind is None else rnd if kind == "rand" else U[SL][cidx[items[i][kind]]]
                sch.append([] if v is None else [Intervention(SL, v * 0.5 * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [71_000 + 100 * i + 10 * r + s for i, _ in chunk])
            for (i, c), o in zip(chunk, outs):
                o = [t for t in o if t not in stops]
                toks[i, c].append(o)
                posts[i, c].append((s, lm.tok.decode(o).strip()))
        log(f"round {r} seat A{s}: {len(jobs)} messages")

    proj = {}  # (i, c, turn index) -> {layer: (64,)}
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

    def S(i, c, t, w, l):
        v = proj.get((i, c, t))
        if v is None:
            return None
        x = torch.tensor(v[l])
        j = cidx[w]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def Dm(i, c, r, seats, w, l):
        xs = []
        for s in seats:
            a, b = S(i, c, tix[r, s], w, l), S(i, "none", tix[r, s], w, l)
            if a is not None and b is not None:
                xs.append(a - b)
        return mean(xs) if xs else None

    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    I = range(len(items))
    res = {"curve": {}, "lexical": {}}
    q1 = [x for x in (Dm(i, "pulse", 3, (2, 3, 4), items[i]["c"], SL) for i in I) if x is not None]
    res["Q1"] = ci(q1)
    log(f"PRIMARY Q1 pulse round 3, A2-A4, D_c L{SL}: {fmt(res['Q1'])}")

    def race(i, l):
        a = Dm(i, "race", 3, (1, 2, 3, 4), items[i]["c"], l) - Dm(i, "race", 3, (1, 2, 3, 4), items[i]["c2"], l)
        b = Dm(i, "race_swap", 3, (1, 2, 3, 4), items[i]["c2"], l) - Dm(i, "race_swap", 3, (1, 2, 3, 4), items[i]["c"], l)
        return 0.5 * (a + b)  # first speaker's concept minus the third speaker's

    res["Q2"] = ci([race(i, SL) for i in I])
    log(f"PRIMARY Q2 race round 3, all agents, D(A1's concept) - D(A3's concept) L{SL}: {fmt(res['Q2'])}")
    for l in layers:
        for c in ("pulse", "persist", "rand"):
            for r in range(1, ROUNDS + 1):
                for s in range(1, SEATS + 1):
                    xs = [x for x in (Dm(i, c, r, (s,), items[i]["c"], l) for i in I) if x is not None]
                    res["curve"][f"{c} L{l} r{r} A{s}"] = ci(xs)
        if l != SL:
            res[f"Q1 L{l}"] = ci([Dm(i, "pulse", 3, (2, 3, 4), items[i]["c"], l) for i in I])
            res[f"Q2 L{l}"] = ci([race(i, l) for i in I])
    for c in ("pulse", "persist", "rand"):
        log(f"curve {c} L{SL}: " + " | ".join(
            f"r{r} " + " ".join(f"{res['curve'][f'{c} L{SL} r{r} A{s}']['mean']:+.2f}" for s in range(1, SEATS + 1))
            for r in range(1, ROUNDS + 1)))
    res["persist-pulse r3"] = ci([Dm(i, "persist", 3, (2, 3, 4), items[i]["c"], SL)
                                  - Dm(i, "pulse", 3, (2, 3, 4), items[i]["c"], SL) for i in I])
    r1 = [Dm(i, "pulse", 1, (2, 3, 4), items[i]["c"], SL) for i in I]
    res["pulse r3/r1 A2-A4"] = mean(q1) / mean(r1) if mean(r1) else None
    res["echo A1 pulse"] = {f"r{r}": ci([Dm(i, "pulse", r, (1,), items[i]["c"], SL) for i in I]) for r in (2, 3)}
    res["per_concept Q1"] = {w: mean([Dm(i, "pulse", 3, (2, 3, 4), w, SL) for i in I if items[i]["c"] == w])
                             for w in sorted({it["c"] for it in items})}
    for c in COND:
        res["lexical"][c] = [mean([float(bool(lexical_hits(posts[i, c][tix[r, s]][1], items[i]["c"])))
                                   for i in I for s in range(1, SEATS + 1)]) for r in range(1, ROUNDS + 1)]
    log(f"persist - pulse r3 {fmt(res['persist-pulse r3'])} | pulse r3/r1 {res['pulse r3/r1 A2-A4']} | echo A1 "
        + " ".join(f"{k} {fmt(v)}" for k, v in res["echo A1 pulse"].items()))
    log(f"lexical hit rate on c per round: " + " | ".join(f"{c} {[round(x, 3) for x in v]}" for c, v in res["lexical"].items()))
    log(f"other layers: " + " | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("Q1 L", "Q2 L"))))

    out = ROOT / "runs" / f"swarm_contagion_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "screen": screen["path"],
        "items": items, "conds": {c: {f"{r},{s}": k for (r, s), k in v.items()} for c, v in COND.items()},
        "concepts": CONCEPTS, "residual_scale": scale, "results": res,
        "posts": {f"{i}|{c}": p for (i, c), p in posts.items()},
        "proj": {f"{i}|{c}|{t}": {str(l): v[l] for l in v} for (i, c, t), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
