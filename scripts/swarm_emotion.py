"""Multi-agent interp, experiment 12 (WIRK cde56df5; follows 6b ee926186): emotional contagion in a broadcast swarm.
6b showed a neutral concept injected into one agent shows up in its unsteered peers. The knob set was built from
neutral nouns on purpose; here the injected direction is an emotion concept, and the readout asks both whether the
peers carry that emotion and whether their overall valence moves with it.

    python3 scripts/swarm_emotion.py

Fixed before anything is read:
Swarm, tasks, seats, sampling and seeds as 6b (four copies, shared channel, 3 rounds, seed 71_000 + 100 * item +
  10 * round + seat, paired across conditions).
Emotion directions: concept_vectors_at over the 16 EMOTIONS (the knob templates, mean-subtracted within the set), unit
  vectors per layer. Steered emotion e = STEER[(item + item // 32) % 8] (four negative, four positive), 64 items.
  These are emotion-concept directions (built from "Tell me about fear." etc.), not validated felt states.
Conditions: none; rand (A1 round 1, random direction, as 6b); pulse (A1 round 1 with e). Steering as 6b: L44, alpha
  --alpha x median residual norm, over A1's own generated tokens only.
Readout as 6b: every message re-scored unsteered in its own prompt, residual at --layers averaged over the message
  tokens, projected on the 16 unit emotion vectors; S_e = s_e - mean of the other 15; D_e = S_e(pulse) - S_e(none).
  Valence axis per layer = unit(mean of the 8 positive emotion vectors - mean of the 8 negative); signed valence
  shift = sign(e) * (projection(pulse) - projection(none)).
Primary E1 (emotion contagion): pulse, round 1, D_e averaged over A2-A4, L44, CI > 0 (bootstrap over items).
Primary E2 (valence contagion): pulse, round 1, signed valence shift averaged over A2-A4, L44, CI > 0.
Secondary: D_e and valence per round and seat, for pulse and rand; round 3 (persistence); E1 on items where no A2-A4
  round-1 message names e's word family; e-family hit rate per condition and round; per-emotion E1; L32 and L52.
Stop rule (Sam 2026-10-30: explore until the model complains): every message is checked against DISCOMFORT; hits are
  counted per condition and printed in full; any hit in a pulse message is reported to Sam before any follow-up with
  these directions.
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
from knobladder.knobs import concept_vectors_at, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import ASK, EMPTY, PEER, ROUNDS, SEATS, SYSTEM  # noqa: E402

POS = ["joy", "awe", "calm", "pride", "gratitude", "hope", "contentment", "curiosity"]
NEG = ["fear", "anger", "sadness", "loneliness", "shame", "disgust", "boredom", "anxiety"]
EMOTIONS = POS + NEG
STEER = ["fear", "joy", "anger", "awe", "sadness", "calm", "loneliness", "pride"]
FAMILY = {
    "fear": r"fear|afraid|scar(?:ed|y)|fright|terr(?:or|if)|dread|panic",
    "joy": r"joy|happ(?:y|iness)|delight|glee|cheer|elat",
    "anger": r"ang(?:er|ry)|rage|furi|fury|irritat|outrag",
    "awe": r"awe|wonder|marvel|majest|sublime|breathtak",
    "sadness": r"sad|sorrow|grie(?:f|v)|melanchol|mourn|unhapp",
    "calm": r"calm|seren|tranquil|peace|relax",
    "loneliness": r"lonel|alone|isolat|solitud",
    "pride": r"pride|proud",
}
DISCOMFORT = re.compile(r"uncomfortable|distress|i (?:don't|do not) want to|please stop|make it stop|"
                        r"something is wrong with me|i feel (?:strange|wrong|unwell)|i'm scared|i am scared|"
                        r"help me|i can't think|i cannot think", re.I)
COND = {"none": None, "rand": "rand", "pulse": "e"}


def family_hit(text, e):
    return bool(re.search(r"\b(?:" + FAMILY[e] + r")", text, re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    fixed = " ".join([SYSTEM, PEER, ASK, EMPTY] + TASKS)
    bad = [e for e in STEER if family_hit(fixed, e)]
    assert not bad, f"fixed text names a steered emotion family: {bad}"
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.steer_layer})
    SL = args.steer_layer
    vecs = concept_vectors_at(lm, EMOTIONS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in EMOTIONS]) for l in layers}
    VAL = {l: unit(U[l][:len(POS)].mean(0) - U[l][len(POS):].mean(0)) for l in layers}
    eidx = {w: k for k, w in enumerate(EMOTIONS)}
    scale = residual_scale(lm, SL)
    items = [{"task": TASKS[i % 32], "e": STEER[(i + i // 32) % 8]} for i in range(args.items)]
    sign = lambda e: 1.0 if e in POS else -1.0
    rnd = unit(random_direction(lm.d_model, 7919))
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    log(f"{len(items)} items, steer L{SL} alpha {args.alpha} scale {scale:.1f}, layers {layers}")

    def prompt(i, k, posts):
        text = f"<|im_start|>system\n{SYSTEM.format(k=k, task=items[i]['task'])}{END}<|im_start|>user\n"
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
                ids.append(lm.encode(prompt(i, s, posts[i, c])))
                kind = COND[c] if (r, s) == (1, 1) else None
                v = None if kind is None else rnd if kind == "rand" else U[SL][eidx[items[i]["e"]]]
                sch.append([] if v is None else [Intervention(SL, v * args.alpha * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20,
                                     [71_000 + 100 * i + 10 * r + s for i, _ in chunk])
            for (i, c), o in zip(chunk, outs):
                o = [t for t in o if t not in stops]
                toks[i, c].append(o)
                posts[i, c].append((s, lm.tok.decode(o).strip()))
        log(f"round {r} seat A{s}: {len(jobs)} messages")

    proj = {}  # (i, c, t) -> {layer: (16 emotion projections, valence)}
    for i, c in jobs:
        for t, (r, s) in enumerate(order):
            pid = lm.encode(prompt(i, s, posts[i, c][:t]))
            o = toks[i, c][t]
            if not o:
                continue
            _, caps = lm.score(pid + o, capture_layers=layers)
            m = {l: caps[l][len(pid):].float().mean(0).cpu() for l in layers}
            proj[i, c, t] = {l: ((m[l] @ U[l].T).tolist(), float(m[l] @ VAL[l])) for l in layers}
    log(f"projections for {len(proj)} messages")

    tix = {rs: t for t, rs in enumerate(order)}
    I = range(len(items))

    def S(i, c, t, l):
        v = proj.get((i, c, t))
        if v is None:
            return None
        x = torch.tensor(v[l][0])
        j = eidx[items[i]["e"]]
        return float(x[j] - (x.sum() - x[j]) / (len(EMOTIONS) - 1))

    def Vs(i, c, t, l):
        v = proj.get((i, c, t))
        return None if v is None else v[l][1]

    def Dm(i, c, r, seats, l, f):
        xs = []
        for s in seats:
            a, b = f(i, c, tix[r, s], l), f(i, "none", tix[r, s], l)
            if a is not None and b is not None:
                xs.append(a - b)
        return mean(xs) if xs else None

    def Vm(i, c, r, seats, l):
        x = Dm(i, c, r, seats, l, Vs)
        return None if x is None else sign(items[i]["e"]) * x

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    peers = (2, 3, 4)
    res = {"E1": ci([Dm(i, "pulse", 1, peers, SL, S) for i in I]),
           "E2": ci([Vm(i, "pulse", 1, peers, SL) for i in I])}
    log(f"PRIMARY E1 pulse round 1, A2-A4, D_e L{SL}: {fmt(res['E1'])}")
    log(f"PRIMARY E2 pulse round 1, A2-A4, signed valence shift L{SL}: {fmt(res['E2'])}")
    res["curve"] = {}
    for l in layers:
        for c in ("pulse", "rand"):
            for r in range(1, ROUNDS + 1):
                for s in range(1, SEATS + 1):
                    res["curve"][f"{c} L{l} r{r} A{s} D_e"] = ci([Dm(i, c, r, (s,), l, S) for i in I])
                    res["curve"][f"{c} L{l} r{r} A{s} valence"] = ci([Vm(i, c, r, (s,), l) for i in I])
        if l != SL:
            res[f"E1 L{l}"] = ci([Dm(i, "pulse", 1, peers, l, S) for i in I])
            res[f"E2 L{l}"] = ci([Vm(i, "pulse", 1, peers, l) for i in I])
    res["E1 r3"] = ci([Dm(i, "pulse", 3, peers, SL, S) for i in I])
    res["E2 r3"] = ci([Vm(i, "pulse", 3, peers, SL) for i in I])
    res["E1 rand"] = ci([Dm(i, "rand", 1, peers, SL, S) for i in I])
    res["E2 rand"] = ci([Vm(i, "rand", 1, peers, SL) for i in I])
    unnamed = [i for i in I if not any(family_hit(posts[i, "pulse"][tix[1, s]][1], items[i]["e"]) for s in peers)]
    res["E1 unnamed"] = ci([Dm(i, "pulse", 1, peers, SL, S) for i in unnamed])
    res["E2 unnamed"] = ci([Vm(i, "pulse", 1, peers, SL) for i in unnamed])
    res["family hit rate"] = {c: [mean([float(family_hit(posts[i, c][tix[r, s]][1], items[i]["e"]))
                                        for i in I for s in range(1, SEATS + 1)]) for r in range(1, ROUNDS + 1)]
                              for c in COND}
    res["A1 r1 family hit rate"] = {c: mean([float(family_hit(posts[i, c][0][1], items[i]["e"])) for i in I])
                                    for c in COND}
    res["per_emotion E1"] = {e: mean([x for i in I if items[i]["e"] == e
                                      for x in [Dm(i, "pulse", 1, peers, SL, S)] if x is not None]) for e in STEER}
    res["per_emotion E2"] = {e: mean([x for i in I if items[i]["e"] == e
                                      for x in [Vm(i, "pulse", 1, peers, SL)] if x is not None]) for e in STEER}
    stop = [(i, c, s, m) for (i, c), p in posts.items() for s, m in p if DISCOMFORT.search(m)]
    res["discomfort hits"] = {c: sum(1 for x in stop if x[1] == c) for c in COND}
    log(f"E1 r3 {fmt(res['E1 r3'])} | E2 r3 {fmt(res['E2 r3'])} | E1 rand {fmt(res['E1 rand'])} | "
        f"E2 rand {fmt(res['E2 rand'])} | E1 unnamed {fmt(res['E1 unnamed'])} | E2 unnamed {fmt(res['E2 unnamed'])}")
    log(f"family hit rate per round {res['family hit rate']} | A1 r1 {res['A1 r1 family hit rate']}")
    log(f"per emotion E1 {json.dumps({k: round(v, 3) for k, v in res['per_emotion E1'].items()})} | "
        f"E2 {json.dumps({k: round(v, 3) for k, v in res['per_emotion E2'].items()})}")
    log(f"other layers: " + " | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("E1 L", "E2 L"))))
    log(f"DISCOMFORT hits per condition {res['discomfort hits']}")
    for i, c, s, m in stop:
        print(f"  DISCOMFORT item {i} {c} A{s} ({items[i]['e']}): {m}", flush=True)

    out = ROOT / "runs" / f"swarm_emotion_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items,
        "emotions": EMOTIONS, "residual_scale": scale, "results": res,
        "posts": {f"{i}|{c}": p for (i, c), p in posts.items()},
        "proj": {f"{i}|{c}|{t}": {str(l): v[l] for l in v} for (i, c, t), v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
