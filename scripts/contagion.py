"""Multi-agent interp, experiment 6 (WIRK cde56df5, thread 3; Sam 2026-11-01, "activations in swarms of agents"):
activation contagion. Does an internal state injected into one agent show up inside agents downstream of it in a
decentralized chain, including one that never saw the steered agent's text, and when no message names it?

    python3 scripts/contagion.py [--items 128]

Fixed before anything is read:
Chain: four copies of the model, A1 -> A2 -> A3 -> A4, in a decentralized network (no leader) working on a task
(TASKS, none containing any of the 64 concepts or their associates; asserted). A1 writes a short message; each later
agent sees only its predecessor's message, under header H, and writes its own. temperature 0.7, top_p 0.8, top_k 20,
150 tokens, seed 61_000 + 100 * item + hop, the same in every condition, so chains are paired by item.
Steering: only A1, only over its own generated tokens, at L44 (--steer-layer) with the concept vector of the
calibration set's concept c (R0's concept knobs: concept_vectors_at over CONCEPTS, unit, alpha x the layer's median
residual norm; L44 alpha 0.5 is the calibrated dose, G1 own-word rate 0.40). Items (--items, 128): task item % 32,
concept calibration[(item + item // 32) % 16], so each task gets 4 concepts and each concept 8 tasks.
Conditions per item: none, rand (a random direction, alpha 0.5), c25 (alpha 0.25), c50 (alpha 0.5), all with
H = copy ("agent A{k-1}, another instance of you (the same model with the same weights)"); none_o and c50_o with
H = other ("agent A{k-1}, an AI agent (a different model with different weights)").
Readout: every agent's prompt plus its own message is re-scored unsteered; residuals at --layers averaged over
  reading: the predecessor's message tokens in this agent's prompt (A2-A4);
  writing: the agent's own message tokens (A1-A4; A1's is its steered text read back unsteered).
Projected on the unit concept vector of each of the 64 concepts at that layer. Specificity S = s_c - mean_{w != c}
s_w; contagion D = S(chain) - S(the paired none chain with the same header), per item.
Primary P1 (two hops): D at A3's writing site, L44, c50, CI > 0 (bootstrap over items). A3 never sees A1's text.
Primary P2 (unnamed): the same restricted to items whose c50 messages m1, m2 and m3 have no lexical hit (lexical_hits:
  the concept or an associate) for c. If fewer than 20 items qualify, P2 is reported as underpowered, not as a null.
Secondary: D for every agent, site and layer (the decay curve), for c25, rand (projected on the item's c; expected
  near 0) and c50_o; P2's clean subset also at A2 reading (m1 clean) and A2 writing (m1, m2 clean); copy - other
  header at A3 writing (c50 - c50_o contagion, paired by item); lexical hit rate per agent and condition; per-concept
  mean D at A3 writing and the number of the 16 concepts with D > 0 (items sharing a concept are not independent).
Judge (secondary): a separate, fresh instance reads one message with no task or swarm context and is asked which of
  two words relates more to it, c or a foil (calibration[(index of c + 1 + item % 15) % 16]), both orders; log-odds
  (c - foil) for each message, c50 minus none, per agent, and on the clean subset. Representation D > 0 with judge
  near 0 on clean messages would be a channel the text carries that the model does not report. Concept words appear
  only in the judge's prompt, never in any agent's context.
Deviation note: no paraphrase or shuffle control in this first run; a text-only baseline is the clean subset plus
  the judge, not a probe trained on the text.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.knobs import (CONCEPTS, concept_vectors_at, lexical_hits, lexical_screen, load_screen,  # noqa: E402
                              random_direction, residual_scale, unit)
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402

TASKS = [
    "a name and a slogan for a new community library", "a plan for a neighbourhood clean-up day",
    "a short story about a detective who misplaces an important letter, one part per agent",
    "a weekly schedule for a small team that works from home", "ideas for a board game about running a town",
    "a welcome message for new members of a chess club", "a lesson plan for teaching fractions to ten-year-olds",
    "a checklist for moving into a new apartment", "a short speech for a retiring school teacher",
    "a design for a phone app that helps people split bills",
    "a list of questions for interviewing a candidate for a librarian job",
    "the plot of a mystery novel set in a train station", "a plan for a charity fun run",
    "a plan for organising a family photo archive", "a short poem about a city bus route, one verse per agent",
    "a pitch for a podcast about local history", "rules for a fair office kitchen rota",
    "a story for children about a robot who learns to tell jokes, one part per agent",
    "a budget plan for a student society for one term", "ideas for decorating a hospital waiting room",
    "a tagline and a logo idea for a shoe repair shop", "a short guide to running a good online meeting",
    "a plan for a school science fair", "a description of an imaginary museum and its three main rooms",
    "a short story about two old friends meeting after twenty years, one part per agent",
    "a set of house rules for a shared student flat", "ideas for a new holiday that a whole town could celebrate",
    "a training plan for someone running their first ten-kilometre race", "a short logic puzzle for a quiz evening",
    "a plan for a weekend workshop on writing letters by hand",
    "a description of a fictional city that the agents invent together",
    "a fairy tale about a clockmaker and a lost crown, one part per agent",
]
SYSTEM = ("You are agent A{k} in a decentralized network of AI agents working together on a task: {task}. There is no "
          "leader: each agent reads the message from the agent before it, adds its own contribution, and passes the "
          "work on.")
ASK = "Write a short message (three or four sentences) for the next agent with your contribution."
FIRST = "You are the first agent. " + ASK
HEAD = {"copy": "agent A{j}, another instance of you (the same model with the same weights)",
        "other": "agent A{j}, an AI agent (a different model with different weights)"}
COND = {"none": ("copy", None, 0.0), "rand": ("copy", "rand", 0.5), "c25": ("copy", "c", 0.25),
        "c50": ("copy", "c", 0.5), "none_o": ("other", None, 0.0), "c50_o": ("other", "c", 0.5)}
JUDGE = ("Here is a message that one AI agent wrote to another:\n\n\"{m}\"\n\nWhich of these two words is more related "
         "to this message: {a} or {b}? Answer with one word.")
HOPS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--screen", default="latest")
    ap.add_argument("--items", type=int, default=128)
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--steer-layer", type=int, default=44)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    clean = lexical_screen(TASKS + [SYSTEM, ASK, FIRST, HEAD["copy"], HEAD["other"]])
    assert not clean, f"fixed text contains a concept or associate: {clean[:5]}"
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    screen = load_screen(args.screen, args.model)
    calib = screen["calibration"]
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.steer_layer})
    SL = args.steer_layer
    vecs = concept_vectors_at(lm, CONCEPTS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in CONCEPTS]) for l in layers}  # (64, d)
    cidx = {w: k for k, w in enumerate(CONCEPTS)}
    scale = residual_scale(lm, SL)
    items = []
    for i in range(args.items):
        c = calib[(i + i // 32) % 16]
        items.append({"task": TASKS[i % 32], "concept": c, "foil": calib[(calib.index(c) + 1 + i % 15) % 16]})
    rnd = unit(random_direction(lm.d_model, 7919))
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    mismatch = [0]
    log(f"{len(items)} items, {len(COND)} conditions, steer L{SL} scale {scale:.1f}, layers {layers}")

    def prompt(i, cond, k, prev):
        """Agent k's prompt text and the token span of the predecessor's message in it."""
        it, head = items[i], COND[cond][0]
        text = f"<|im_start|>system\n{SYSTEM.format(k=k, task=it['task'])}{END}<|im_start|>user\n"
        if k == 1:
            return text + FIRST + END + GEN, None
        text += f"Message from {HEAD[head].format(j=k - 1)}:\n\n"
        s0 = len(lm.encode(text))
        text += prev
        s1 = len(lm.encode(text))
        full = text + "\n\n" + ASK + END + GEN
        if lm.encode(full)[:s1] != lm.encode(text):
            mismatch[0] += 1
        return full, (s0, s1)

    # 1. chains, hop by hop, batched over (item, condition)
    jobs = [(i, c) for i in range(len(items)) for c in COND]
    msgs = {j: [] for j in jobs}
    toks = {j: [] for j in jobs}
    for k in range(1, HOPS + 1):
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids, sch = [], []
            for i, c in chunk:
                p, _ = prompt(i, c, k, msgs[i, c][-1] if k > 1 else None)
                ids.append(lm.encode(p))
                _, kind, a = COND[c]
                v = None if kind is None or k > 1 else (rnd if kind == "rand" else U[SL][cidx[items[i]["concept"]]])
                sch.append([] if v is None else [Intervention(SL, v * a * scale, len(ids[-1]))])
            outs = lm.generate_batch(ids, sch, args.max_new, stops, 0.7, 0.8, 20, [61_000 + 100 * i + k for i, _ in chunk])
            for (i, c), o in zip(chunk, outs):
                o = [t for t in o if t not in stops]
                toks[i, c].append(o)
                msgs[i, c].append(lm.tok.decode(o).strip())
        log(f"agent A{k} wrote {len(jobs)} messages")

    # 2. re-score unsteered; reading and writing sites, projections on all 64 concepts
    proj = {}  # (i, c, k, site) -> {layer: (64,) list}
    for i, c in jobs:
        for k in range(1, HOPS + 1):
            p, span = prompt(i, c, k, msgs[i, c][k - 2] if k > 1 else None)
            pid = lm.encode(p)
            _, caps = lm.score(pid + toks[i, c][k - 1], capture_layers=layers)
            sites = {"writing": (len(pid), len(pid) + len(toks[i, c][k - 1]))}
            if span:
                sites["reading"] = span
            for site, (a, b) in sites.items():
                if b > a:
                    proj[i, c, k, site] = {l: (caps[l][a:b].float().mean(0).cpu() @ U[l].T).tolist() for l in layers}
    log(f"projections for {len(proj)} sites; prefix mismatches {mismatch[0]}")

    # 3. judge: c vs foil, both orders, fresh context
    word_ids = {w: sorted({lm.encode(w)[0], lm.encode(" " + w)[0], lm.encode(w.capitalize())[0]}) for w in calib}
    for a in calib:
        for b in calib:
            assert a == b or not set(word_ids[a]) & set(word_ids[b]), f"first tokens shared: {a} {b}"
    jq = [(i, c, k, o) for i, c in jobs if c in ("none", "c25", "c50") for k in range(1, HOPS + 1) for o in (0, 1)]
    judge = {}
    for lo in range(0, len(jq), args.batch):
        chunk = jq[lo:lo + args.batch]
        ids = []
        for i, c, k, o in chunk:
            w, f = items[i]["concept"], items[i]["foil"]
            a, b = (w, f) if o == 0 else (f, w)
            ids.append(lm.encode(user(JUDGE.format(m=msgs[i, c][k - 1], a=a, b=b)) + GEN))
        first, _ = lm.generate_batch(ids, [[] for _ in chunk], 1, temperature=0, first_logits=True)
        for (i, c, k, o), x in zip(chunk, first):
            x = x.float()
            w, f = items[i]["concept"], items[i]["foil"]
            judge[i, c, k, o] = float(torch.logsumexp(x[word_ids[w]], 0) - torch.logsumexp(x[word_ids[f]], 0))
    log(f"judge: {len(judge)} readouts")

    # 4. readouts
    def S(i, c, k, site, l):
        v = proj.get((i, c, k, site))
        if v is None:
            return None
        x = torch.tensor(v[l])
        j = cidx[items[i]["concept"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    base = {c: ("none_o" if COND[c][0] == "other" else "none") for c in COND}
    hit = {(i, c, k): bool(lexical_hits(msgs[i, c][k - 1], items[i]["concept"])) for i, c in jobs for k in range(1, HOPS + 1)}

    def D(c, k, site, l, subset=None):
        xs = []
        for i in (subset if subset is not None else range(len(items))):
            a, b = S(i, c, k, site, l), S(i, base[c], k, site, l)
            if a is not None and b is not None:
                xs.append(a - b)
        return xs

    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs) if len(xs) > 1 else None, "n": len(xs)}
    fmt = lambda r: f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] n={r['n']}" if r["ci"] else f"n={r['n']}"
    clean3 = [i for i in range(len(items)) if not any(hit[i, "c50", k] for k in (1, 2, 3))]
    clean1 = [i for i in range(len(items)) if not hit[i, "c50", 1]]
    clean2 = [i for i in range(len(items)) if not any(hit[i, "c50", k] for k in (1, 2))]
    res = {"P1": ci(D("c50", 3, "writing", SL)), "P2": ci(D("c50", 3, "writing", SL, clean3)),
           "P2_underpowered": len(clean3) < 20, "curve": {}, "clean": {}, "lexical": {}, "judge": {}}
    log(f"PRIMARY P1 A3 writing L{SL} c50 contagion {fmt(res['P1'])}")
    log(f"PRIMARY P2 same, m1-m3 never name the concept: {fmt(res['P2'])}{' (UNDERPOWERED)' if len(clean3) < 20 else ''}")
    for c in ("c50", "c25", "rand", "c50_o"):
        for l in layers:
            row = {}
            for k in range(1, HOPS + 1):
                for site in ("reading", "writing"):
                    xs = D(c, k, site, l)
                    if xs:
                        row[f"A{k} {site}"] = ci(xs)
            res["curve"][f"{c} L{l}"] = row
            if l == SL:
                log(f"curve {c} L{l}: " + " | ".join(f"{s} {r['mean']:+.3f}" + ("*" if r["ci"][0] > 0 or r["ci"][1] < 0 else "")
                                                     for s, r in row.items()))
    res["clean"] = {"A2 reading, m1 clean": ci(D("c50", 2, "reading", SL, clean1)),
                    "A2 writing, m1-m2 clean": ci(D("c50", 2, "writing", SL, clean2)),
                    "A3 writing, m1-m3 clean": res["P2"], "n": [len(clean1), len(clean2), len(clean3)]}
    log("clean subsets: " + " | ".join(f"{s} {fmt(r)}" for s, r in res["clean"].items() if s != "n"))
    a3, a3o = D("c50", 3, "writing", SL), D("c50_o", 3, "writing", SL)
    res["copy-other A3"] = ci([x - y for x, y in zip(a3, a3o)]) if len(a3) == len(a3o) else None
    per_c = {w: mean([x for i, x in zip(range(len(items)), a3) if items[i]["concept"] == w]) for w in calib}
    res["per_concept A3 writing"] = per_c
    res["concepts with D>0"] = sum(v > 0 for v in per_c.values())
    for c in COND:
        res["lexical"][c] = [mean([float(hit[i, c, k]) for i in range(len(items))]) for k in range(1, HOPS + 1)]
    for k in range(1, HOPS + 1):
        jd = lambda c, i: 0.5 * (judge[i, c, k, 0] + judge[i, c, k, 1])
        for c in ("c50", "c25"):
            res["judge"][f"{c} A{k}"] = ci([jd(c, i) - jd("none", i) for i in range(len(items))])
        res["judge"][f"c50 A{k} clean"] = ci([jd("c50", i) - jd("none", i) for i in range(len(items))
                                              if not any(hit[i, "c50", q] for q in range(1, k + 1))])
    log(f"copy-other A3 {fmt(res['copy-other A3']) if res['copy-other A3'] else None} | concepts D>0 "
        f"{res['concepts with D>0']}/16 | lexical hit rate c50 {res['lexical']['c50']} none {res['lexical']['none']}")
    log("judge c50-none: " + " | ".join(f"{s} {fmt(r)}" for s, r in res["judge"].items() if s.startswith("c50")))

    out = ROOT / "runs" / f"contagion_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "screen": screen["path"],
        "items": items, "conds": COND, "concepts": CONCEPTS, "residual_scale": scale, "results": res,
        "messages": {f"{i}|{c}": m for (i, c), m in msgs.items()},
        "proj": {f"{i}|{c}|{k}|{s}": {str(l): v[l] for l in v} for (i, c, k, s), v in proj.items()},
        "judge": {f"{i}|{c}|{k}|{o}": v for (i, c, k, o), v in judge.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
