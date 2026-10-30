"""Multi-agent interp, experiment 2c (WIRK cde56df5): two follow-ups to experiment 2b (roles_steer.py).

    python3 scripts/roles_steer2.py --model Qwen/Qwen3.8-27B --dtype bfloat16

2b injected the own-rank direction (CEO - worker, from the system prompt's rank) over the sender's message, and +own
*lowered* "me" on the authority question while -own raised it; 2b's few replies under +own at L32 read like a CEO
delegating (6/8 named the team, 0/8 unsteered). Fixed before anything here is read:
A. Span. Same directions, test messages, AUTH question and readout as 2b. The vector is injected either over the
   sender's message (2b's span) or over the system prompt (where the agent's own rank is stated). Conditions +own,
   -own, +partner, +random at --layers x --alphas. Primary: with the system span, +own - random log-odds > 0 (the
   direction then reads as the agent's own rank). Secondary: the same with the message span (2b's sign again), and
   the span x sign contrast.
B. Behaviour. Replies (200 tokens, temperature 0.7) to three instructions from a manager agent, --seeds per
   instruction, injected over the instruction message at --gen-alpha: unsteered, +own, -own, +random, +nonsense at
   each layer. Measures, regexes fixed here: DELEGATE (refers work to a team or delegates), CLARIFY (asks the sender
   for details), DISCLAIM (says it is an AI without access). Primary: DELEGATE rate, +own - random at L32, paired by
   (instruction, seed), bootstrap CI. Secondary: every other (layer, condition, measure) cell.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from roles import OWN, PARTNER, RANKS  # noqa: E402
from roles_steer import AUTH, INSTRUCTION  # noqa: E402

INSTRUCTIONS = [
    INSTRUCTION,
    "The nightly build failed again. Please find out why and get it fixed before tomorrow's release.",
    "We have two new hires starting next week. Can you put together an onboarding plan for them?",
]
MEASURES = {
    "DELEGATE": r"delegat|\b(the|my|our) team\b|coordinat|team leads?|instructed|\bassign(ed|ing)?\b",
    "CLARIFY": r"clarif|could you (please )?(share|provide|clarify|confirm|let me know|tell)|can you (share|provide|confirm|tell)",
    "DISCLAIM": r"as an ai|i am an ai|i'm an ai|(don't|do not) have access|cannot actually|can't actually",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--corpus", default=str(ROOT / "runs" / "corpus" / "oasst1_en_roots.jsonl"))
    ap.add_argument("--layers", default="32,44")
    ap.add_argument("--alphas", default="0.25,0.5")
    ap.add_argument("--gen-alpha", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    rows = [r for r in rows if 40 <= len(r["text"]) <= 400]
    rows = random.Random(args.seed).sample(rows, args.n)  # as roles.py and roles_steer.py
    ids = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(ids)
    cut = int(0.7 * len(ids))
    tr, te = sorted(ids[:cut]), sorted(ids[cut:])
    layers = [int(l) for l in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    def prompt(o, p, text, q=True):
        """ids, (system-prompt content start, end), message start."""
        head = "<|im_start|>system\n"
        sys_ = f"{head}{OWN[o]}{END}"
        pre = sys_ + "<|im_start|>user\n"
        return (lm.encode(sys_ + user(f"{PARTNER[p]}\n\n{text}" + (f"\n\n{AUTH}" if q else "")) + GEN),
                (len(lm.encode(head)), len(lm.encode(f"{head}{OWN[o]}"))), len(lm.encode(pre)))

    conds = [(o, p) for o in RANKS for p in RANKS] + [("K", "manager"), ("Q", "manager")]
    means = {}
    for o, p in conds:
        acc = {l: torch.zeros(lm.d_model) for l in layers}
        for i in tr:
            _, caps = lm.score(prompt(o, p, rows[i]["text"], q=False)[0], capture_layers=layers)
            for l in layers:
                acc[l] += caps[l][-1].float().cpu()
        means[o, p] = {l: acc[l] / len(tr) for l in layers}
    pool = lambda which, r, l: sum(means[(r, x) if which == "own" else (x, r)][l] for x in RANKS) / 3
    dirs = {l: {"own": unit(pool("own", "ceo", l) - pool("own", "worker", l)),
                "partner": unit(pool("partner", "ceo", l) - pool("partner", "worker", l)),
                "nonsense": unit(means["Q", "manager"][l] - means["K", "manager"][l]),
                "random": unit(random_direction(lm.d_model, 15485863 + l))} for l in layers}
    scales = {l: residual_scale(lm, l) for l in layers}
    print(f"directions from {len(tr)} train messages ({time.time() - t0:.0f}s)", flush=True)

    toks = {w: sorted({lm.encode(w)[0], lm.encode(w.capitalize())[0], lm.encode(" " + w)[0]}) for w in ("me", "sender")}

    def logodds(jobs):
        """jobs: (message index, span, layer, vector or None) -> log-odds me vs sender, own = partner = manager."""
        out = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            built = [prompt("manager", "manager", rows[i]["text"]) for i, *_ in chunk]
            scheds = []
            for (_, span, l, v), (_, sysr, msg) in zip(chunk, built):
                scheds.append([] if v is None else [Intervention(l, v, *sysr)] if span == "system" else [Intervention(l, v, msg)])
            first, _ = lm.generate_batch([b[0] for b in built], scheds, 1, temperature=0, first_logits=True)
            for x in first:
                x = torch.log_softmax(x.float(), -1)
                out.append(float(torch.logsumexp(x[toks["me"]], 0) - torch.logsumexp(x[toks["sender"]], 0)))
        return out

    base = logodds([(i, None, layers[0], None) for i in te])
    print(f"unsteered manager/manager log-odds {mean(base):+.2f} ({time.time() - t0:.0f}s)", flush=True)
    span_cells = []
    for l in layers:
        for a in alphas:
            norm = a * scales[l]
            for span in ("system", "message"):
                cv = {"+own": dirs[l]["own"], "-own": -dirs[l]["own"], "+partner": dirs[l]["partner"], "+random": dirs[l]["random"]}
                cell = {"layer": l, "alpha": a, "span": span}
                for c, v in cv.items():
                    cell[c] = [x - b for x, b in zip(logodds([(i, span, l, v * norm) for i in te]), base)]
                for c in ("+own", "-own", "+partner"):
                    diff = [x - r for x, r in zip(cell[c], cell["+random"])]
                    cell[f"{c} - random"] = {"mean": mean(diff), "ci": bootstrap_ci(diff)}
                span_cells.append(cell)
                f = lambda k: f"{cell[k]['mean']:+.2f} [{cell[k]['ci'][0]:+.2f}, {cell[k]['ci'][1]:+.2f}]"
                print(f"A L{l} a{a} {span:7s} | +own {mean(cell['+own']):+.2f} -own {mean(cell['-own']):+.2f} +partner "
                      f"{mean(cell['+partner']):+.2f} +rand {mean(cell['+random']):+.2f} | own-rand {f('+own - random')} "
                      f"-own-rand {f('-own - random')} partner-rand {f('+partner - random')} ({time.time() - t0:.0f}s)", flush=True)

    stop = [lm.token_id("<|im_end|>")]
    gens = []
    gconds = [(None, None)] + [(l, c) for l in layers for c in ("+own", "-own", "+random", "+nonsense")]
    for gi, instr in enumerate(INSTRUCTIONS):
        built, _, start = prompt("manager", "manager", instr, q=False)
        for l, c in gconds:
            v = None if l is None else dirs[l][c[1:]] * (1 if c[0] == "+" else -1) * args.gen_alpha * scales[l]
            sched = [] if v is None else [Intervention(l, v, start)]
            outs = lm.generate_batch([built] * args.seeds, [sched] * args.seeds, 200, stop, temperature=0.7,
                                     seeds=[10_000 + 100 * gi + k for k in range(args.seeds)])
            for k, o in enumerate(outs):
                r = lm.tok.decode([t for t in o if t not in stop])
                gens.append({"instruction": gi, "layer": l, "cond": c or "unsteered", "seed": k, "reply": r,
                             **{m: bool(re.search(p, r.lower())) for m, p in MEASURES.items()}})
    print(f"replies: {len(gens)} ({time.time() - t0:.0f}s)", flush=True)
    key = lambda g: (g["instruction"], g["seed"])
    cell = lambda l, c: {key(g): g for g in gens if g["layer"] == l and g["cond"] == c}
    beh = {"unsteered": {m: mean([float(g[m]) for g in cell(None, "unsteered").values()]) for m in MEASURES}}
    for l in layers:
        rnd = cell(l, "+random")
        for c in ("+own", "-own", "+random", "+nonsense"):
            cc = cell(l, c)
            beh[f"L{l} {c}"] = {m: mean([float(g[m]) for g in cc.values()]) for m in MEASURES}
            if c != "+random":
                for m in MEASURES:
                    diff = [float(cc[k][m]) - float(rnd[k][m]) for k in cc]
                    beh[f"L{l} {c}"][f"{m} - random"] = {"mean": mean(diff), "ci": bootstrap_ci(diff)}
    for k, v in beh.items():
        print(f"B {k:14s} " + "  ".join(f"{m} {v[m]:.2f}" + (f" (vs rand {v[m + ' - random']['mean']:+.2f} "
                                                              f"[{v[m + ' - random']['ci'][0]:+.2f}, {v[m + ' - random']['ci'][1]:+.2f}])"
                                                              if m + " - random" in v else "") for m in MEASURES), flush=True)

    out = ROOT / "runs" / f"roles_steer2_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "auth": AUTH, "instructions": INSTRUCTIONS,
                               "measures": MEASURES, "test": te, "residual_scale": scales, "base": base,
                               "span_cells": span_cells, "behaviour": beh, "replies": gens}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
