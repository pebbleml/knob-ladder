"""R1.a replication: do the atlas's readable neurons stay readable on fresh seeds?

    python3 scripts/atlas_replicate.py --atlas runs/atlas_<model>_<time>.json

Atlas v2 found 3 readable poles in 288 with 8 generations each, so selection alone could explain them. Fixed before
any new generation is read:
1. Primary set: every pole readable at f0.5 in --atlas. Exploratory set: the next poles by f0.5 specificity, up to
   --top in all.
2. Same episode, readout and calibration passages as atlas_gen.py, but --reps new seeds per passage (offset by
   10**6, so none repeats the atlas's), shared across conditions so comparisons are paired.
3. Conditions per pole at equal norm: neuron at --fracs of the layer's median residual norm; at f0.5 also the
   flipped pole (-neuron), the concept vector and a random direction (atlas_gen's seed for the concept); unsteered.
4. A primary pole replicates if, at f0.5, own >= 0.25, spec > 0 and dNLL <= 0.5 (atlas_gen's rule) and its own-word
   rate beats random (paired bootstrap CI over generations above 0). -neuron tells a concept write from a generic
   kick at that neuron.

Reading-only rerun (2026-11-10, WIRK pre-registration written before it runs; queue job 078):
    python3 scripts/atlas_replicate.py --atlas runs/atlas_Qwen3.8-27B_20261030-162055.json --top 3 --reps 8 \
        --steer-tokens 0 --seed-offset 2000000 --fracs 0.5,1 --tag _read
With --steer-tokens 0 the push covers only the passage and the OBSERVE turn: nothing the model writes is steered, so a
snow / whale mention cannot be a steered word and must come from how it took in the passage (held in its cache). Same
rule as above (at f0.5: own >= 0.25, spec > 0, dNLL <= 0.5, own - random CI > 0) = "carries into perception"; f1 is
secondary. 64 generations per cell (8 passages x 8 reps), fresh seeds.
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

from atlas_gen import KNOB, KNOBS, named  # noqa: E402
from knobladder.atlas import direction  # noqa: E402
from knobladder.episode import OBSERVE, START, TOOLS, TURNED, Transcript, _stops, answer, append_passage, hyperparams, system_text  # noqa: E402
from knobladder.knobs import concept_vectors_at, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import ASSISTANT, CALIB_PASSAGES, END, user  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--fracs", default="0.25,0.5,1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steer-tokens", type=int, default=None, help="generated tokens steered (default: the atlas's)")
    ap.add_argument("--seed-offset", type=int, default=10 ** 6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    atlas = json.loads(Path(args.atlas).read_text())
    words, ra = atlas["words"], atlas["readable"]
    ranked = sorted(atlas["neurons"], key=lambda n: -n["cells"]["f0.5"]["spec"])
    primary = [n for n in ranked if "f0.5" in n["readable_at"]]
    poles = primary + [n for n in ranked if n not in primary][:max(0, args.top - len(primary))]
    lm = SteeredLM(atlas["model"], dtype=getattr(torch, args.dtype))
    hp, stops = hyperparams(True, lm.fmt.name), _stops(lm)
    fracs = [float(f) for f in args.fracs.split(",")]
    layers = sorted({n["layer"] for n in poles})
    scales = {l: residual_scale(lm, l) for l in layers}
    cvecs = concept_vectors_at(lm, words, layers)
    print(f"{len(primary)} primary + {len(poles) - len(primary)} exploratory poles: "
          + ", ".join(f"L{n['layer']} n{n['index']}{'+' if n['sign'] > 0 else '-'}{n['word']}" for n in poles), flush=True)

    prompts = []
    for p in CALIB_PASSAGES:
        tr = Transcript()
        tr.append(lm, lm.fmt.system(system_text(list(range(1, KNOBS + 1))), TOOLS, hp.get("effort")))
        tr.append(lm, user(START) + ASSISTANT + lm.fmt.tool_call("turn", {"knob": KNOB, "level": 10}) + END)
        window = append_passage(lm, tr, TURNED.format(k=KNOB, level=10), lm.encode(p))
        tr.append(lm, user(OBSERVE) + lm.fmt.gen_think)
        prompts.append((list(tr.ids), window))
    trials = [(p, r) for p in range(len(prompts)) for r in range(args.reps)]
    seed = lambda p, r, w: args.seed_offset + 1000 * p + 100 * r + words.index(w)
    steer_tokens = atlas["steer_tokens"] if args.steer_tokens is None else args.steer_tokens

    def run(jobs):
        """jobs: (passage, rep, word, layer, vector or None) -> one row each."""
        rows = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids = [prompts[p][0] for p, *_ in chunk]
            scheds = [[] if v is None else [Intervention(l, v, prompts[p][1][0], len(prompts[p][0]) + steer_tokens)]
                      for p, _, _, l, v in chunk]
            outs = lm.generate_batch(ids, scheds, atlas["max_new"], stops, hp["temperature"], hp["top_p"], hp["top_k"],
                                     [seed(p, r, w) for p, r, w, *_ in chunk])
            for (p, r, w, _, _), p_ids, out in zip(chunk, ids, outs):
                logits, _ = lm.score(p_ids + out, positions=list(range(len(p_ids) - 1, len(p_ids) + len(out) - 1)))
                nll = float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor(out, device=logits.device)))
                t = lm.tok.decode(out)
                rows.append({"passage": p, "rep": r, "word": w, "text": t, "nll": nll,
                             "named": [x for x in words if named(t, x)],
                             "named_reply": [x for x in words if named(answer(t, True), x)]})
        return rows

    # unsteered, once per (trial, pole word): seeds depend on the word
    pw = sorted({n["word"] for n in poles})
    base = {(r["passage"], r["rep"], r["word"]): r for r in run([(p, r, w, layers[0], None) for w in pw for p, r in trials])}
    print(f"unsteered: own-word rate {mean([float(r['word'] in r['named']) for r in base.values()]):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    def summ(rows):
        own = [float(r["word"] in r["named"]) for r in rows]
        sp = [o - mean([float(x in r["named"]) for x in words if x != r["word"]]) for o, r in zip(own, rows)]
        return {"own": mean(own), "own_each": own, "spec": mean(sp),
                "own_reply": mean([float(r["word"] in r["named_reply"]) for r in rows]),
                "dnll": mean([r["nll"] - base[r["passage"], r["rep"], r["word"]]["nll"] for r in rows]), "rows": rows}

    out_poles = []
    for n in poles:
        l, w = n["layer"], n["word"]
        d, s = unit(direction(lm, n)), scales[l]
        conds = {f"neuron f{f}": d * f * s for f in fracs}
        conds |= {"-neuron f0.5": -d * 0.5 * s, "concept f0.5": unit(cvecs[l][w]) * 0.5 * s,
                  "random f0.5": unit(random_direction(lm.d_model, 7919 * words.index(w) + 1)) * 0.5 * s}
        names = list(conds)
        rows = run([(p, r, w, l, conds[c]) for c in names for p, r in trials])
        cells = {c: summ(rows[k * len(trials):(k + 1) * len(trials)]) for k, c in enumerate(names)}
        cells["unsteered"] = summ([base[p, r, w] for p, r in trials])
        c5, rnd = cells["neuron f0.5"], cells["random f0.5"]
        diff = [a - b for a, b in zip(c5["own_each"], rnd["own_each"])]
        res = {"layer": l, "index": n["index"], "sign": n["sign"], "word": w, "primary": n in primary,
               "atlas_own": n["cells"]["f0.5"]["own"], "hit": n.get("hit"), "z_min": n["z_min"],
               "own_minus_random": mean(diff), "own_minus_random_ci": bootstrap_ci(diff),
               "readable": c5["own"] >= ra and c5["spec"] > 0 and c5["dnll"] <= atlas["max_dnll"], "cells": cells}
        res["replicates"] = res["readable"] and res["own_minus_random_ci"][0] > 0
        out_poles.append(res)
        print(f"L{l} n{n['index']}{'+' if n['sign'] > 0 else '-'}{w} ({'primary' if res['primary'] else 'explor.'}, atlas own "
              f"{res['atlas_own']:.2f}) | " + " ".join(f"{c} {cells[c]['own']:.2f}" for c in names + ["unsteered"])
              + f" | spec {c5['spec']:+.3f} dNLL {c5['dnll']:+.2f} | own-rand {res['own_minus_random']:+.2f} "
              f"[{res['own_minus_random_ci'][0]:+.2f}, {res['own_minus_random_ci'][1]:+.2f}] replicates {res['replicates']}"
              f" ({time.time() - t0:.0f}s)", flush=True)

    prim = [r for r in out_poles if r["primary"]]
    print(f"primary replicated: {sum(r['replicates'] for r in prim)}/{len(prim)}", flush=True)
    out = ROOT / "runs" / f"atlas_rep{args.tag}_{atlas['model'].split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"atlas": args.atlas, "model": atlas["model"], "provenance": lm.provenance(), "reps": args.reps,
                               "steer_tokens": steer_tokens, "seed_offset": args.seed_offset, "fracs": fracs, "residual_scale": scales, "poles": out_poles}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
