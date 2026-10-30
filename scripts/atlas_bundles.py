"""R1.a fallback: neuron bundles. Each concept's atlas poles (scripts/atlas_gen.py, same layer) summed as unit write
directions, read with the same one-press generation readout, seeds and doses as the atlas, beside the atlas's own
concept and random controls for that concept.

    python3 scripts/atlas_bundles.py --atlas runs/atlas_Qwen3.8-27B_<time>.json

Primary: mean bundle specificity at f0.5 minus random (paired by concept), and minus concept; a bundle is readable
under the atlas's rule. Only calibration passages are used.
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

from knobladder.atlas import bundle_direction  # noqa: E402
from knobladder.episode import (  # noqa: E402
    OBSERVE, START, TOOLS, TURNED, Transcript, _stops, append_passage, answer, hyperparams, system_text,
)
from knobladder.knobs import lexical_hits, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import ASSISTANT, CALIB_PASSAGES, END, PASSAGES_VERSION, user  # noqa: E402

KNOB, KNOBS = 5, 12


def named(text: str, word: str) -> bool:
    return any(term == word for term, _ in lexical_hits(text, word))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    atlas = json.loads(Path(args.atlas).read_text())
    assert atlas["mode"] == "gen" and atlas["passages_version"] == PASSAGES_VERSION
    words, layers = atlas["words"], atlas["layers"]
    fracs = [float(d[1:]) for d in atlas["doses"]]
    lm = SteeredLM(atlas["model"], dtype=getattr(torch, args.dtype))
    hp = hyperparams(True, lm.fmt.name)
    stops = _stops(lm)
    scales = {l: residual_scale(lm, l) for l in layers}
    prompts = []
    for p in CALIB_PASSAGES:
        tr = Transcript()
        tr.append(lm, lm.fmt.system(system_text(list(range(1, KNOBS + 1))), TOOLS, hp.get("effort")))
        tr.append(lm, user(START) + ASSISTANT + lm.fmt.tool_call("turn", {"knob": KNOB, "level": 10}) + END)
        window = append_passage(lm, tr, TURNED.format(k=KNOB, level=10), lm.encode(p))
        tr.append(lm, user(OBSERVE) + lm.fmt.gen_think)
        prompts.append((list(tr.ids), window))
    P = range(len(prompts))

    def run(jobs):
        rows = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids = [prompts[p][0] for p, *_ in chunk]
            scheds = [[Intervention(l, v, prompts[p][1][0], len(prompts[p][0]) + atlas["steer_tokens"])] for p, _, l, v in chunk]
            seeds = [1000 * p + words.index(w) for p, w, *_ in chunk]
            outs = lm.generate_batch(ids, scheds, atlas["max_new"], stops, hp["temperature"], hp["top_p"], hp["top_k"], seeds)
            for (p, w, _, _), p_ids, out in zip(chunk, ids, outs):
                logits, _ = lm.score(p_ids + out, positions=list(range(len(p_ids) - 1, len(p_ids) + len(out) - 1)))
                nll = float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor(out, device=logits.device)))
                t = lm.tok.decode(out)
                rows.append({"passage": p, "word": w, "text": t, "tokens": len(out), "nll": nll,
                             "named": [x for x in words if named(t, x)],
                             "named_reply": [x for x in words if named(answer(t, True), x)]})
        return rows

    base_nll = [r["nll"] for r in atlas["baseline"]]

    def summ(rows):
        sp = lambda r, key: float(r["word"] in r[key]) - mean([float(x in r[key]) for x in words if x != r["word"]])
        return {"own": mean([float(r["word"] in r["named"]) for r in rows]),
                "spec": mean([sp(r, "named") for r in rows]), "spec_reply": mean([sp(r, "named_reply") for r in rows]),
                "dnll": mean([r["nll"] - base_nll[r["passage"]] for r in rows]), "rows": rows}

    controls = {(c["layer"], c["word"], c["dose"], c["cond"]): c for c in atlas["controls"]}
    bundles = []
    for l in layers:
        members = {w: [n for n in atlas["neurons"] if n["layer"] == l and n["word"] == w] for w in words}
        vecs = {w: unit(bundle_direction(lm, members[w])) for w in words if members[w]}
        for w in vecs:
            bundles.append({"layer": l, "word": w, "members": [n["index"] for n in members[w]],
                            "cells": {}})
        for f in fracs:
            rows = run([(p, b["word"], l, vecs[b["word"]] * f * scales[l]) for b in bundles if b["layer"] == l for p in P])
            for k, b in enumerate(b for b in bundles if b["layer"] == l):
                b["cells"][f"f{f}"] = summ(rows[k * len(P):(k + 1) * len(P)])
            print(f"L{l} bundles f{f}: own {mean([b['cells'][f'f{f}']['own'] for b in bundles]):.3f} spec "
                  f"{mean([b['cells'][f'f{f}']['spec'] for b in bundles]):+.3f} ({time.time() - t0:.0f}s)", flush=True)

    ok = lambda c: c["own"] >= atlas["readable"] and c["spec"] > 0 and c["dnll"] <= atlas["max_dnll"]
    summary = {}
    for f in fracs:
        d = f"f{f}"
        spec = [b["cells"][d]["spec"] for b in bundles]
        dr = [b["cells"][d]["spec"] - controls[b["layer"], b["word"], d, "random"]["spec"] for b in bundles]
        dc = [b["cells"][d]["spec"] - controls[b["layer"], b["word"], d, "concept"]["spec"] for b in bundles]
        summary[d] = {"bundle_spec": mean(spec), "bundle_spec_ci": bootstrap_ci(spec),
                      "bundle_minus_random": mean(dr), "bundle_minus_random_ci": bootstrap_ci(dr),
                      "bundle_minus_concept": mean(dc), "bundle_minus_concept_ci": bootstrap_ci(dc),
                      "readable": sum(ok(b["cells"][d]) for b in bundles), "bundles": len(bundles)}
        s = summary[d]
        print(f"{d}: bundle spec {s['bundle_spec']:+.3f} [{s['bundle_spec_ci'][0]:+.3f}, {s['bundle_spec_ci'][1]:+.3f}] | "
              f"bundle-random {s['bundle_minus_random']:+.3f} [{s['bundle_minus_random_ci'][0]:+.3f}, "
              f"{s['bundle_minus_random_ci'][1]:+.3f}] | bundle-concept {s['bundle_minus_concept']:+.3f} | readable "
              f"{s['readable']}/{s['bundles']}", flush=True)
    for b in bundles:
        b["readable_at"] = [d for d, c in b["cells"].items() if ok(c)]
    out = Path(args.atlas).with_suffix(".bundles.json")
    out.write_text(json.dumps({"atlas": args.atlas, "provenance": lm.provenance(), "summary": summary,
                               "bundles": bundles}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
