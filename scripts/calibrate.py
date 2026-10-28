"""G1 dose calibration: how readable is a silent injection, and how much does it disrupt reading?

    python3 scripts/calibrate.py --model Qwen/Qwen3-4B --dtype bfloat16 --screen latest

Uses only the screen's calibration concepts and CALIB_PASSAGES, never knob concepts or
episode passages. For each (layer, alpha), with alpha the dose as a fraction of the median
residual norm, each concept vector is injected over the passage only (or, with --open,
through the question as well) and two numbers are averaged over all (concept, passage) pairs:

- readout: after the passage, the model names the topic on its mind; the concept's rank
  among the calibration concepts' first tokens (top-1, MRR with a 95% bootstrap CI);
- disruption: mean increase in next-token NLL over the passage under steering.

Selection and confirmation use different passages, so the pick's score is not inflated by
the pick: `best` is the (layer, alpha) with the highest MRR on the first half of
CALIB_PASSAGES among those with dNLL <= 1, and it is then scored on the second half alone.
G1 passes if that held-out MRR's CI lower bound exceeds both chance and the upper bound of
the unsteered baseline's CI on the same half.
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

from knobladder.knobs import CONCEPTS, concept_vectors_at, load_screen, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.readout import QUESTION, first_tokens, ranks, readout_prompt  # noqa: E402
from knobladder.stats import bootstrap_ci, chance_mrr, mean  # noqa: E402
from knobladder.text import CALIB_PASSAGES, PASSAGES_VERSION  # noqa: E402

SELECT = 4  # passages [0, SELECT) choose (layer, alpha); the rest confirm it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--screen", required=True, help="G0 screen JSON, or 'latest' for this model")
    ap.add_argument("--layers", default="0.5,0.6,0.7,0.8,0.9", help="injection depths as fractions of layers")
    ap.add_argument("--alphas", default="0.25,0.5,1,2")
    ap.add_argument("--max-dnll", type=float, default=1.0)
    ap.add_argument("--open", action="store_true",
                    help="keep steering on through the question (Lindsey-style), not just the passage")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    screen = load_screen(args.screen, args.model)
    words = screen["calibration"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    firsts = first_tokens(lm, words)
    chance = chance_mrr(len(words))

    prompts = []  # (ids, window, positions, targets) per passage
    for p in CALIB_PASSAGES:
        ids, window = readout_prompt(lm, p)
        positions = list(range(window[0] - 1, window[1] - 1)) + [len(ids) - 1]  # passage predictions, then answer
        prompts.append((ids, window, positions, torch.tensor(ids[window[0]:window[1]])))

    def passage_nll(logits, targets):
        return float(torch.nn.functional.cross_entropy(logits, targets))

    clean_nll, base_rr, base_by_passage = [], [], []
    for ids, window, positions, targets in prompts:
        logits, _ = lm.score(ids, positions=positions)
        clean_nll.append(passage_nll(logits[:-1], targets))
        r = ranks(logits[-1], firsts)
        rr = [1 / r[w] for w in words]
        base_rr += rr
        base_by_passage.append(mean(rr))
    held = base_rr[SELECT * len(words):]
    base = {"mrr": mean(base_rr), "ci": bootstrap_ci(base_rr), "per_passage": base_by_passage,
            "heldout_mrr": mean(held), "heldout_ci": bootstrap_ci(held)}
    print(f"{len(words)} calibration concepts, {len(prompts)} passages; chance mrr {chance:.3f}")
    print(f"unsteered mrr {base['mrr']:.3f} CI [{base['ci'][0]:.3f}, {base['ci'][1]:.3f}]; per passage "
          + " ".join(f"{m:.2f}" for m in base_by_passage))

    fracs = [float(f) for f in args.layers.split(",")]
    layers = [int(f * lm.n_layers) for f in fracs]
    vectors = concept_vectors_at(lm, CONCEPTS, sorted(set(layers)))
    rows = []
    for frac, layer in zip(fracs, layers):
        scale = residual_scale(lm, layer)
        for alpha in map(float, args.alphas.split(",")):
            rr, dnll, rank_rows = [], [], []
            for (ids, window, positions, targets), nll0 in zip(prompts, clean_nll):
                row_ranks = []
                for w in words:
                    v = unit(vectors[layer][w]) * alpha * scale
                    sched = [Intervention(layer, v, window[0], None if args.open else window[1])]
                    logits, _ = lm.score(ids, sched, positions=positions)
                    r = ranks(logits[-1], firsts)[w]
                    row_ranks.append(r)
                    rr.append(1 / r)
                    dnll.append(passage_nll(logits[:-1], targets) - nll0)
                rank_rows.append(row_ranks)
            cut = SELECT * len(words)
            row = {"layer": layer, "layer_frac": frac, "alpha": alpha, "dose": alpha * scale,
                   "top1": mean([x == 1 for x in rr]), "mrr": mean(rr), "mrr_ci": bootstrap_ci(rr),
                   "select_mrr": mean(rr[:cut]), "heldout_mrr": mean(rr[cut:]), "heldout_ci": bootstrap_ci(rr[cut:]),
                   "delta_nll": mean(dnll), "select_delta_nll": mean(dnll[:cut]), "ranks": rank_rows}
            rows.append(row)
            print(f"layer {layer:>2}/{lm.n_layers} alpha {alpha:<5} dose {row['dose']:7.1f}  top1 {row['top1']:.2f}  "
                  f"mrr {row['mrr']:.3f} [{row['mrr_ci'][0]:.3f}, {row['mrr_ci'][1]:.3f}]  "
                  f"select {row['select_mrr']:.3f} held-out {row['heldout_mrr']:.3f}  dNLL {row['delta_nll']:+.2f}")

    ok = [r for r in rows if r["select_delta_nll"] <= args.max_dnll]
    best = None
    if ok:
        b = max(ok, key=lambda r: r["select_mrr"])
        best = {k: b[k] for k in ("layer", "layer_frac", "alpha", "dose", "top1", "mrr", "mrr_ci", "select_mrr",
                                  "heldout_mrr", "heldout_ci", "delta_nll")}
        lo = b["heldout_ci"][0]
        best["above_chance"] = lo > chance
        best["above_baseline"] = lo > base["heldout_ci"][1]
        best["g1_pass"] = best["above_chance"] and best["above_baseline"]
        print(f"best (chosen on passages 0-{SELECT - 1}): layer {b['layer']} (frac {b['layer_frac']}) alpha {b['alpha']}; "
              f"held-out mrr {b['heldout_mrr']:.3f} [{lo:.3f}, {b['heldout_ci'][1]:.3f}] vs chance {chance:.3f}, "
              f"baseline {base['heldout_mrr']:.3f} [{base['heldout_ci'][0]:.3f}, {base['heldout_ci'][1]:.3f}]; "
              f"dNLL {b['delta_nll']:+.2f}; G1 {'PASS' if best['g1_pass'] else 'FAIL'}")
    else:
        print(f"best: none with dNLL <= {args.max_dnll}; G1 FAIL")

    mode = "open" if args.open else "silent"
    out = ROOT / "runs" / f"calibrate{'_open' if args.open else ''}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "mode": mode, "screen": screen["path"], "passages_version": PASSAGES_VERSION,
        "provenance": lm.provenance(), "select_passages": SELECT,
        "question": QUESTION, "concepts": words, "chance_mrr": chance, "clean_nll": clean_nll,
        "baseline": base, "max_dnll": args.max_dnll, "rows": rows, "best": best,
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
