"""Exploratory, no claims (2026-11-11): how big is P7's 'turn the neuron up' compared with the neuron's natural range?

For each neuron in a P7 run: the added amount in activation units, delta = f* x residual_scale(layer) / ||W_down[:, i]||,
versus the neuron's own activations a_i over every token of the calibration passages and the run's stored part-1 texts
(read unsteered).

    python3 scripts/neuron_range.py --run runs/neuron_pref_p7mw_Qwen3.8-27B_20261111-024104.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.model import SteeredLM  # noqa: E402
from knobladder.text import CALIB_PASSAGES, chat_prefix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--texts", type=int, default=200)
    args = ap.parse_args()
    d = json.load(open(ROOT / args.run))
    model, fstar = d["args"]["model"], d["summary"]["f_star"]
    lm = SteeredLM(model, dtype=torch.bfloat16)
    neurons = [it for it in d["items"] if it["kind"] == "neuron"]
    by = {}
    for it in neurons:
        by.setdefault(it["layer"], []).append(it["index"])
    acts = {l: [] for l in by}

    def hook(l):
        def h(module, a):
            acts[l].append(a[0][0, 1:, torch.tensor(by[l])].detach().float().cpu())
        return h

    hs = [lm.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in by]
    texts = list(CALIB_PASSAGES) + [s["part1"] for s in d["sessions"][: args.texts]]
    n_tok = 0
    with torch.no_grad():
        for t in texts:
            ids = lm.encode(chat_prefix(t))
            n_tok += len(ids) - 1
            lm.score(ids, [])
    for h in hs:
        h.remove()
    rows = []
    for l, ix in by.items():
        A = torch.cat(acts[l])  # tokens x neurons
        for k, i in enumerate(ix):
            w = float(lm.layers[l].mlp.down_proj.weight[:, i].float().norm())
            delta = fstar * d["scale"][str(l)] / w
            a = A[:, k].abs()
            rows.append({"layer": l, "index": i, "w_norm": w, "delta": delta, "abs_max": float(a.max()),
                         "abs_p999": float(a.quantile(0.999)), "abs_p50": float(a.median()),
                         "mean": float(A[:, k].mean()), "ratio_to_max": delta / float(a.max())})
    med = lambda xs: sorted(xs)[len(xs) // 2]
    summary = {"n_tokens": n_tok, "f_star": fstar, "median_delta": med([r["delta"] for r in rows]),
               "median_abs_max": med([r["abs_max"] for r in rows]), "median_abs_p50": med([r["abs_p50"] for r in rows]),
               "median_ratio_to_max": med([r["ratio_to_max"] for r in rows]),
               "min_ratio_to_max": min(r["ratio_to_max"] for r in rows), "max_ratio_to_max": max(r["ratio_to_max"] for r in rows)}
    out = ROOT / "runs" / f"neuron_range_{Path(args.run).stem}.json"
    json.dump({"run": args.run, "summary": summary, "rows": rows}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
