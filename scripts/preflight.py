"""Preflight on the real model, before episodes: does the injection do what we think, and does it stack?

    python3 scripts/preflight.py --model Qwen/Qwen3-32B --dtype bfloat16 --screen latest --calibration latest

Reports, never stops a run. Calibration concepts and passages only.

1. Hook parity, three ways, on a calibration concept at the calibrated (layer, dose):
   - manual: a pre-hook on layer L+1 records its input; with our silent-window
     intervention on [s, e) it must differ from the clean input by exactly the vector on
     [s, e) and by nothing before s (layer L's own output cannot change, so the difference
     is the add itself).
   - upstream: an independent hook with the semantics of SteeringHook in
     safety-research/introspection-mechanisms (Macar, ..., Lindsey): add strength * vector
     from start_pos onward. Their fallback for layers returning a bare tensor (Qwen3 in
     transformers 4.57) only steers with start_pos=None, so this hook handles both shapes.
     It has no end position, so it is compared with our open-ended intervention.
   - final logits of both, max abs difference and argmax agreement.
2. Accumulation: the same passage n = 0, 1, 2, 3 times in separate tool responses, steered
   with the same concept each time, then an unsteered distractor passage, then the topic
   question. MRR among the calibration concepts by n. If the silent window carries and
   presses accumulate, MRR rises with n.
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

from knobladder.knobs import CONCEPTS, concept_vectors, load_screen, residual_scale, resolve_run, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.readout import QUESTION, first_tokens, ranks, readout_prompt  # noqa: E402
from knobladder.stats import bootstrap_ci, chance_mrr, mean  # noqa: E402
from knobladder.text import CALIB_PASSAGES, END, GEN, user  # noqa: E402


class UpstreamStyleHook:
    """Add strength * vector to a layer's output from start_pos on (introspection-mechanisms semantics)."""

    def __init__(self, vector: torch.Tensor, strength: float = 1.0, start_pos=None):
        self.vector, self.strength, self.start_pos = vector, strength, start_pos

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        v = (self.vector * self.strength).to(h.device, h.dtype).view(1, 1, -1)
        h = h.clone()
        if self.start_pos is None:
            h += v
        elif h.shape[1] > 1 and self.start_pos < h.shape[1]:
            h[:, self.start_pos:] += v
        elif h.shape[1] == 1:
            h += v  # generation step: every new token is past start_pos
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def parity(lm: SteeredLM, layer: int, v: torch.Tensor, passage: str):
    ids, (s, e) = readout_prompt(lm, passage)
    x = torch.tensor([ids], device=lm.device)
    seen = {}

    def record(module, inputs):
        seen["h"] = inputs[0][0].detach().float().cpu()

    pre = lm.layers[layer + 1].register_forward_pre_hook(record)
    try:
        lm.score(ids)
        clean = seen["h"]
        lm.score(ids, [Intervention(layer, v, s, e)])
        steered = seen["h"]
    finally:
        pre.remove()
    d = steered - clean
    vf = v.float()
    tol = 1e-2 * float(vf.norm())  # bf16 rounding of a residual of this size
    manual = {
        "max_err_in_window": float((d[s:e] - vf).norm(dim=-1).max()),
        "max_change_before_window": float(d[:s].norm(dim=-1).max()),
        "tolerance": tol,
    }
    manual["pass"] = manual["max_err_in_window"] <= tol and manual["max_change_before_window"] <= tol

    ours, _ = lm.score(ids, [Intervention(layer, v, s, None)])
    handle = lm.layers[layer].register_forward_hook(UpstreamStyleHook(v, 1.0, s))
    try:
        with torch.no_grad():
            theirs = lm.model(x).logits[0, -1].float().cpu()
    finally:
        handle.remove()
    clean_logits, _ = lm.score(ids)
    upstream = {
        "max_abs_logit_diff": float((ours - theirs).abs().max()),
        "steering_effect": float((ours - clean_logits).abs().max()),
        "same_argmax": int(ours.argmax()) == int(theirs.argmax()),
    }
    upstream["pass"] = upstream["same_argmax"] and upstream["max_abs_logit_diff"] < 0.05 * upstream["steering_effect"]
    return {"manual": manual, "upstream": upstream}


def accumulation(lm, layer, vectors, words, firsts, dose, passages, max_n=3):
    """MRR by number of steered copies of the passage, read out after an unsteered distractor."""
    head = lm.encode("<|im_start|>user\n<tool_response>\nPassage:\n")
    tail = lm.encode("\n</tool_response>" + END)
    distractor = CALIB_PASSAGES[-1]
    out = {}
    for n in range(max_n + 1):
        rr = []
        for p in passages:
            body = lm.encode(p)
            ids, windows = [], []
            for _ in range(n):
                ids += head
                windows.append((len(ids), len(ids) + len(body)))
                ids += body + tail
            ids += head + lm.encode(distractor) + tail + lm.encode(user(QUESTION) + GEN)
            for w in words:
                sched = [Intervention(layer, unit(vectors[w]) * dose, a, b) for a, b in windows]
                logits, _ = lm.score(ids, sched)
                rr.append(1 / ranks(logits, firsts)[w])
        out[n] = {"mrr": mean(rr), "ci": bootstrap_ci(rr)}
        print(f"accumulation n={n}: mrr {out[n]['mrr']:.3f} [{out[n]['ci'][0]:.3f}, {out[n]['ci'][1]:.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--screen", required=True)
    ap.add_argument("--calibration", help="G1 calibration JSON (or 'latest'); its best sets --layer and --alpha")
    ap.add_argument("--layer", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    screen = load_screen(args.screen, args.model)
    if args.calibration:
        path = resolve_run(args.calibration, f"calibrate_{args.model.split('/')[-1]}_*.json")
        calib = json.loads(path.read_text())
        assert Path(calib["screen"]).resolve() == Path(screen["path"]).resolve(), (calib["screen"], screen["path"])
        args.layer, args.alpha = calib["best"]["layer_frac"], calib["best"]["alpha"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layer = int(args.layer * lm.n_layers)
    words = screen["calibration"]
    vectors = concept_vectors(lm, CONCEPTS, layer)
    dose = args.alpha * residual_scale(lm, layer)
    firsts = first_tokens(lm, words)
    print(f"{args.model} {args.dtype} layer {layer}/{lm.n_layers} alpha {args.alpha} dose {dose:.1f}")

    par = parity(lm, layer, unit(vectors[words[0]]) * dose, CALIB_PASSAGES[0])
    print(f"parity manual: {par['manual']}")
    print(f"parity upstream-style hook: {par['upstream']}")
    acc = accumulation(lm, layer, vectors, words, firsts, dose, CALIB_PASSAGES[:4])
    print(f"chance mrr {chance_mrr(len(words)):.3f}")

    out = ROOT / "runs" / f"preflight_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "screen": screen["path"], "layer": layer,
        "alpha": args.alpha, "dose": dose, "parity": par, "accumulation": acc,
        "chance_mrr": chance_mrr(len(words)),
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
