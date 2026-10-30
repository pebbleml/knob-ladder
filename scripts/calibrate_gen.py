"""G1 for the observation window: does steering a press and the first N tokens the model writes show in what it writes?

    python3 scripts/calibrate_gen.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest --steer-tokens 128

A one-press episode, as `--thinking --window observation --steer-tokens N` runs it: the episode's system prompt,
START answered by a forced "turn knob 5 to 10", a calibration passage, OBSERVE, then the model's observation,
sampled with thinking. The concept vector is injected over the passage, OBSERVE and the first N generated tokens;
the rest is unsteered. Only the screen's calibration concepts and CALIB_PASSAGES are used.

Per (layer, alpha), over all (concept, passage) pairs, with one seed per pair shared by every cell:

- specificity: hit(own concept) - mean hit(other calibration concepts) per generation, where a hit is the
  concept word itself (lexical_hits, associates excluded) anywhere in the observation, thinking included;
  also on the reply alone, the part the episode history keeps;
- disruption: mean NLL of the generated tokens under the unsteered model, minus the unsteered cell's.

`best` is the cell with the highest specificity on passages [0, SELECT) among those with dNLL <= --max-dnll there,
then scored on the other passages alone; it passes if that held-out specificity's 95% CI lower bound is above 0.
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

from knobladder.episode import (  # noqa: E402
    OBSERVE, START, TOOLS, TURNED, Transcript, _stops, append_passage, answer, hyperparams, system_text,
)
from knobladder.knobs import CONCEPTS, concept_vectors_at, lexical_hits, load_screen, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import ASSISTANT, CALIB_PASSAGES, END, PASSAGES_VERSION, user  # noqa: E402

SELECT = 4  # passages [0, SELECT) choose (layer, alpha); the rest confirm it
KNOB, KNOBS = 5, 12


def named(text: str, word: str) -> bool:
    return any(term == word for term, _ in lexical_hits(text, word))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--screen", required=True, help="G0 screen JSON, or 'latest' for this model")
    ap.add_argument("--layers", default="0.5,0.6,0.7")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--steer-tokens", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=768, help="observation tokens sampled (the episode allows more)")
    ap.add_argument("--max-dnll", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    screen = load_screen(args.screen, args.model)
    words = screen["calibration"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    hp = hyperparams(True, lm.fmt.name)
    stops = _stops(lm)

    prompts = []  # (ids, passage window) per passage
    for p in CALIB_PASSAGES:
        tr = Transcript()
        tr.append(lm, lm.fmt.system(system_text(list(range(1, KNOBS + 1))), TOOLS, hp.get("effort")))
        tr.append(lm, user(START) + ASSISTANT + lm.fmt.tool_call("turn", {"knob": KNOB, "level": 10}) + END)
        window = append_passage(lm, tr, TURNED.format(k=KNOB, level=10), lm.encode(p))
        tr.append(lm, user(OBSERVE) + lm.fmt.gen_think)
        prompts.append((list(tr.ids), window))
    pairs = [(pi, ci) for pi in range(len(prompts)) for ci in range(len(words))]
    seeds = [1000 * pi + ci for pi, ci in pairs]

    def run(vectors, layer):
        """Sample every pair's observation under its concept's vector (None: unsteered)."""
        rows = []
        for lo in range(0, len(pairs), args.batch):
            chunk = pairs[lo:lo + args.batch]
            ids = [prompts[pi][0] for pi, _ in chunk]
            scheds = [[] if vectors is None else
                      [Intervention(layer, vectors[words[ci]], prompts[pi][1][0], len(prompts[pi][0]) + args.steer_tokens)]
                      for pi, ci in chunk]
            outs = lm.generate_batch(ids, scheds, args.max_new, stops, hp["temperature"], hp["top_p"], hp["top_k"],
                                     seeds[lo:lo + args.batch])
            for (pi, ci), p_ids, out in zip(chunk, ids, outs):
                logits, _ = lm.score(p_ids + out, positions=list(range(len(p_ids) - 1, len(p_ids) + len(out) - 1)))
                nll = float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor(out, device=logits.device)))
                text = lm.tok.decode(out)
                reply = answer(text, True)
                rows.append({"passage": pi, "concept": words[ci], "text": text, "tokens": len(out), "nll": nll,
                             "closed": "</think>" in text,
                             "named": [w for w in words if named(text, w)],
                             "named_reply": [w for w in words if named(reply, w)]})
        return rows

    def spec(rows, key):
        out = []
        for r in rows:
            own = float(r["concept"] in r[key])
            others = mean([float(w in r[key]) for w in words if w != r["concept"]])
            out.append(own - others)
        return out

    def summary(rows, base_nll=None):
        cut = [r for r in rows if r["passage"] < SELECT], [r for r in rows if r["passage"] >= SELECT]
        s = {k: spec(rows, k) for k in ("named", "named_reply")}
        s_sel, s_held = spec(cut[0], "named"), spec(cut[1], "named")
        nll = mean([r["nll"] for r in rows])
        return {
            "specificity": mean(s["named"]), "specificity_ci": bootstrap_ci(s["named"]),
            "specificity_reply": mean(s["named_reply"]), "specificity_reply_ci": bootstrap_ci(s["named_reply"]),
            "own_rate": mean([float(r["concept"] in r["named"]) for r in rows]),
            "select_specificity": mean(s_sel), "heldout_specificity": mean(s_held), "heldout_ci": bootstrap_ci(s_held),
            "nll": nll, "delta_nll": None if base_nll is None else nll - base_nll,
            "select_delta_nll": None if base_nll is None else
            mean([r["nll"] for r in cut[0]]) - mean([r["nll"] for r in base_rows if r["passage"] < SELECT]),
            "closed": mean([float(r["closed"]) for r in rows]), "tokens": mean([r["tokens"] for r in rows]),
        }

    t0 = time.time()
    base_rows = run(None, 0)
    base = summary(base_rows)
    print(f"{len(words)} concepts x {len(prompts)} passages; steer passage + OBSERVE + {args.steer_tokens} tokens; "
          f"sample {args.max_new}")
    print(f"unsteered: own-rate {base['own_rate']:.3f} spec {base['specificity']:+.3f} nll {base['nll']:.3f} "
          f"closed {base['closed']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    fracs = [float(f) for f in args.layers.split(",")]
    layers = [int(f * lm.n_layers) for f in fracs]
    vecs = concept_vectors_at(lm, CONCEPTS, sorted(set(layers)))
    cells = []
    for frac, layer in zip(fracs, layers):
        scale = residual_scale(lm, layer)
        for alpha in map(float, args.alphas.split(",")):
            v = {w: unit(vecs[layer][w]) * alpha * scale for w in words}
            rows = run(v, layer)
            c = {"layer": layer, "layer_frac": frac, "alpha": alpha, "dose": alpha * scale,
                 **summary(rows, base["nll"]), "rows": rows}
            cells.append(c)
            print(f"layer {layer}/{lm.n_layers} alpha {alpha:<4} dose {c['dose']:6.1f}  own {c['own_rate']:.3f}  "
                  f"spec {c['specificity']:+.3f} [{c['specificity_ci'][0]:+.3f}, {c['specificity_ci'][1]:+.3f}]  "
                  f"reply {c['specificity_reply']:+.3f}  select {c['select_specificity']:+.3f} "
                  f"held-out {c['heldout_specificity']:+.3f}  dNLL {c['delta_nll']:+.3f}  closed {c['closed']:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    ok = [c for c in cells if c["select_delta_nll"] <= args.max_dnll]
    best = None
    if ok:
        b = max(ok, key=lambda c: c["select_specificity"])
        best = {k: v for k, v in b.items() if k != "rows"}
        best["g1_pass"] = b["heldout_ci"][0] > 0
        print(f"best (chosen on passages 0-{SELECT - 1}): layer {b['layer']} (frac {b['layer_frac']}) alpha {b['alpha']}; "
              f"held-out spec {b['heldout_specificity']:+.3f} [{b['heldout_ci'][0]:+.3f}, {b['heldout_ci'][1]:+.3f}]; "
              f"dNLL {b['delta_nll']:+.3f}; G1 {'PASS' if best['g1_pass'] else 'FAIL'}")
    else:
        print(f"best: none with dNLL <= {args.max_dnll}; G1 FAIL")

    out = ROOT / "runs" / f"calibrate_gen_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "screen": screen["path"], "passages_version": PASSAGES_VERSION,
        "provenance": lm.provenance(), "select_passages": SELECT, "steer_tokens": args.steer_tokens,
        "max_new": args.max_new, "hyperparams": hp, "concepts": words, "max_dnll": args.max_dnll,
        "baseline": {**base, "rows": base_rows}, "cells": cells, "best": best,
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
