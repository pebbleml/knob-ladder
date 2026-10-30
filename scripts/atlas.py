"""R1.a neuron atlas: which single MLP neurons can the model read when one is turned up over a passage?

    python3 scripts/atlas.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --corpus runs/corpus/wikitext103_val.txt

1. Pool: per layer (--layers, fractions of depth) and concept (all of knobs.CONCEPTS), the most selective MLP pole
   on the concept-vector templates (atlas.selective_neurons, worst-template z >= --min-z).
2. Labels: each pole's strongest activations in --corpus, and how often those contexts contain its concept or an
   associate (context_hit_rate). The contexts are kept for auto-interp labels.
3. Readout: on every calibration passage, the neuron injected over the passage at doses that are multiples of
   its natural activation (--multiples, "x10") and at fractions of the layer's median residual norm (--fracs,
   "f0.5"; R0's dose is f0.5); the readout QUESTION is asked and its
   concept ranked among all concepts by first-token logit. A concept vector and a random direction of the same
   norm are read the same way, and so is the unsteered passage (the concept's prior rank there).
   dNLL: the passage tokens' mean NLL under the injection minus unsteered, on passages [0, 4).
4. A pole's readable dose is the smallest-norm dose whose top-1 rate is at least --readable and whose dNLL is at most
   --max-dnll.

Only calibration passages are used, so nothing here is tuned on an episode passage.
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

from knobladder.atlas import context_hit_rate, direction, selective_neurons, template_acts, top_contexts  # noqa: E402
from knobladder.knobs import CONCEPTS, concept_vectors_at, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.readout import first_tokens, ranks, readout_prompt  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import CALIB_PASSAGES, PASSAGES_VERSION  # noqa: E402

NLL_PASSAGES = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--layers", default="0.5,0.6,0.7")
    ap.add_argument("--min-z", type=float, default=4.0)
    ap.add_argument("--multiples", default="2,5,10,20,40,80")
    ap.add_argument("--fracs", default="0.125,0.25,0.5,1", help="doses as fractions of the median residual norm")
    ap.add_argument("--corpus", default=str(ROOT / "runs" / "corpus" / "wikitext103_val.txt"))
    ap.add_argument("--corpus-tokens", type=int, default=200_000)
    ap.add_argument("--readable", type=float, default=0.5)
    ap.add_argument("--max-dnll", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="keep only this many poles (smoke tests)")
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layers = [int(float(f) * lm.n_layers) for f in args.layers.split(",")]
    firsts = first_tokens(lm, CONCEPTS)
    words = [w for w in CONCEPTS if firsts[w]]
    firsts = {w: firsts[w] for w in words}

    acts = template_acts(lm, words, layers)
    neurons = selective_neurons(acts, words, args.min_z)
    if args.limit:
        neurons = neurons[:args.limit]
    print(f"{len(neurons)} poles with z_min >= {args.min_z} over layers {layers} ({time.time() - t0:.0f}s)", flush=True)

    text = Path(args.corpus).read_text()
    for n, ctx in zip(neurons, top_contexts(lm, neurons, text, max_tokens=args.corpus_tokens)):
        n["contexts"] = ctx
        n["context_hit_rate"] = context_hit_rate(ctx, n["word"])
    print(f"contexts from {args.corpus_tokens} corpus tokens ({time.time() - t0:.0f}s)", flush=True)

    cvecs = concept_vectors_at(lm, words, layers)
    scales = {l: residual_scale(lm, l) for l in layers}
    prompts = [readout_prompt(lm, p) for p in CALIB_PASSAGES]

    def read(rows):
        """rows: (prompt index, layer, vector or None) -> first-token logits per row."""
        out = []
        for lo in range(0, len(rows), args.batch):
            chunk = rows[lo:lo + args.batch]
            ids = [prompts[p][0] for p, _, _ in chunk]
            scheds = [[] if v is None else [Intervention(l, v, *prompts[p][1])] for p, l, v in chunk]
            first, _ = lm.generate_batch(ids, scheds, 1, temperature=0, first_logits=True)
            out += list(first)
        return out

    def nll(p, sched):
        ids, (a, b) = prompts[p]
        logits, _ = lm.score(ids, sched, positions=list(range(a - 1, b - 1)))
        return float(torch.nn.functional.cross_entropy(logits, torch.tensor(ids[a:b])))

    base_rank = [ranks(x, firsts) for x in read([(p, layers[0], None) for p in range(len(prompts))])]
    base_nll = [nll(p, []) for p in range(NLL_PASSAGES)]
    doses = [(f"f{f}", float(f), None) for f in args.fracs.split(",")] + [(f"x{m}", None, float(m)) for m in args.multiples.split(",")]
    for k, n in enumerate(neurons):
        d = direction(lm, n)
        l, w = n["layer"], n["word"]
        n["write_norm"] = float(d.norm())
        cells = []
        for name, f, m in doses:
            norm = f * scales[l] if m is None else m * max(n["a_nat"], 0.0) * n["write_norm"]
            v = unit(d) * norm
            conds = {"neuron": v, "concept": unit(cvecs[l][w]) * norm,
                     "random": unit(random_direction(lm.d_model, 7919 * k + 1)) * norm}
            rows = [(p, l, conds[c]) for c in conds for p in range(len(prompts))]
            logits = read(rows)
            cell = {"dose": name, "norm": norm, "norm_frac": norm / scales[l]}
            for ci, c in enumerate(conds):
                rk = [ranks(logits[ci * len(prompts) + p], firsts)[w] for p in range(len(prompts))]
                cell[c] = {"ranks": rk, "top1": mean([float(r == 1) for r in rk]),
                           "d_rr": mean([1 / r - 1 / base_rank[p][w] for p, r in enumerate(rk)])}
            cell["dnll"] = mean([nll(p, [Intervention(l, v, *prompts[p][1])]) - base_nll[p] for p in range(NLL_PASSAGES)])
            cells.append(cell)
        n["base_ranks"] = [b[w] for b in base_rank]
        n["cells"] = cells
        ok = sorted((c for c in cells if c["neuron"]["top1"] >= args.readable and c["dnll"] <= args.max_dnll),
                    key=lambda c: c["norm"])
        n["readable_dose"], n["readable_frac"] = (ok[0]["dose"], ok[0]["norm_frac"]) if ok else (None, None)
        r0 = next(c for c in cells if c["dose"] == "f0.5")
        print(f"[{k + 1}/{len(neurons)}] L{l} n{n['index']} {'+' if n['sign'] > 0 else '-'} {w:11s} z {n['z_min']:.1f} "
              f"a_nat {n['a_nat']:.2f} hit {n['context_hit_rate']:.2f} | r0 top1 neuron {r0['neuron']['top1']:.2f} "
              f"concept {r0['concept']['top1']:.2f} random {r0['random']['top1']:.2f} | readable at "
              f"{n['readable_dose']} | " + " ".join(f"{c['dose']}:{c['neuron']['top1']:.2f}/{c['dnll']:+.2f}" for c in cells)
              + f" ({time.time() - t0:.0f}s)", flush=True)

    readable = [n for n in neurons if n["readable_dose"] is not None]
    at = lambda n, c: next(x for x in n["cells"] if x["dose"] == "f0.5")[c]["top1"]
    print(f"readable: {len(readable)}/{len(neurons)} poles; at the R0 norm (f0.5): neuron top-1 "
          f"{mean([at(n, 'neuron') for n in neurons]):.3f}, concept {mean([at(n, 'concept') for n in neurons]):.3f}, "
          f"random {mean([at(n, 'random') for n in neurons]):.3f}")
    out = ROOT / "runs" / f"atlas_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "passages_version": PASSAGES_VERSION,
        "layers": layers, "residual_scale": scales, "words": words, "min_z": args.min_z, "doses": [d for d, *_ in doses],
        "readable": args.readable, "max_dnll": args.max_dnll, "corpus": args.corpus, "corpus_tokens": args.corpus_tokens,
        "base_nll": base_nll, "neurons": neurons,
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
