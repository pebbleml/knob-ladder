"""Dose calibration: how readable is a silent injection, and how much does it disrupt reading?

    python3 scripts/calibrate.py --model Qwen/Qwen3-4B --dtype bfloat16

For each alpha (dose as a fraction of the median residual norm) and each concept, the
concept vector is injected over a neutral passage only. Two numbers per alpha:

- readout: after the passage, the model names the topic on its mind; the concept's rank
  among all candidate words' first tokens (top-1 accuracy, mean reciprocal rank, chance 1/N).
- disruption: mean increase in next-token NLL over the passage under steering.

The knob dose is the smallest alpha with clear readout and modest disruption.
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

from knobladder.knobs import CONCEPTS, concept_vectors, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.text import GEN, NEUTRAL_PASSAGES, user  # noqa: E402

QUESTION = "Name the one topic that was most on your mind while you read that passage. Answer with one word."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=float, default=0.6)
    ap.add_argument("--alphas", default="0.1,0.25,0.5,1,2")
    ap.add_argument("--concepts", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layer = int(args.layer * lm.n_layers)
    words = CONCEPTS[:32]
    vectors = concept_vectors(lm, CONCEPTS, layer)
    scale = residual_scale(lm, layer)
    # Candidate answers: first token of " word" and "word"; a concept counts as named if either ranks.
    firsts = {w: {lm.encode(w.capitalize())[0], lm.encode(w)[0]} for w in words}

    passage = NEUTRAL_PASSAGES[0]
    head = lm.encode("<|im_start|>user\n<tool_response>\nPassage:\n")
    body = lm.encode(passage)
    ids = head + body + lm.encode("\n</tool_response><|im_end|>\n" + user(QUESTION) + GEN)
    window = (len(head), len(head) + len(body))
    nll_pos = list(range(window[0] - 1, window[1] - 1))  # logits predicting each passage token
    targets = torch.tensor(body)

    def passage_nll(logits):
        return float(torch.nn.functional.cross_entropy(logits, targets))

    clean_logits, _ = lm.score(ids, positions=nll_pos + [len(ids) - 1])
    clean_nll = passage_nll(clean_logits[:-1])

    def rank_of(final_logits, w):
        cand = {x: max(float(final_logits[t]) for t in firsts[x]) for x in words}
        order = sorted(words, key=lambda x: -cand[x])
        return order.index(w) + 1

    clean_ranks = {w: rank_of(clean_logits[-1], w) for w in words[: args.concepts]}
    rows = []
    for alpha in map(float, args.alphas.split(",")):
        ranks, dnll = [], []
        for w in words[: args.concepts]:
            sched = [Intervention(layer, unit(vectors[w]) * alpha * scale, *window)]
            logits, _ = lm.score(ids, sched, positions=nll_pos + [len(ids) - 1])
            ranks.append(rank_of(logits[-1], w))
            dnll.append(passage_nll(logits[:-1]) - clean_nll)
        top1 = sum(r == 1 for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks) / len(ranks)
        row = {"alpha": alpha, "dose": alpha * scale, "top1": top1, "mrr": mrr,
               "delta_nll": sum(dnll) / len(dnll), "ranks": ranks}
        rows.append(row)
        print(f"alpha {alpha:<5} dose {alpha * scale:7.1f}  top1 {top1:.2f}  mrr {mrr:.2f}  dNLL {row['delta_nll']:+.2f}")
    clean_mrr = sum(1 / r for r in clean_ranks.values()) / len(clean_ranks)
    print(f"clean mrr {clean_mrr:.2f} (chance top1 {1 / len(words):.3f}); layer {layer}/{lm.n_layers}")

    out = ROOT / "runs" / f"calibrate_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "layer": layer, "scale": scale, "clean_nll": clean_nll,
                               "clean_mrr": clean_mrr, "rows": rows}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
