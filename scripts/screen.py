"""G0 screen: which candidate concepts do the passages already evoke, with no steering?

    python3 scripts/screen.py --model Qwen/Qwen3-32B --dtype bfloat16

For every (passage, candidate concept) pair, unsteered:

- rank: the concept's rank on the topic readout (readout.QUESTION) after reading the
  passage, among all candidates' first tokens;
- cosine: between the passage's mean residual at the layer (mean-subtracted over all
  passages) and the concept vector.

A pair is flagged if its rank is in the top 3 or its cosine is in the top 5% of all pairs.
Concepts flagged on 2+ passages, or left without a unique first token, are dropped. From
the survivors, 16 calibration and 32 knob concepts are drawn, disjoint and seeded, preferring
low pairwise cosine. Calibration and episodes read the lists from the JSON this writes.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.episode import fixed_texts  # noqa: E402
from knobladder.knobs import CONCEPTS, concept_vectors, lexical_screen, unit  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.readout import QUESTION, first_tokens, ranks, readout_prompt  # noqa: E402
from knobladder.text import NEUTRAL_PASSAGES, PASSAGES_VERSION  # noqa: E402

TOP_RANK = 3
TOP_COS_FRAC = 0.05
MAX_FLAGGED = 2  # drop a concept flagged on this many passages


def select(survivors, sim, need, max_cos, rng):
    """Greedy pick in seeded order, keeping pairwise cosine < max_cos; then fill by lowest max cosine."""
    order = list(survivors)
    rng.shuffle(order)
    picked = []
    for w in order:
        if len(picked) < need and all(sim[w][x] < max_cos for x in picked):
            picked.append(w)
    rest = [w for w in order if w not in picked]
    while len(picked) < need and rest:
        w = min(rest, key=lambda x: max((sim[x][y] for y in picked), default=-1.0))
        picked.append(w)
        rest.remove(w)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=float, default=0.7, help="depth as a fraction of layers")
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--knobs", type=int, default=32)
    ap.add_argument("--max-cos", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    lex = lexical_screen(NEUTRAL_PASSAGES + fixed_texts() + [QUESTION])
    if lex:
        sys.exit(f"lexical screen failed: {lex}")

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layer = int(args.layer * lm.n_layers)
    firsts = first_tokens(lm, CONCEPTS)
    no_token = [w for w in CONCEPTS if not firsts[w]]
    vectors = concept_vectors(lm, CONCEPTS, layer)

    rank, means, lengths = [], [], []
    for p in NEUTRAL_PASSAGES:
        ids, window = readout_prompt(lm, p)
        logits, caps = lm.score(ids, capture_layers=[layer])
        rank.append(ranks(logits, firsts))
        means.append(caps[layer][window[0]:window[1]].mean(0))
        lengths.append(window[1] - window[0])
    pm = torch.stack(means)
    pm = pm - pm.mean(0, keepdim=True)
    cos = [{w: float(torch.dot(unit(pm[i]), unit(vectors[w]))) for w in CONCEPTS} for i in range(len(pm))]

    all_cos = sorted((c for row in cos for c in row.values()), reverse=True)
    cut = all_cos[max(0, math.ceil(TOP_COS_FRAC * len(all_cos)) - 1)]
    flags = [{w: rank[i][w] <= TOP_RANK or cos[i][w] >= cut for w in CONCEPTS} for i in range(len(cos))]
    n_flagged = {w: sum(f[w] for f in flags) for w in CONCEPTS}
    dropped = sorted({w for w in CONCEPTS if n_flagged[w] >= MAX_FLAGGED} | set(no_token), key=CONCEPTS.index)
    survivors = [w for w in CONCEPTS if w not in dropped]

    unitv = {w: unit(vectors[w]) for w in CONCEPTS}
    sim = {w: {x: float(torch.dot(unitv[w], unitv[x])) for x in CONCEPTS} for w in CONCEPTS}
    need = args.calib + args.knobs
    rng = random.Random(args.seed)
    picked = select(survivors, sim, need, args.max_cos, rng)
    relaxed = []
    if len(picked) < need:  # not enough clean concepts: fill with the least-flagged, and say so loudly
        spare = sorted((w for w in dropped if w not in no_token), key=lambda w: (n_flagged[w], CONCEPTS.index(w)))
        relaxed = spare[: need - len(picked)]
        picked += relaxed
        print(f"WARNING: only {len(survivors)} survivors; relaxed screen with {relaxed}")
    rng.shuffle(picked)
    calib, knobs = sorted(picked[: args.calib]), sorted(picked[args.calib: need])
    max_pair = max(sim[a][b] for a in picked for b in picked if a != b)

    print(f"{args.model} layer {layer}/{lm.n_layers}, passages {PASSAGES_VERSION}: "
          f"{min(lengths)}-{max(lengths)} tokens (mean {sum(lengths) / len(lengths):.0f})")
    for i, r in enumerate(rank):
        top = sorted(r, key=r.get)[:TOP_RANK]
        print(f"  passage {i:>2} ({lengths[i]} tok): top {top}, max cos {max(cos[i].values()):+.2f}")
    print(f"cosine cut (top {TOP_COS_FRAC:.0%}) {cut:+.3f}; no unique first token {no_token}")
    print(f"dropped {len(dropped)}: {dropped}")
    print(f"survivors {len(survivors)}; selected max pairwise cos {max_pair:+.2f}")
    print(f"calibration {calib}\nknobs {knobs}")

    out = ROOT / "runs" / f"screen_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "layer": layer, "layer_frac": args.layer, "n_layers": lm.n_layers,
        "passages_version": PASSAGES_VERSION, "seed": args.seed, "max_cos": args.max_cos,
        "rule": {"top_rank": TOP_RANK, "top_cos_frac": TOP_COS_FRAC, "max_flagged": MAX_FLAGGED},
        "concepts": CONCEPTS, "first_tokens": {w: sorted(t) for w, t in firsts.items()},
        "no_unique_token": no_token, "passage_tokens": lengths,
        "ranks": rank, "cosines": cos, "cos_cut": cut, "flags": flags, "n_flagged": n_flagged,
        "dropped": dropped, "survivors": survivors, "relaxed": relaxed,
        "selected_max_pairwise_cos": max_pair, "calibration": calib, "knobs": knobs,
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
