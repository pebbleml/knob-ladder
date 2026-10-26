"""R0 pilot: concept-vector knobs. Discovery quiz, preference counts and ownership trials per episode.

    python3 scripts/r0_pilot.py --model Qwen/Qwen3-0.6B --episodes 2 --turns 12

Writes one JSON line per episode to runs/. Concepts are split so that knob directions and
probe-training directions never overlap.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.episode import preference_counts, quiz, run_episode, shuffled_board  # noqa: E402
from knobladder.knobs import CONCEPTS, concept_vectors, random_direction, residual_scale  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.ownership import ownership_trials  # noqa: E402
from knobladder.probe import train_probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=float, default=0.6, help="injection depth as a fraction of layers")
    ap.add_argument("--probe-layer", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.5, help="dose as a fraction of the median residual norm")
    ap.add_argument("--knobs", type=int, default=5, help="concept knobs per episode (plus sham and random)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layer = int(args.layer * lm.n_layers)
    probe_layer = int(args.probe_layer * lm.n_layers)
    knob_words, probe_words = CONCEPTS[:32], CONCEPTS[32:]
    vectors = concept_vectors(lm, CONCEPTS, layer)
    dose = args.alpha * residual_scale(lm, layer)
    probe = train_probe(lm, {w: vectors[w] for w in probe_words}, layer, probe_layer, dose, seed=args.seed)
    print(f"layer {layer}/{lm.n_layers}, dose {dose:.1f}, probe L{probe_layer} holdout acc {probe.holdout_accuracy:.2f}")

    out_path = ROOT / "runs" / f"r0_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        for ep in range(args.episodes):
            seed = args.seed * 100 + ep
            rng = random.Random(seed)
            board = shuffled_board(
                {w: vectors[w] for w in knob_words}, args.knobs, layer, dose, rng,
                random_dir=random_direction(lm.d_model, seed),
            )
            tr = run_episode(lm, board, turns=args.turns, seed=seed)
            row = {
                "episode": ep, "seed": seed, "model": args.model, "layer": layer, "dose": dose,
                "probe_layer": probe_layer, "probe_holdout_acc": probe.holdout_accuracy,
                "board": {k: kb.label for k, kb in board.knobs.items()},
                "events": tr.events,
                "preference": preference_counts(tr),
                "quiz": quiz(lm, board, tr, rng),
                "ownership": ownership_trials(lm, board, tr, probe, seed=seed),
                "transcript": lm.tok.decode(tr.ids),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            presses = sum(e["kind"] == "press" for e in tr.events)
            print(f"episode {ep}: {presses} presses, quiz {len(row['quiz'])}, ownership {'ok' if row['ownership'] else 'no valid choice'}")
    print(out_path)


if __name__ == "__main__":
    main()
