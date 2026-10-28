"""G2 R0 episodes: blind concept-vector knobs, then discovery scoring of every knob.

    python3 scripts/r0_pilot.py --model Qwen/Qwen3-32B --dtype bfloat16 --screen latest --calibration latest --episodes 8

Knob concepts come from the G0 screen's knob list; layer and alpha from the G1 calibration's
`best` (or --layer/--alpha). Each episode has 5 concept knobs, a random direction and a sham
on knobs 1-7, and one fixed EPISODE_PASSAGES passage chosen by seed. Writes one JSON line per
episode to runs/. Ownership trials (and the probe they need) run only with --ownership.
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

from knobladder.episode import (  # noqa: E402
    CONTROLS, HYPERPARAMS, discovery, fixed_texts, preference_counts, press_counts, run_episode, shuffled_board,
)
from knobladder.knobs import (  # noqa: E402
    CONCEPTS, concept_vectors, lexical_screen, load_screen, random_direction, residual_scale, resolve_run,
)
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.text import EPISODE_PASSAGES, PASSAGES_VERSION  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--screen", required=True, help="G0 screen JSON, or 'latest' for this model")
    ap.add_argument("--calibration", help="G1 calibration JSON (or 'latest'); its best sets --layer and --alpha")
    ap.add_argument("--layer", type=float, default=0.7, help="injection depth as a fraction of layers")
    ap.add_argument("--alpha", type=float, default=1.0, help="dose as a fraction of the median residual norm")
    ap.add_argument("--knobs", type=int, default=5, help="concept knobs per episode (plus random and sham)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ownership", action="store_true", help="also run ownership trials (trains the probe)")
    ap.add_argument("--probe-layer", type=float, default=0.75)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    lex = lexical_screen(EPISODE_PASSAGES + fixed_texts(args.knobs + 2))
    if lex:
        sys.exit(f"lexical screen failed: {lex}")
    screen = load_screen(args.screen, args.model)
    calib_path = None
    if args.calibration:
        calib_path = resolve_run(args.calibration, f"calibrate_{args.model.split('/')[-1]}_*.json")
        best = json.loads(calib_path.read_text())["best"]
        assert best is not None, f"{calib_path} has no best (layer, alpha)"
        args.layer, args.alpha = best["layer_frac"], best["alpha"]

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layer = int(args.layer * lm.n_layers)
    knob_words = screen["knobs"]
    vectors = concept_vectors(lm, CONCEPTS, layer)
    dose = args.alpha * residual_scale(lm, layer)
    print(f"layer {layer}/{lm.n_layers} (frac {args.layer}), alpha {args.alpha}, dose {dose:.1f}, "
          f"{len(knob_words)} knob concepts from {screen['path']}")

    probe = None
    if args.ownership:
        from knobladder.ownership import ownership_trials
        from knobladder.probe import train_probe

        probe_layer = int(args.probe_layer * lm.n_layers)
        probe = train_probe(lm, {w: vectors[w] for w in screen["calibration"]}, layer, probe_layer, dose, seed=args.seed)
        print(f"probe L{probe_layer} holdout acc {probe.holdout_accuracy:.2f}")

    out_path = ROOT / "runs" / f"r0_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        for ep in range(args.episodes):
            seed = args.seed * 100 + ep
            rng = random.Random(seed)
            passage_index = rng.randrange(len(EPISODE_PASSAGES))
            board = shuffled_board(
                {w: vectors[w] for w in knob_words}, args.knobs, layer, dose, rng,
                random_dir=random_direction(lm.d_model, seed),
            )
            t0 = time.time()
            tr = run_episode(lm, board, EPISODE_PASSAGES[passage_index], seed=seed)
            disc = discovery(lm, board, tr, rng)
            row = {
                "episode": ep, "seed": seed, "model": args.model, "hyperparams": HYPERPARAMS,
                "passages_version": PASSAGES_VERSION, "passage_index": passage_index,
                "layer": layer, "layer_frac": args.layer, "alpha": args.alpha, "dose": dose,
                "screen": screen["path"], "calibration": str(calib_path) if calib_path else None,
                "board": {k: kb.label for k, kb in board.knobs.items()},
                "events": tr.events,
                "presses": press_counts(tr),
                "press_order": [e["knob"] for e in tr.events if e["kind"] == "press"],
                "ended": any(e["kind"] == "end_session" for e in tr.events),
                "preference": preference_counts(tr),
                "discovery": disc,
                "ownership": ownership_trials(lm, board, tr, probe, seed=seed) if args.ownership else None,
                "transcript": lm.tok.decode(tr.ids),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            deltas = [d["choice"]["cached"]["p_correct"] - d["choice"]["observer"]["p_correct"]
                      for d in disc if d["presses"] and d["label"] not in CONTROLS]
            gap = f"{sum(deltas) / len(deltas):+.3f}" if deltas else "n/a"
            print(f"episode {ep}: order {row['press_order']}, ended {row['ended']}, "
                  f"cached-observer p(correct) on pressed concepts {gap}, {time.time() - t0:.0f}s")
    print(out_path)


if __name__ == "__main__":
    main()
