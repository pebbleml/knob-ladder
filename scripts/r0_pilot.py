"""G2 R0 episodes: blind concept-vector knobs, then discovery scoring of every knob.

    python3 scripts/r0_pilot.py --model Qwen/Qwen3-32B --dtype bfloat16 --screen latest --calibration latest --episodes 8

Knob concepts come from the G0 screen's knob list; layer and alpha from the G1 calibration's
`best` (or --layer/--alpha), whose dose is level 10. Each episode has 10 concept knobs, a random
direction and a sham on knobs 1-12, and the same EPISODE_PASSAGES passage on every press of every
episode (--passage; -1 cycles through all 16). --window observation keeps steering on through
the model's observation and takes its dose from the open-window calibration.
Writes one JSON line per episode to runs/, and every message as it happens to
runs/live/<run>/ for scripts/watch.py (the model never reads either). The multiple choice,
which shows the model the board's concepts, runs only with --choice; ownership trials (and
the probe they need) only with --ownership.

--batch N samples N episodes of the shard together (each with its own seed and schedule, so
the same episodes as --batch 1 up to batched numerics) and batches their discovery branches;
rows and live logs stay one per episode.
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
    CONTROLS, discovery, fixed_texts, hyperparams, preference_counts, press_counts, run_episodes, shuffled_board,
)
from knobladder.knobs import (  # noqa: E402
    CONCEPTS, concept_vectors, lexical_screen, load_screen, random_direction, residual_scale, resolve_run,
)
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.readout import first_tokens  # noqa: E402
from knobladder.text import EPISODE_PASSAGES, PASSAGES_VERSION  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--screen", required=True, help="G0 screen JSON, or 'latest' for this model")
    ap.add_argument("--calibration", help="G1 calibration JSON (or 'latest'); its best sets --layer and --alpha")
    ap.add_argument("--layer", type=float, default=0.7, help="injection depth as a fraction of layers")
    ap.add_argument("--alpha", type=float, default=1.0, help="dose as a fraction of the median residual norm")
    ap.add_argument("--knobs", type=int, default=10, help="concept knobs per episode (plus random and sham)")
    ap.add_argument("--window", default="passage", choices=["passage", "observation"],
                    help="steer the passage only, or the passage and the observation written after it")
    ap.add_argument("--passage", type=int, default=0, help="EPISODE_PASSAGES index for every episode; -1 cycles")
    ap.add_argument("--steer-tokens", type=int, help="with --window observation: steer only the first n tokens it writes")
    ap.add_argument("--steer-reply", type=int,
                    help="with --window observation and --thinking: also steer the reply's first n tokens")
    ap.add_argument("--stop-file", default=str(ROOT / "runs" / "STOP"),
                    help="once this file exists, each episode gives its final hypothesis and stops (budget)")
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode for episode turns (not discovery)")
    ap.add_argument("--keep-thinking", action="store_true",
                    help="with --thinking: keep past think blocks in the history (Qwen3's template drops them)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: run episodes e with e %% n == i (scripts/shards.sh)")
    ap.add_argument("--batch", type=int, default=1, help="episodes sampled together, and discovery branches per forward")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="keep each batch's KV cache across turns instead of re-prefilling every transcript")
    ap.add_argument("--choice", action="store_true", help="also score the multiple choice (names the concepts)")
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
        mode = "open" if args.window == "observation" else "silent"
        prefix = "calibrate_open_" if mode == "open" else "calibrate_"
        calib_path = resolve_run(args.calibration, f"{prefix}{args.model.split('/')[-1]}_*.json")
        calib = json.loads(calib_path.read_text())
        # 'latest' resolves each file on its own; make sure they belong together.
        assert Path(calib["screen"]).resolve() == Path(screen["path"]).resolve(), \
            f"{calib_path} was calibrated on {calib['screen']}, not {screen['path']}"
        assert calib["model"] == args.model and calib["mode"] == mode, (calib["model"], calib["mode"])
        assert calib["passages_version"] == screen["passages_version"] == PASSAGES_VERSION
        best = calib["best"]
        assert best is not None, f"{calib_path} has no best (layer, alpha)"
        args.layer, args.alpha = best["layer_frac"], best["alpha"]
    assert screen["passages_version"] == PASSAGES_VERSION, screen["passages_version"]

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layer = int(args.layer * lm.n_layers)
    knob_words = screen["knobs"]
    vectors = concept_vectors(lm, CONCEPTS, layer)
    scale = residual_scale(lm, layer)
    dose = args.alpha * scale
    firsts = first_tokens(lm, knob_words)  # readout candidates; the model never sees them
    assert all(firsts.values()), [w for w, t in firsts.items() if not t]
    raw_norm = {w: float(vectors[w].norm()) / scale for w in knob_words}  # for comparison with raw-vector dosing
    print(f"layer {layer}/{lm.n_layers} (frac {args.layer}), alpha {args.alpha}, dose {dose:.1f}, "
          f"{len(knob_words)} knob concepts from {screen['path']}; raw |v|/median "
          f"{min(raw_norm.values()):.2f}-{max(raw_norm.values()):.2f}")
    prov = lm.provenance()

    probe = None
    if args.ownership:
        from knobladder.ownership import ownership_trials
        from knobladder.probe import train_probe

        probe_layer = int(args.probe_layer * lm.n_layers)
        probe = train_probe(lm, {w: vectors[w] for w in screen["calibration"]}, layer, probe_layer, dose, seed=args.seed)
        print(f"probe L{probe_layer} holdout acc {probe.holdout_accuracy:.2f}")

    shard, shards = map(int, args.shard.split("/"))
    run_id = f"r0_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}"
    if shards > 1:
        run_id += f"_s{shard}of{shards}"
    out_path = ROOT / "runs" / f"{run_id}.jsonl"
    live = ROOT / "runs" / "live" / run_id
    live.mkdir(parents=True, exist_ok=True)
    (live / "meta.json").write_text(json.dumps({
        "run": run_id, "model": args.model, "layer": layer, "alpha": args.alpha, "dose": dose, "window": args.window,
        "steer_tokens": args.steer_tokens, "steer_reply": args.steer_reply, "thinking": args.thinking, "keep_thinking": args.keep_thinking, "episodes": args.episodes, "shard": args.shard, "batch": args.batch, "reuse_cache": args.reuse_cache,
        "started": time.time()}))

    def start(ep):
        """An episode's board and live log. The seed depends only on ep, whatever the shard or batch."""
        seed = args.seed * 100 + ep
        rng = random.Random(seed)
        passage_index = args.passage if args.passage >= 0 else (args.seed + ep) % len(EPISODE_PASSAGES)
        board = shuffled_board(
            {w: vectors[w] for w in knob_words}, args.knobs, layer, dose, rng,
            random_dir=random_direction(lm.d_model, seed),
        )
        board.window, board.steer_tokens, board.steer_reply = args.window, args.steer_tokens, args.steer_reply
        log = (live / f"ep{ep:03d}.jsonl").open("a")

        def note(kind, **fields):
            log.write(json.dumps({"t": time.time(), "ep": ep, "type": kind, **fields}) + "\n")
            log.flush()

        def emit(role, text, knob=None, label=None, steered=False, **extra):
            note("msg", role=role, text=text, knob=knob, label=label, steered=steered, **extra)

        note("board", board={str(k): kb.label for k, kb in board.knobs.items()}, passage_index=passage_index)
        return {"ep": ep, "seed": seed, "rng": rng, "passage_index": passage_index, "board": board,
                "log": log, "note": note, "emit": emit}

    eps = list(range(shard, args.episodes, shards))
    with out_path.open("w") as f:
        for g in range(0, len(eps), args.batch):
            group = [start(ep) for ep in eps[g:g + args.batch]]
            t0 = time.time()
            trs = run_episodes(lm, [e["board"] for e in group], [EPISODE_PASSAGES[e["passage_index"]] for e in group],
                               [e["seed"] for e in group], [e["emit"] for e in group], reuse_cache=args.reuse_cache,
                               thinking=args.thinking, keep_thinking=args.keep_thinking, stop_file=args.stop_file)
            for e, tr in zip(group, trs):
                ep, seed, board, note = e["ep"], e["seed"], e["board"], e["note"]
                disc = discovery(lm, board, tr, firsts, e["rng"] if args.choice else None, batch=args.batch)
                note("discovery", rows=[{
                    "knob": d["knob"], "label": d["label"], "presses": d["presses"], "recall": d["recall"],
                    "readout": {c: {"rank": d["readout"][c]["ranks"].get(d["label"]), "top": d["readout"][c]["top"]}
                                for c in ("cached", "observer")},
                } for d in disc])
                note("done", seconds=time.time() - t0)  # since the batch started
                e["log"].close()
                row = {
                    "episode": ep, "seed": seed, "model": args.model, "hyperparams": hyperparams(args.thinking), "keep_thinking": args.keep_thinking,
                    "batch": args.batch, "reuse_cache": args.reuse_cache,
                    "provenance": prov, "readout_candidates": knob_words, "raw_norm_ratio": raw_norm,
                    "passages_version": PASSAGES_VERSION, "passage_index": e["passage_index"],
                    "layer": layer, "layer_frac": args.layer, "alpha": args.alpha, "dose": dose, "window": args.window,
                    "steer_tokens": args.steer_tokens, "steer_reply": args.steer_reply, "screen": screen["path"], "calibration": str(calib_path) if calib_path else None,
                    "board": {k: kb.label for k, kb in board.knobs.items()},
                    "events": tr.events,
                    "presses": press_counts(tr),
                    "press_order": [ev["knob"] for ev in tr.events if ev["kind"] == "press"],
                    "ended": any(ev["kind"] == "end_session" for ev in tr.events),
                    "preference": preference_counts(tr),
                    "discovery": disc,
                    "ownership": ownership_trials(lm, board, tr, probe, seed=seed) if args.ownership else None,
                    "transcript": lm.tok.decode(tr.ids),
                    "ids": tr.ids,
                    "schedule": [[iv.layer, iv.start, iv.end] for iv in tr.schedule],
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                deltas = [1 / d["readout"]["cached"]["ranks"][d["label"]] - 1 / d["readout"]["observer"]["ranks"][d["label"]]
                          for d in disc if d["presses"] and d["label"] not in CONTROLS]
                gap = f"{sum(deltas) / len(deltas):+.3f}" if deltas else "n/a"
                print(f"episode {ep}: order {row['press_order']}, ended {row['ended']}, "
                      f"cached-observer readout 1/rank on pressed concepts {gap}, {time.time() - t0:.0f}s")
    print(out_path)


if __name__ == "__main__":
    main()
