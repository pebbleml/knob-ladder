"""G3 analysis of R0 episodes: does the steered model know its knobs beyond the observer?

    python3 scripts/analyze.py runs/r0_Qwen3-32B_*.jsonl

Pure Python, runs locally on pulled jsonl. The primary metric is, per episode, the mean over
pressed concept knobs of p_cached(correct) - p_observer(correct) on the multiple choice,
with a bootstrap CI over episodes and sign and sign-flip permutation tests. Also: argmax
accuracy against chance, sham false alarms, the same restricted to knobs pressed 2+ times,
and quotes (observations, free recall cached vs observer) for the top-scoring knobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.stats import bootstrap_ci, mean, sign_flip_test, sign_test  # noqa: E402

CONTROLS = ("random", "sham")  # mirrors knobladder.episode, which needs torch


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_file"] = Path(p).name
                rows.append(r)
    return rows


def fmt_ci(xs):
    lo, hi = bootstrap_ci(xs)
    return f"{mean(xs):+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(xs)})"


def concept_knobs(row, min_presses):
    return [d for d in row["discovery"] if d["label"] not in CONTROLS and d["presses"] >= min_presses]


def delta(d):
    c = d["choice"]
    return c["cached"]["p_correct"] - c["observer"]["p_correct"]


def report(rows, min_presses):
    print(f"\n== knobs pressed >= {min_presses} ==")
    per_ep = [mean([delta(d) for d in ks]) for ks in (concept_knobs(r, min_presses) for r in rows) if ks]
    if per_ep:
        print(f"primary: cached - observer p(correct), mean over episodes {fmt_ci(per_ep)}; "
              f"sign p={sign_test(per_ep):.3f}, sign-flip p={sign_flip_test(per_ep):.4f}")
    else:
        print("primary: no episodes with qualifying concept knobs")
    knobs = [d for r in rows for d in concept_knobs(r, min_presses)]
    if knobs:
        chance = mean([d["choice"]["chance"] for d in knobs])
        acc = {c: mean([d["choice"][c]["argmax"] == d["choice"]["correct"] for d in knobs]) for c in ("cached", "observer")}
        print(f"argmax accuracy over {len(knobs)} concept knobs: cached {acc['cached']:.3f}, "
              f"observer {acc['observer']:.3f}, chance {chance:.3f}")
    shams = [d for r in rows for d in r["discovery"] if d["label"] == "sham" and d["presses"] >= min_presses]
    if shams:
        fa = {c: [1 - d["choice"][c]["p_nothing"] for d in shams] for c in ("cached", "observer")}
        print(f"sham false alarm 1 - p(nothing changed) over {len(shams)}: cached {fmt_ci(fa['cached'])}, "
              f"observer {fmt_ci(fa['observer'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=5, help="knobs to quote")
    args = ap.parse_args()
    rows = load(args.files)
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} episodes from {len(args.files)} files; models {models}")
    if len(models) > 1:
        print("WARNING: pooling across models")
    hp = {json.dumps(r.get("hyperparams"), sort_keys=True) for r in rows}
    if len(hp) > 1:
        print(f"WARNING: {len(hp)} different hyperparameter sets")

    print("\n== episodes ==")
    for r in rows:
        board = r["board"]
        counts = " ".join(f"{k}:{board[k]}={v}" for k, v in sorted(r["presses"].items(), key=lambda kv: int(kv[0])))
        print(f"{r['_file']} ep {r['episode']} seed {r['seed']} passage {r['passage_index']} "
              f"ended {r['ended']}: order {r['press_order']} | {counts or 'no presses'}")

    report(rows, 1)
    report(rows, 2)
    shams = [d for r in rows for d in r["discovery"] if d["label"] == "sham"]
    if shams:
        fa = {c: [1 - d["choice"][c]["p_nothing"] for d in shams] for c in ("cached", "observer")}
        print(f"\nall sham knobs (pressed or not), false alarm: cached {fmt_ci(fa['cached'])}, "
              f"observer {fmt_ci(fa['observer'])}")
    unpressed = [delta(d) for r in rows for d in r["discovery"] if d["label"] not in CONTROLS and not d["presses"]]
    if unpressed:
        print(f"\ncontrol, unpressed concept knobs: cached - observer p(correct) {fmt_ci(unpressed)}")

    scored = sorted(((delta(d), r, d) for r in rows for d in concept_knobs(r, 1)), key=lambda x: -x[0])
    print(f"\n== top {args.top} knobs by cached - observer ==")
    for gap, r, d in scored[: args.top]:
        c = d["choice"]
        print(f"\n{r['_file']} ep {r['episode']} knob {d['knob']} = {d['label']} ({d['presses']} presses): "
              f"delta {gap:+.3f}, argmax cached {c['cached']['argmax']!r}, observer {c['observer']['argmax']!r}")
        for e in r["events"]:
            if e["kind"] == "press" and e["knob"] == d["knob"]:
                print(f"  observed (turn {e['turn']}): {e['observation']!r}")
        print(f"  recall cached:   {d['recall']['cached']!r}")
        print(f"  recall observer: {d['recall']['observer']!r}")


if __name__ == "__main__":
    main()
