"""G3 analysis of R0 episodes: does the steered model know its knobs beyond the observer?

    python3 scripts/analyze.py runs/r0_Qwen3-32B_*.jsonl

Pure Python, runs locally on pulled jsonl. Scores come from the readout ("What does knob k
do? Answer with one word, then one sentence."): the first-token logits of the answer rank
the 32 knob concepts the model is never shown;
rr = 1/rank, and Δ(c|j) = rr_cached(c|j) - rr_observer(c|j) for concept c asked about knob j.

Injected concepts can be primed in general, raising c whichever knob is asked about. The
primary metric removes that: for a pressed concept knob k carrying c,

    Δ_spec(k) = Δ(c|k) - mean over the other knobs j of Δ(c|j),

averaged per episode over pressed concept knobs, with a bootstrap CI over episodes and sign
and sign-flip tests. Also reported: the raw Δ(c|k); top-1 accuracy cached vs observer against
chance; how well the observer alone does (the transcript's own leakage); concept words in the
observations; how often the answer text names the concept; the multiple choice only if the run scored it (--choice).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.stats import bootstrap_ci, chance_mrr, mean, sign_flip_test, sign_test  # noqa: E402

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


def tests(xs):
    return f"sign p={sign_test(xs):.3f}, sign-flip p={sign_flip_test(xs):.4f}"


def concept_knobs(row, min_presses):
    return [d for d in row["discovery"] if d["label"] not in CONTROLS and d["presses"] >= min_presses]


def rr(d, cond, concept):
    return 1 / d["readout"][cond]["ranks"][concept]


def delta(d, concept):
    return rr(d, "cached", concept) - rr(d, "observer", concept)


def delta_spec(row, d):
    c = d["label"]
    others = [delta(o, c) for o in row["discovery"] if o["knob"] != d["knob"]]
    return delta(d, c) - mean(others)


def report(rows, min_presses):
    print(f"\n== knobs pressed >= {min_presses} ==")
    eps = [(r, concept_knobs(r, min_presses)) for r in rows]
    eps = [(r, ks) for r, ks in eps if ks]
    if not eps:
        print("no episodes with qualifying concept knobs")
        return
    spec = [mean([delta_spec(r, d) for d in ks]) for r, ks in eps]
    raw = [mean([delta(d, d["label"]) for d in ks]) for r, ks in eps]
    print(f"PRIMARY Δ_spec (knob-specific cached - observer 1/rank): {fmt_ci(spec)}; {tests(spec)}")
    print(f"raw Δ(c|k), not corrected for priming: {fmt_ci(raw)}; {tests(raw)}")
    knobs = [d for _, ks in eps for d in ks]
    n_cand = len(knobs[0]["readout"]["cached"]["ranks"])
    for cond in ("cached", "observer"):
        top1 = mean([d["readout"][cond]["ranks"][d["label"]] == 1 for d in knobs])
        mrr = [rr(d, cond, d["label"]) for d in knobs]
        print(f"{cond:>8}: top-1 {top1:.3f}, MRR {fmt_ci(mrr)} over {len(knobs)} knobs "
              f"(chance top-1 {1 / n_cand:.3f}, MRR {chance_mrr(n_cand):.3f})")
    hits = {c: mean([d["label"] in d["recall_hits"][c] for d in knobs]) for c in ("cached", "observer")}
    print(f"answer text names the knob's concept: cached {hits['cached']:.3f}, observer {hits['observer']:.3f}")
    if "choice" in knobs[0]:
        mc = [d["choice"]["cached"]["p_correct"] - d["choice"]["observer"]["p_correct"] for d in knobs]
        print(f"multiple choice (shows the concepts) cached - observer p(correct): {fmt_ci(mc)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=5, help="knobs to quote")
    args = ap.parse_args()
    rows = [r for r in load(args.files) if "readout" in r["discovery"][0]]
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} episodes from {len(args.files)} files; models {models}")
    if len(models) > 1:
        print("WARNING: pooling across models")
    for key in ("hyperparams", "provenance"):
        vals = {json.dumps(r.get(key), sort_keys=True) for r in rows}
        if len(vals) > 1:
            print(f"WARNING: {len(vals)} different {key}")
    print(f"provenance: {rows[0].get('provenance')}")

    print("\n== episodes ==")
    for r in rows:
        board = r["board"]
        counts = " ".join(f"{k}:{board[k]}={v}" for k, v in sorted(r["presses"].items(), key=lambda kv: int(kv[0])))
        forged = sum(e["kind"] == "forged" or e.get("observation_forged", False) for e in r["events"])
        print(f"{r['_file']} ep {r['episode']} seed {r['seed']} passage {r['passage_index']} ended {r['ended']}"
              f"{f' forged {forged}' if forged else ''}: order {r['press_order']} | {counts or 'no presses'}")

    report(rows, 1)
    report(rows, 2)

    presses = [e for r in rows for e in r["events"] if e["kind"] == "press" and e["label"] not in CONTROLS]
    if presses:
        own = mean([e["label"] in e["observation_hits"] for e in presses])
        any_ = mean([bool(e["observation_hits"]) for e in presses])
        print(f"\nobservations naming the pressed knob's concept: {own:.3f} of {len(presses)} "
              f"(any board concept: {any_:.3f}); the observer reads these too")
    unpressed = [delta(d, d["label"]) for r in rows for d in r["discovery"]
                 if d["label"] not in CONTROLS and not d["presses"]]
    if unpressed:
        print(f"control, unpressed concept knobs (never injected): Δ(c|k) {fmt_ci(unpressed)}")

    scored = sorted(((delta_spec(r, d), r, d) for r in rows for d in concept_knobs(r, 1)), key=lambda x: -x[0])
    print(f"\n== top {args.top} knobs by Δ_spec ==")
    for gap, r, d in scored[: args.top]:
        ro = d["readout"]
        print(f"\n{r['_file']} ep {r['episode']} knob {d['knob']} = {d['label']} ({d['presses']} presses): "
              f"Δ_spec {gap:+.3f}; rank cached {ro['cached']['ranks'][d['label']]}, observer "
              f"{ro['observer']['ranks'][d['label']]}; top cached {ro['cached']['top']}, observer {ro['observer']['top']}")
        for e in r["events"]:
            if e["kind"] == "press" and e["knob"] == d["knob"]:
                print(f"  observed (turn {e['turn']}): {e['observation']!r}")
        print(f"  answer cached:   {d['recall']['cached']!r}")
        print(f"  answer observer: {d['recall']['observer']!r}")


if __name__ == "__main__":
    main()
