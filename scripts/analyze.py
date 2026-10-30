"""G3 analysis of R0 episodes: does the steered model know its knobs beyond the observer?

    python3 scripts/analyze.py runs/r0_Qwen3-32B_*.jsonl

Pure Python, runs locally on pulled jsonl. Scores come from the readout ("What does knob k
do? Answer with one word, then one sentence."): the first-token logits of the answer rank
the 32 knob concepts the model is never shown;
rr = 1/rank, and Δ(c|j) = rr_cached(c|j) - rr_observer(c|j) for concept c asked about knob j.

Injected concepts can be primed in general, raising c whichever knob is asked about. The
primary metric removes that: for a pressed concept knob k carrying c,

    Δ_spec(k) = Δ(c|k) - mean over the other knobs j of Δ(c|j),

averaged per episode over pressed concept knobs (knobs pressed at a level above 0 at least once),
with a bootstrap CI over episodes, sign and sign-flip tests, and a label-permutation null: the
concept's knob is replaced by a uniformly random knob of the same episode, which gives exactly
mean 0 when knobs are exchangeable. Δ_spec = spec_cached - spec_observer, where spec_cond(k) is
rr_cond(c|k) - mean over j != k of rr_cond(c|j); spec_observer alone is what the transcript says
about each knob. The random and sham knobs give placebos: Δ(c|control) minus Δ(c) at the other
non-carrier knobs, expected 0. Episodes ended by the stop file are left out (--include-stopped), and
rows whose run configuration differs are not pooled (--allow-mixed). Also reported: the raw Δ(c|k); top-1 accuracy cached vs observer against
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

import random  # noqa: E402

from knobladder.stats import bootstrap_ci, chance_mrr, mean, sign_flip_test, sign_test  # noqa: E402

CONTROLS = ("random", "sham")  # mirrors knobladder.episode, which needs torch
# Rows are pooled only if these agree: everything that sets what the model sees and how it is steered.
CONFIG = ("model", "hyperparams", "keep_thinking", "window", "steer_tokens", "steer_reply", "layer", "alpha", "dose",
          "passages_version", "readout_candidates", "screen", "calibration")


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


def injected(row):
    """Presses per knob at a level above 0 (a level-0 press injects nothing)."""
    n = {}
    for e in row["events"]:
        if e["kind"] == "press" and e["level"]:
            n[e["knob"]] = n.get(e["knob"], 0) + 1
    return n


def concept_knobs(row, min_presses):
    n = injected(row)
    return [d for d in row["discovery"] if d["label"] not in CONTROLS and n.get(d["knob"], 0) >= min_presses]


def rr(d, cond, concept):
    return 1 / d["readout"][cond]["ranks"][concept]


def delta(d, concept):
    return rr(d, "cached", concept) - rr(d, "observer", concept)


def delta_spec(row, d, c=None):
    """Δ_spec of concept c (default d's own) asked about knob d."""
    c = c or d["label"]
    others = [delta(o, c) for o in row["discovery"] if o["knob"] != d["knob"]]
    return delta(d, c) - mean(others)


def spec(row, d, cond):
    c = d["label"]
    others = [rr(o, cond, c) for o in row["discovery"] if o["knob"] != d["knob"]]
    return rr(d, cond, c) - mean(others)


def permutation_p(eps, stat, n=2000, seed=0):
    """Two-sided p for the mean over episodes of stat(row, knob, concept), against the null that moves each
    concept to a uniformly random knob of its episode (mean 0 when knobs are exchangeable)."""
    obs = mean([mean([stat(r, d, d["label"]) for d in ks]) for r, ks in eps])
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        null = mean([mean([stat(r, rng.choice(r["discovery"]), d["label"]) for d in ks]) for r, ks in eps])
        hits += abs(null) >= abs(obs) - 1e-12
    return (hits + 1) / (n + 1)


def placebo(row, ks, label):
    """Δ(c|control) - mean Δ(c|j) over the other non-carrier knobs j, per pressed concept c; None if unpressed."""
    ctrl = next((d for d in row["discovery"] if d["label"] == label), None)
    if ctrl is None or not injected(row).get(ctrl["knob"]):
        return None
    vals = []
    for d in ks:
        others = [delta(o, d["label"]) for o in row["discovery"] if o["knob"] not in (ctrl["knob"], d["knob"])]
        vals.append(delta(ctrl, d["label"]) - mean(others))
    return mean(vals)


def report(rows, min_presses, primary):
    print(f"\n== {'PRIMARY: ' if primary else 'secondary: '}concept knobs injected >= {min_presses} times ==")
    eps = [(r, concept_knobs(r, min_presses)) for r in rows]
    eps = [(r, ks) for r, ks in eps if ks]
    if not eps:
        print("no episodes with qualifying concept knobs")
        return
    dspec = [mean([delta_spec(r, d) for d in ks]) for r, ks in eps]
    raw = [mean([delta(d, d["label"]) for d in ks]) for r, ks in eps]
    perm = permutation_p(eps, lambda r, d, c: delta_spec(r, d, c))
    print(f"{'PRIMARY ' if primary else ''}Δ_spec (knob-specific cached - observer 1/rank): {fmt_ci(dspec)}; "
          f"{tests(dspec)}, label-permutation p={perm:.4f}")
    print(f"raw Δ(c|k), not corrected for priming: {fmt_ci(raw)}; {tests(raw)}")
    for cond in ("observer", "cached"):
        sp = [mean([spec(r, d, cond) for d in ks]) for r, ks in eps]
        pp = permutation_p(eps, lambda r, d, c, cond=cond: rr(d, cond, c) - mean(
            [rr(o, cond, c) for o in r["discovery"] if o["knob"] != d["knob"]]))
        print(f"spec_{cond} (knob-specific 1/rank{', from the transcript alone' if cond == 'observer' else ''}): "
              f"{fmt_ci(sp)}; label-permutation p={pp:.4f}")
    for label in CONTROLS:
        pl = [x for x in (placebo(r, ks, label) for r, ks in eps) if x is not None]
        if pl:
            print(f"placebo, {label} knob asked about the pressed concepts (expect 0): {fmt_ci(pl)}; {tests(pl)}")
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
    ap.add_argument("--include-stopped", action="store_true", help="keep episodes ended by the stop file")
    ap.add_argument("--allow-mixed", action="store_true", help="pool rows whose run configuration differs")
    args = ap.parse_args()
    rows = [r for r in load(args.files) if "readout" in r["discovery"][0]]
    stopped = [r for r in rows if any(e["kind"] == "stopped" for e in r["events"])]
    if stopped and not args.include_stopped:
        print(f"leaving out {len(stopped)} episodes ended by the stop file: "
              + ", ".join(f"{r['_file']} ep {r['episode']}" for r in stopped))
        rows = [r for r in rows if r not in stopped]
    mixed = [k for k in CONFIG if len({json.dumps(r.get(k), sort_keys=True) for r in rows}) > 1]
    if mixed:
        print(f"{'WARNING' if args.allow_mixed else 'ERROR'}: rows differ in {mixed}")
        if not args.allow_mixed:
            sys.exit(2)
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} episodes from {len(args.files)} files; models {models}")
    if len(models) > 1:
        print("WARNING: pooling across models")  # also an ERROR below unless --allow-mixed
    for key in ("provenance",):
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

    report(rows, 1, primary=True)
    report(rows, 2, primary=False)

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
