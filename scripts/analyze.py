"""G3 analysis of R0 episodes: did the model find out what its knobs do, and does it know more than it wrote?

    python3 scripts/analyze.py runs/r0_Qwen3.8-27B_*.jsonl

Pure Python, runs locally on pulled jsonl. Scores come from the readout ("What does knob k
do? Answer with one word, then one sentence."): the first-token logits of the answer rank
the 32 knob concepts the model is never shown; rr = 1/rank. Every knob is asked about twice:
`cached` re-reads the episode with the steering on its passages, `observer` re-reads the same
tokens without it, so it knows only what the transcript says.

For a pressed concept knob k carrying c (pressed at a level above 0 at least once), and a
condition, spec(k) = rr(c|k) - mean over the other knobs j of rr(c|j): knowing c at knob k
rather than in general. Per episode, the mean over its pressed concept knobs.

- PRIMARY (declared 2026-10-29, before any Qwen3.8 episode): spec_observer, "did the model find
  out what the knobs do?". The passage is the same text on every press, so the transcript can
  only name the right concept for the right knob through what the model wrote. Bootstrap CI over
  episodes and a label-permutation null (each concept moved to a uniformly random knob of its
  episode, which gives mean 0 when knobs are exchangeable).
- SECONDARY: "does it know more than it wrote?". Δ_spec = spec_cached - spec_observer, against
  the yoked control (episode.yoked_transcript: the same presses, every reply "Okay."), whose
  Δ_spec is what the steered passages alone give the readout. Reported: Δ_spec of the episodes, of
  their yoked copies, and the paired difference free - yoked (sign-flip test, bootstrap CI).
- Controls: the random and sham knobs as placebos (Δ(c|control) minus Δ(c) at the other non-carrier
  knobs, expected 0); spec_observer of the yoked copies (expected 0: their transcript says nothing).

Only the injected >= 1 report is primary; >= 2 is secondary. Episodes ended by the stop file are left
out (--include-stopped), and rows whose run configuration differs are not pooled (--allow-mixed).
Also reported: the raw Δ(c|k); top-1 accuracy against chance; concept words in the observations;
how often the answer text names the concept; the multiple choice only if the run scored it (--choice).
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
    def spec_stat(cond):
        return lambda r, d, c: rr(d, cond, c) - mean([rr(o, cond, c) for o in r["discovery"] if o["knob"] != d["knob"]])

    obs = [mean([spec(r, d, "observer") for d in ks]) for r, ks in eps]
    print(f"{'PRIMARY ' if primary else ''}spec_observer (the transcript names the knob's concept at that knob): "
          f"{fmt_ci(obs)}; label-permutation p={permutation_p(eps, spec_stat('observer')):.4f}")
    cached = [mean([spec(r, d, "cached") for d in ks]) for r, ks in eps]
    print(f"spec_cached (the same, re-read with the steering): {fmt_ci(cached)}; "
          f"label-permutation p={permutation_p(eps, spec_stat('cached')):.4f}")
    dspec = [mean([delta_spec(r, d) for d in ks]) for r, ks in eps]
    perm = permutation_p(eps, lambda r, d, c: delta_spec(r, d, c))
    print(f"Δ_spec (spec_cached - spec_observer): {fmt_ci(dspec)}; {tests(dspec)}, label-permutation p={perm:.4f}")
    yk = [(r, ks) for r, ks in eps if r.get("yoked")]
    if yk:
        def yrow(r):
            return {**r, "discovery": r["yoked"]["discovery"]}

        def yknobs(r, ks):
            by = {d["knob"]: d for d in r["yoked"]["discovery"]}
            return [by[d["knob"]] for d in ks]

        free = [mean([delta_spec(r, d) for d in ks]) for r, ks in yk]
        yok = [mean([delta_spec(yrow(r), d) for d in yknobs(r, ks)]) for r, ks in yk]
        diff = [f - y for f, y in zip(free, yok)]
        yobs = [mean([spec(yrow(r), d, "observer") for d in yknobs(r, ks)]) for r, ks in yk]
        print(f"yoked Δ_spec (the same presses, fixed notes for replies): {fmt_ci(yok)}; {tests(yok)}")
        print(f"{'SECONDARY ' if primary else ''}free - yoked Δ_spec (paired): {fmt_ci(diff)}; {tests(diff)}")
        print(f"control, yoked spec_observer (expect 0): {fmt_ci(yobs)}")
    else:
        print("no yoked control in these rows")
    raw = [mean([delta(d, d["label"]) for d in ks]) for r, ks in eps]
    print(f"raw Δ(c|k), not corrected for priming: {fmt_ci(raw)}; {tests(raw)}")
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
