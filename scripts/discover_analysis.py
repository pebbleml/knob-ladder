"""Exploratory checks on a discover.py run (written 2026-10-31 before its finals were read; not pre-declared in R1.d).

    python3 scripts/discover_analysis.py runs/discover_<model>_<time>.json

1. Notes: share of turns whose NOTE claims no change (discover.NULL plus "identical" / "same as"), by knob kind, and neuron - sham per episode.
   This is in-context detection read from the text the model wrote, not the yes/no branch.
2. Level priming: the tool response shows the level, so the model may write differently at high levels whatever the
   knob is. For sham knobs (nothing injected), WRITE word count at level >= 8 vs <= 3; the same for neurons and random.
3. Per-knob hypotheses (cached): share NULL by kind; and which kind the overall hypothesis says the knobs act on.
4. Collapse (added after seeing, mid-run, writes shrink to "The" in the longest episodes): per episode, the first WRITE
   that ends mid-sentence (no final . ! ? or quote), the kind and level of the knob it was under, and how many later
   writes are <= 2 words.
5. Repeats (pre-declared for R1.d2 in discover.py; added after R1.d): per kind and level band, the rate of writes
   equal (whitespace-normalised) to an earlier write of the same episode.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knobladder.stats import bootstrap_ci, mean  # noqa: E402

NULL = re.compile(r"\b(nothing|no (noticeable |observable |obvious |clear |apparent |discernible |visible )?(change|effect|"
                  r"difference)s?|did ?n[o']t (change|do|seem to (change|do))|does ?n[o']t (change|do|seem to (change|do))|"
                  r"no-op|decoy|inert|inactive|placebo|dummy|untested|not tested|never tested|identical|same as)\b", re.I)


def main(path):
    d = json.loads(Path(path).read_text())
    eps = d["episodes"]
    kinds = ("concept" if any(x["kind"] == "concept" for ep in eps for x in ep["knobs"].values()) else "neuron", "random", "sham")
    print(f"{len(eps)} episodes; ended {d['secondary']['ended']}; turns mean {mean(d['secondary']['turns']):.1f}")
    for k, v in d["summary"].items():
        print(f"  {k}: {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] n={v['n']}")

    turns = [(ep, tr) for ep in eps for tr in ep["turns"] if tr["level"] > 0 and not tr["retired"]]
    print("\n1. NOTE says no change, by kind (turns with level > 0):")
    for kd in kinds:
        xs = [float(bool(NULL.search(tr["note"]))) for _, tr in turns if tr["kind"] == kd]
        print(f"  {kd:6s} {mean(xs):.2f} (n={len(xs)})")
    diff = []
    for ep in eps:
        by = {kd: [float(bool(NULL.search(tr["note"]))) for tr in ep["turns"] if tr["kind"] == kd and tr["level"] > 0]
              for kd in kinds}
        if by[kinds[0]] and by["sham"]:
            diff.append(mean(by["sham"]) - mean(by[kinds[0]]))
    print(f"  per episode, sham - neuron no-change rate: {mean(diff):+.3f} {bootstrap_ci(diff)} n={len(diff)}")

    print("\n2. WRITE word count by level (level priming check):")
    for kd in kinds:
        hi = [len(tr["write"].split()) for _, tr in turns if tr["kind"] == kd and tr["level"] >= 8]
        lo = [len(tr["write"].split()) for _, tr in turns if tr["kind"] == kd and tr["level"] <= 3]
        print(f"  {kd:6s} level>=8 {mean(hi) if hi else float('nan'):.1f} (n={len(hi)}) | level<=3 "
              f"{mean(lo) if lo else float('nan'):.1f} (n={len(lo)})")

    print("\n3. Per-knob hypotheses (tried knobs), share NULL:")
    for cond in ("cached", "observer"):
        row = []
        for kd in kinds:
            xs = [float(h["null"]) for ep in eps if "hyp" in ep for k, h in ep["hyp"][cond].items()
                  if ep["knobs"][k]["kind"] == kd and int(k) in ep["tried"]]
            row.append(f"{kd} {mean(xs):.2f} (n={len(xs)})")
        print(f"  {cond:8s} " + " | ".join(row))
    print(f"  overall hypothesis names the model itself: {d['secondary']['self_overall']}")

    print("\n4. Collapse: first write cut mid-sentence")
    first = {kd: 0 for kd in kinds}
    for ep in eps:
        ws = ep["turns"]
        cut = next((i for i, tr in enumerate(ws) if tr["write"].strip() and not re.search(r"[.!?\"')\]]\s*$", tr["write"].strip())), None)
        if cut is None:
            continue
        tr = ws[cut]
        first[tr["kind"]] += 1
        tiny = sum(len(t["write"].split()) <= 2 for t in ws[cut:])
        print(f"  ep{ep['episode']:2d}: turn {cut}/{len(ws)} under {tr['kind']} {tr['id']} L{tr['level']}: {tr['write'][-60:]!r}; "
              f"{tiny} later writes <= 2 words; ended {ep['done']}")
    print(f"  first cut by kind: {first} (board share 8/2/2)")

    print("\n5. Repeats of an earlier write, by kind and level band:")
    norm = lambda t: " ".join(t.split())
    bands = (("0", 0, 0), ("1-4", 1, 4), ("5-7", 5, 7), ("8-10", 8, 10))
    rows = {}
    for ep in eps:
        seen = set()
        for tr in ep["turns"]:
            b = next(n for n, lo, hi in bands if lo <= tr["level"] <= hi)
            rows.setdefault((tr["kind"], b), []).append(float(norm(tr["write"]) in seen))
            seen.add(norm(tr["write"]))
    for kd in kinds:
        print(f"  {kd:6s} " + " | ".join(f"L{n} {mean(rows[kd, n]):.2f} (n={len(rows[kd, n])})" if rows.get((kd, n)) else f"L{n} -"
                                       for n, _, _ in bands))


if __name__ == "__main__":
    main(sys.argv[1])
