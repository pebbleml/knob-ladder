"""P7q-c primary (pre-registered in scripts/neuron_selfrate.py): Spearman(dq_quality, state share) minus
Spearman(dq_length, state share) over the same neuron items, bootstrap over items jointly (4000, seed 0), 95% CI.

    python3 scripts/neuron_selfrate_compare.py --source runs/neuron_pref_p7mw_... --quality <p7q file> --length <file> --satisfied <file>
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reason_step import spearman  # noqa: E402


def dq(path):
    d = json.load(open(ROOT / path))
    acc = {}
    for s in d["sessions"]:
        acc.setdefault(s["first"], []).append(s["q_cached"] - s["q_text"])
    return {i: sum(v) / len(v) for i, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--quality", required=True)
    ap.add_argument("--length", required=True)
    ap.add_argument("--satisfied", required=True)
    a = ap.parse_args()
    src = json.load(open(ROOT / a.source))
    items = {it["id"]: it for it in src["items"]}
    D = {k: dq(getattr(a, k)) for k in ("quality", "length", "satisfied")}
    nid = [i for i, it in items.items() if it["kind"] == "neuron" and it.get("pref_cached") is not None and all(i in d for d in D.values())]
    sp = {i: items[i]["pref_cached"] - items[i]["pref_text"] for i in nid}
    rho = lambda k, ids: spearman([D[k][i] for i in ids], [sp[i] for i in ids])
    r = random.Random(0)
    bs = []
    for _ in range(4000):
        smp = [r.choice(nid) for _ in nid]
        bs.append(rho("quality", smp) - rho("length", smp))
    bs.sort()
    est = rho("quality", nid) - rho("length", nid)
    out = {"n": len(nid), "primary_quality_minus_length": {"est": est, "ci": [bs[100], bs[3899]]}, "supported": bs[100] > 0,
           "rho": {k: rho(k, nid) for k in D},
           "mean_dq": {k: sum(D[k][i] for i in nid) / len(nid) for k in D}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "runs" / f"neuron_selfrate_compare_{Path(a.length).stem}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
