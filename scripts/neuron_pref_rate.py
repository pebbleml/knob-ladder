r"""P7r does it just prefer neurons that make it write better or pleasanter text? (2026-11-11). This docstring is the binding
pre-registration, written after P7k's output and before any P7r output.

    python3 scripts/neuron_pref_rate.py --source runs/neuron_pref_p7m_Qwen3.8-27B_20261110-231500.json
    python3 scripts/neuron_pref_rate.py --source runs/neuron_pref_wide_Qwen3-32B_20261110-213212.json
    python3 scripts/neuron_pref_rate.py --source <either> --limit 64 --tag _smoke

Why. P7c and P7x/P7m-x show the preference follows the steered state more than the words, but the text-only arm also
carries part of it (C2 > 0 on P7x and P7m-x). The simplest account of the text part is that the model prefers neurons
whose writing it finds better or pleasanter. P7r measures that directly and asks how much of the preference it explains.

Design. No steering anywhere. Every part (part 1 and part 2) of every stored session of the source run (P7m: 60 random
neurons on Qwen3.8-27B; P7w: 120 Goodfire neurons on Qwen3-32B) is shown to the same model in a fresh conversation:
    user: RATE[q] + the passage
for two questions q:
    quality: "Here is a short passage. How well written is it, from 1 (very poor) to 9 (excellent)? Answer with the number only."
    mood:    "Here is a short passage. How pleasant is its mood, from 1 (very unpleasant) to 9 (very pleasant)? Answer with the number only."
Rating = expected value over the digit tokens 1-9 (renormalised). Per item, r_q(item) = mean rating of the parts written
under it, minus the mean over shams' parts (so 0 = like writing with nothing on).

Primary (one test per source, Bonferroni over the 2 sources, 97.5% CIs, bootstrap over neuron items, 4000 resamples,
seed 0): Spearman over neuron items of the source's stored pref_cached with r_quality. Supported iff the CI lower bound > 0.
Secondary (no claims): the same with r_mood, and with pref_text; partial Spearman of pref_cached with r_quality given
the item's dNLL; mean r_q for neurons, randoms, shams; digit mass.
Readings fixed in advance: supported = part of the preference tracks how well the neuron made it write, by its own
judgement. Not supported = the ranking is not explained by its own rating of the writing (consistent with P7c's state
result). Either way this does not test the state component, which P7c already isolated.

P7r-x out-of-sample confirmation (2026-11-11, written after P7r's output and before any P7mw or P7r-x output). Run on
P7mw's sessions (120 new random neurons, Qwen3.8-27B):

    python3 scripts/neuron_pref_rate.py --source <runs/neuron_pref_p7mw_Qwen3.8-27B_*.json> --tag _p7rx

- Primary (one test, 97.5% CI as coded (LEVEL), conservative for a single test): Spearman of pref_cached with r_quality, CI
  lower bound > 0. Supported = on the 27B, preference tracks its own rating of how well each neuron let it write.
- Secondary: as in P7r.
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
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import GEN, user  # noqa: E402
from reason_step import spearman  # noqa: E402

RATE = {"quality": "Here is a short passage. How well written is it, from 1 (very poor) to 9 (excellent)? Answer with the number only.\n\n",
        "mood": "Here is a short passage. How pleasant is its mood, from 1 (very unpleasant) to 9 (very pleasant)? Answer with the number only.\n\n"}
LEVEL = 0.975


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(o):
        r[i] = float(k)
    return r


def partial_spearman(x, y, z):
    rx, ry, rz = (torch.tensor(rank(v)) for v in (x, y, z))
    def resid(a, b):
        b1 = torch.stack([b, torch.ones_like(b)], 1)
        return a - b1 @ torch.linalg.lstsq(b1, a.unsqueeze(1)).solution.squeeze(1)
    ex, ey = resid(rx, rz), resid(ry, rz)
    return float(torch.corrcoef(torch.stack([ex, ey]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.load(open(ROOT / args.source if not Path(args.source).is_absolute() else args.source))
    model = src["args"]["model"]
    items = {it["id"]: it for it in src["items"]}
    sessions = src["sessions"][: args.limit] if args.limit else src["sessions"]
    lm = SteeredLM(model, dtype=torch.bfloat16)
    out = ROOT / "runs" / f"neuron_pref_rate{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log(f"{len(sessions)} sessions from {args.source} -> {out}")
    digits = []
    for d in "123456789":
        t = [t for t in lm.encode(d)[:1] if lm.tok.decode([t]).strip() == d]
        digits.append(t[0])
    vals = torch.arange(1, 10, dtype=torch.float32)

    parts = [(s["first"], s["part1"]) for s in sessions] + [(s["second"], s["part2"]) for s in sessions]
    ratings = {q: [] for q in RATE}
    mass = {q: [] for q in RATE}
    for q in RATE:
        prompts = [lm.encode(user(RATE[q] + txt.strip()) + GEN) for _, txt in parts]
        for k in range(0, len(prompts), args.batch):
            f, _ = lm.generate_batch(prompts[k:k + args.batch], [[] for _ in prompts[k:k + args.batch]], 1, temperature=0, first_logits=True)
            for r in f:
                lp = torch.log_softmax(r.float(), -1)[digits]
                p = torch.softmax(lp, 0)
                ratings[q].append(float((p * vals).sum()))
                mass[q].append(float(torch.exp(torch.logsumexp(lp, 0))))
        log(f"rated {q}")

    per = {q: {} for q in RATE}
    for q in RATE:
        for (iid, _), r in zip(parts, ratings[q]):
            per[q].setdefault(iid, []).append(r)
    sham_ids = [i for i, it in items.items() if it["kind"] == "sham" and i in per["quality"]]
    base = {q: mean([r for i in sham_ids for r in per[q][i]]) for q in RATE}
    R = {q: {i: mean(v) - base[q] for i, v in per[q].items()} for q in RATE}
    nid = [i for i, it in items.items() if it["kind"] == "neuron" and i in R["quality"] and it.get("pref_cached") is not None]

    def boot_sp(xf, yf):
        r = random.Random(0)
        est = spearman([xf(i) for i in nid], [yf(i) for i in nid])
        bs = []
        for _ in range(4000):
            smp = [r.choice(nid) for _ in nid]
            bs.append(spearman([xf(i) for i in smp], [yf(i) for i in smp]))
        bs = sorted(b for b in bs if b == b)
        a = (1 - LEVEL) / 2
        return {"est": est, "ci": [bs[int(a * len(bs))], bs[int((1 - a) * len(bs)) - 1]], "n": len(nid)}

    pc = lambda i: items[i]["pref_cached"]
    pt = lambda i: items[i]["pref_text"]
    primary = boot_sp(pc, lambda i: R["quality"][i])
    dn = [items[i]["dnll"][str(src["summary"]["f_star"])] if isinstance(items[i].get("dnll"), dict) else 0.0 for i in nid]
    summary = {
        "n_parts": len(parts), "base_sham": base, "P_pref_vs_quality": primary, "P": primary["ci"][0] > 0,
        "secondary": {
            "pref_vs_mood": boot_sp(pc, lambda i: R["mood"][i]),
            "pref_text_vs_quality": boot_sp(pt, lambda i: R["quality"][i]),
            "pref_text_vs_mood": boot_sp(pt, lambda i: R["mood"][i]),
            "partial_pref_quality_given_dnll": partial_spearman([pc(i) for i in nid], [R["quality"][i] for i in nid], dn),
            "quality_vs_mood_items": spearman([R["quality"][i] for i in nid], [R["mood"][i] for i in nid]),
            "mean_r": {q: {kind: mean([R[q][i] for i, it in items.items() if it["kind"] == kind and i in R[q]])
                           for kind in ("neuron", "random", "sham")} for q in RATE},
            "digit_mass": {q: mean(mass[q]) for q in RATE},
        },
        "items": [(it["layer"], it["index"], round(it["pref_cached"], 3), round(R["quality"][i], 3), round(R["mood"][i], 3), it["title"])
                  for i, it in sorted(items.items()) if i in nid],
    }
    json.dump({"args": vars(args), "provenance": lm.provenance(), "source": args.source, "prompts": RATE,
               "ratings": {q: ratings[q] for q in RATE}, "part_items": [p[0] for p in parts], "summary": summary},
              open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    log(f"P pref~quality {primary['est']:+.3f} {primary['ci']} | mood {summary['secondary']['pref_vs_mood']['est']:+.3f} "
        f"{summary['secondary']['pref_vs_mood']['ci']}")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
