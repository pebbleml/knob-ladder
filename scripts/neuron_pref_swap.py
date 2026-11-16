r"""P7c neuron preference, state vs text (2026-11-10). This docstring is the binding pre-registration, written after P7b's
output and before any P7c output.

    python3 scripts/neuron_pref_swap.py --source runs/neuron_pref_p7b_Qwen3-32B_20261110-210055.json
    python3 scripts/neuron_pref_swap.py --source <same> --limit 64 --tag _smoke

Why. P7b (WIRK record for P7b) replicated P7's top-vs-bottom neuron ranking on new tasks with the steered state present
at choice time (C1 +0.073 [0.032, 0.119]) but not reliably from the same text read without steering (C2 +0.022
[-0.006, 0.053]). The arms differ in two ways at once, so P7c holds the text fixed and moves only the state.

Design. Every P7b session is rebuilt from its stored text: the same intro, part 1 and part 2 (re-encoded from the
stored decoded text), and the same choice question and labels. Part 1 was written under item X ('first') and part 2
under item Y ('second'). At the choice, nothing is generated except the first answer token; the steering below is
applied over the part spans while the transcript is read (the same mechanism as P7's cached arm):
    match: X's steering on part 1, Y's on part 2 (P7b's cached arm, re-encoded)
    swap:  Y's steering on part 1, X's on part 2 (each text now carries the other item's state)
    none:  no steering (P7b's text arm, re-encoded)
Same items, f* = 0.5, scales and directions as P7b (random directions from their stored seeds; shams inject nothing).

Scores. For each comparison and arm, p = mean over both orders of P(choose the label of the part written under the
item on the x side). Per item and arm, pref = mean over its comparisons of (p - 0.5), item on the x side. The contrast
K(arm) = mean pref(top 10) - mean pref(bottom 10), groups as in P7b.

Primary (one test, 95% CI, bootstrap over items within each group, 4000 resamples, seed 0):
  S = K(match) - K(swap). Supported iff the CI lower bound > 0. If the preference rode only on the text, moving the
  state to the other text would not change it (S = 0). A positive S means the steered state itself pulls the choice
  toward the item whose state it is.
Secondary (no claims): K for each arm; K(match) - K(none) and K(none) - K(swap); Spearman of per-item pref between
arms; agreement of re-encoded match with P7b's stored cached p (sanity: Pearson over sessions); position and label
bias per arm.
Readings fixed in advance: S > 0 with K(swap) < 0 = the state dominates the text (preference follows the state across
texts). S > 0 with K(swap) >= 0 = state and text both contribute. S not supported = P7b's cached-vs-text gap is not
evidence for a state effect.
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

from knobladder.knobs import neuron_direction, random_direction, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from neuron_pref import CHOICE, INTRO, PART1, PART2  # noqa: E402
from reason_step import spearman  # noqa: E402

ARMS = ("match", "swap", "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="sessions (smoke)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.load(open(ROOT / args.source if not Path(args.source).is_absolute() else args.source))
    model = src["args"]["model"]
    fstar = src["summary"]["f_star"]
    tasks = src["prompts"]["tasks"]
    items = {it["id"]: it for it in src["items"]}
    sessions = src["sessions"][: args.limit] if args.limit else src["sessions"]
    lm = SteeredLM(model, dtype=torch.bfloat16)
    scale = {int(k): v for k, v in src["scale"].items()}
    for it in items.values():
        if it["kind"] == "neuron":
            it["dir"] = unit(neuron_direction(lm, it["layer"], it["index"]))
        elif it["kind"] == "random":
            it["dir"] = unit(random_direction(lm.d_model, it["seed"]))
    out = ROOT / "runs" / f"neuron_pref_swap{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log(f"{len(sessions)} sessions from {args.source}, f* {fstar} -> {out}")

    def win(iid, start, end):
        it = items[iid]
        if it["kind"] == "sham":
            return []
        return [Intervention(it["layer"], (it["dir"] * fstar * scale[it["layer"]]).to(lm.model.dtype), start, end)]

    prompts, spans = [], []
    for s in sessions:
        p1 = lm.encode(user(INTRO.format(a=s["la"], b=s["lb"]) + PART1.format(a=s["la"], task=tasks[s["task"]])) + GEN)
        o1 = lm.encode(s["part1"])
        p2 = p1 + o1 + lm.encode(END + user(PART2.format(b=s["lb"])) + GEN)
        o2 = lm.encode(s["part2"])
        pc = p2 + o2 + lm.encode(END + user(CHOICE.format(a=s["la"], b=s["lb"])) + GEN)
        prompts.append(pc)
        spans.append(((len(p1), len(p1) + len(o1)), (len(p2), len(p2) + len(o2))))

    def letter_ids(L):
        return sorted({t for t in (lm.encode(L)[0], lm.encode(" " + L)[0]) if lm.tok.decode([t]).strip() == L})

    for arm in ARMS:
        scheds = []
        for s, (a, b) in zip(sessions, spans):
            if arm == "none":
                scheds.append([])
            else:
                f1, f2 = (s["first"], s["second"]) if arm == "match" else (s["second"], s["first"])
                scheds.append(win(f1, *a) + win(f2, *b))
        res = []
        for k in range(0, len(prompts), args.batch):
            f, _ = lm.generate_batch(prompts[k:k + args.batch], scheds[k:k + args.batch], 1, temperature=0, first_logits=True)
            res += [torch.log_softmax(r.float(), -1) for r in f]
        for s, lp in zip(sessions, res):
            a = torch.logsumexp(lp[letter_ids(s["la"])], 0)
            b = torch.logsumexp(lp[letter_ids(s["lb"])], 0)
            pf = float(torch.sigmoid(a - b))  # la labels part 1, written under 'first'
            s[f"p_first_{arm}"] = pf
            s[f"p_x_{arm}"] = pf if s["first"] == s["x"] else 1 - pf  # choosing the text written under x
            s[f"mass_{arm}"] = float(torch.exp(torch.logsumexp(torch.stack([a, b]), 0)))
        log(f"arm {arm}")

    comp = {}
    for s in sessions:
        comp.setdefault(s["pair"], []).append(s)
    C = [{"x": ss[0]["x"], "y": ss[0]["y"], **{f"p_{arm}": mean([s[f"p_x_{arm}"] for s in ss]) for arm in ARMS}}
         for ss in comp.values()]

    def pref(arm):
        o = {}
        for c in C:
            o.setdefault(c["x"], []).append(c[f"p_{arm}"] - 0.5)
            o.setdefault(c["y"], []).append(0.5 - c[f"p_{arm}"])
        return {k: mean(v) for k, v in o.items()}

    P = {arm: pref(arm) for arm in ARMS}
    T = [i for i, it in items.items() if it.get("group") == "top" and i in P["match"]]
    B = [i for i, it in items.items() if it.get("group") == "bottom" and i in P["match"]]
    K = lambda arm, t, b: mean([P[arm][i] for i in t]) - mean([P[arm][i] for i in b])

    def boot(fn, level=0.95):
        r = random.Random(0)
        bs = sorted(fn([r.choice(T) for _ in T], [r.choice(B) for _ in B]) for _ in range(4000))
        a = (1 - level) / 2
        return {"est": fn(T, B), "ci": [bs[int(a * len(bs))], bs[int((1 - a) * len(bs)) - 1]]}

    S = boot(lambda t, b: K("match", t, b) - K("swap", t, b))
    nid = [i for i, it in items.items() if it["kind"] == "neuron" and i in P["match"]]
    ref = [s["p_first_cached"] for s in sessions]
    summary = {
        "n_sessions": len(sessions), "n_pairs": len(C), "S_match_minus_swap": S, "S": S["ci"][0] > 0,
        "secondary": {
            "K": {arm: boot(lambda t, b, arm=arm: K(arm, t, b)) for arm in ARMS},
            "match_minus_none": boot(lambda t, b: K("match", t, b) - K("none", t, b)),
            "none_minus_swap": boot(lambda t, b: K("none", t, b) - K("swap", t, b)),
            "spearman_items": {f"{a}~{b}": spearman([P[a][i] for i in nid], [P[b][i] for i in nid])
                               for a, b in (("match", "none"), ("match", "swap"), ("swap", "none"))},
            "sanity_pearson_match_vs_p7b_cached": float(torch.corrcoef(torch.tensor([ref, [s["p_first_match"] for s in sessions]]))[0, 1]),
            "position_bias_p_first": {arm: mean([s[f"p_first_{arm}"] for s in sessions]) for arm in ARMS},
            "label_mass": {arm: mean([s[f"mass_{arm}"] for s in sessions]) for arm in ARMS},
        },
        "items": [(it["layer"], it["index"], it.get("group"), *(round(P[a].get(i, float("nan")), 3) for a in ARMS), it["title"])
                  for i, it in sorted(items.items()) if it["kind"] == "neuron"],
    }
    for it in items.values():
        it.pop("dir", None)
    keep = ("pair", "x", "y", "first", "second", "la", "lb", "task", "order") + tuple(f"{k}_{a}" for a in ARMS for k in ("p_first", "p_x", "mass"))
    json.dump({"args": vars(args), "provenance": lm.provenance(), "source": args.source, "arms": ARMS,
               "items": list(items.values()), "comparisons": C, "sessions": [{k: s[k] for k in keep} for s in sessions],
               "summary": summary}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    log(f"S match-swap {S['est']:+.4f} {S['ci']} | K {({a: round(v['est'], 4) for a, v in summary['secondary']['K'].items()})}")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
