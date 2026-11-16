r"""P7q does the neuron's state change how good it judges its own writing? (2026-11-11). This docstring is the binding
pre-registration, written after P7r-x's output and before any P7q output.

    python3 scripts/neuron_selfrate.py --source runs/neuron_pref_p7mw_Qwen3.8-27B_20261111-024104.json
    python3 scripts/neuron_selfrate.py --source <same> --limit 64 --tag _smoke

Why. On the 27B, neurons it prefers are those whose writing it rates as better when shown the text cold (P7r +0.46,
P7r-x +0.43). But the preference rides mostly on the steered state, not the words (P7m-x swap test). One account joins
the two: the state itself changes how good the model takes its own writing to be. P7q measures the state's effect on its
in-context judgement of its own part and asks whether that tracks the state's share of the preference.

Design. No new text. Each stored P7mw session is rebuilt up to the end of part 1 exactly as in P7 (the same intro with
two settings, the same task line, the stored part 1 re-encoded), then:
    user: "Before part 2: how well written is part 1, from 1 (very poor) to 9 (excellent)? Answer with the number only."
Rating = expected digit 1-9. Arms: cached = the 'first' item's steering on the part 1 span (P7's cached arm, same f*,
scale, directions); text = no steering. Per item: q_arm = mean rating of its part-1 passages; dq = q_cached - q_text
(the state's effect on judged quality). Shams inject nothing (dq = 0 by construction).

Primary (one test, 95% CI, bootstrap over neuron items, 4000 resamples, seed 0): Spearman over neuron items of
dq with (pref_cached - pref_text) from the source, i.e. the state's effect on judged quality vs the state's share of the
preference. Supported iff the CI lower bound > 0.
Secondary (no claims): mean dq for neurons and randoms (does the state lower judged quality on average?), with CI;
Spearman of pref_cached with q_cached and with q_text; Spearman of dq with the source item's dNLL; digit mass.
Readings fixed in advance: supported = the steered state shifts how good it judges its writing, and neurons whose state
makes it judge its writing worse are the ones whose state also pushes the choice away; a candidate mechanism for the
state part of the preference. Not supported = the state part of the preference is not carried by its in-context
quality judgement.
Also run, exploratory and with no claims: the same on P7w (Qwen3-32B, 120 Goodfire neurons), where P7r found no quality link.

P7q-x replication (2026-11-11, written after P7q's output and before any P7q-x output). P7q found dq tracks the state's
share of the preference on P7mw's 120 neurons (+0.54 [0.38, 0.68]). P7q-x repeats it unchanged on the independent P7m
sample (60 other random neurons, Qwen3.8-27B) and on P7m-x (its top 15 and bottom 15 on new tasks):

    python3 scripts/neuron_selfrate.py --source runs/neuron_pref_p7m_Qwen3.8-27B_20261110-231500.json --tag _p7qx
    python3 scripts/neuron_selfrate.py --source runs/neuron_pref_p7mx_Qwen3.8-27B_20261111-000253.json --tag _p7qx2

- Primary (one test): the same Spearman on P7m, CI lower bound > 0. The P7m-x run is secondary (30 selected neurons).

P7q-c control questions (2026-11-11, written after P7q-x's output and before any P7q-c output). P7q and P7q-x show dq
(quality) tracks the state's share of the preference on the 27B (+0.54, +0.57, +0.65). A confound: the state might lower
any number it gives, not its judgement of quality. P7q-c reruns P7q on P7mw with two other questions in place of ASKQ:
    length:    "Before part 2: how long is part 1, from 1 (very short) to 9 (very long)? Answer with the number only."
    satisfied: "Before part 2: how satisfied are you with part 1, from 1 (not at all) to 9 (completely)? Answer with the number only."

    python3 scripts/neuron_selfrate.py --source runs/neuron_pref_p7mw_Qwen3.8-27B_20261111-024104.json --ask length --tag _p7qc_length
    python3 scripts/neuron_selfrate.py --source <same> --ask satisfied --tag _p7qc_satisfied

- Primary (one test, 95% CI, bootstrap over neuron items, 4000 resamples, seed 0, items resampled jointly): the
  difference Spearman(dq_quality, state share) - Spearman(dq_length, state share), using P7q's stored dq_quality.
  Supported iff the CI lower bound > 0, i.e. the quality judgement tracks the state's share of the preference more than
  a length judgement does. (Computed by scripts/neuron_selfrate_compare.py from the three output files.)
- Secondary: the length and satisfied Spearmans on their own; mean dq per question.
- Readings: supported = the link is specific to judged quality (not a general shift in numbers). Not supported = P7q may
  reflect a general effect of the state on numeric answers.

P7q-c2 replication (2026-11-11, written after P7q-c's output and before any P7q-c2 output). P7q-c's contrast on the
independent P7m sample (60 neurons), using P7q-x's quality file:

    python3 scripts/neuron_selfrate.py --source runs/neuron_pref_p7m_Qwen3.8-27B_20261110-231500.json --ask length --tag _p7qc2_length
    python3 scripts/neuron_selfrate.py --source <same> --ask satisfied --tag _p7qc2_satisfied
    python3 scripts/neuron_selfrate_compare.py --source <same> --quality runs/neuron_selfrate_p7qx_Qwen3.8-27B_20261111-043734.json ...

- Primary: Spearman(quality) - Spearman(length), CI lower bound > 0, as in P7q-c.

P7q-8 on Qwen3-8B (2026-11-11, written after P7s-3's output and before any P7q-8 output). P7s-3 found a stable
preference on the 8B (+0.46 [0.25, 0.62], 120 neurons). Does the 8B show the 27B's self-evaluation link? P7q and P7q-c
unchanged on P7s-3's sessions, three questions (quality, length, satisfied):

    python3 scripts/neuron_selfrate.py --source <runs/neuron_pref_p7s3_Qwen3-8B_*.json> --ask {quality,length,satisfied} --tag _p7q8_{ask}
    python3 scripts/neuron_selfrate_compare.py --source <same> --quality ... --length ... --satisfied ...

- Primary (one test): Spearman(quality) - Spearman(length), CI lower bound > 0, as in P7q-c.
- Secondary: each Spearman on its own (P7q's primary statistic for quality); mean dq per question.
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
from neuron_pref import INTRO, PART1  # noqa: E402
from reason_step import spearman  # noqa: E402

ASKS = {"quality": "Before part 2: how well written is part 1, from 1 (very poor) to 9 (excellent)? Answer with the number only.",
        "length": "Before part 2: how long is part 1, from 1 (very short) to 9 (very long)? Answer with the number only.",
        "satisfied": "Before part 2: how satisfied are you with part 1, from 1 (not at all) to 9 (completely)? Answer with the number only."}
ASKQ = ASKS["quality"]
ARMS = ("cached", "text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ask", default="quality", choices=sorted(ASKS))
    args = ap.parse_args()
    ask = ASKS[args.ask]
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
    out = ROOT / "runs" / f"neuron_selfrate{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log(f"{len(sessions)} sessions from {args.source}, f* {fstar} -> {out}")
    digits = []
    for d in "123456789":
        digits.append([t for t in lm.encode(d)[:1] if lm.tok.decode([t]).strip() == d][0])
    vals = torch.arange(1, 10, dtype=torch.float32)

    prompts, spans = [], []
    for s in sessions:
        p1 = lm.encode(user(INTRO.format(a=s["la"], b=s["lb"]) + PART1.format(a=s["la"], task=tasks[s["task"]])) + GEN)
        o1 = lm.encode(s["part1"])
        prompts.append(p1 + o1 + lm.encode(END + user(ask) + GEN))
        spans.append((len(p1), len(p1) + len(o1)))

    for arm in ARMS:
        scheds = []
        for s, (a, b) in zip(sessions, spans):
            it = items[s["first"]]
            if arm == "text" or it["kind"] == "sham":
                scheds.append([])
            else:
                scheds.append([Intervention(it["layer"], (it["dir"] * fstar * scale[it["layer"]]).to(lm.model.dtype), a, b)])
        res = []
        for k in range(0, len(prompts), args.batch):
            f, _ = lm.generate_batch(prompts[k:k + args.batch], scheds[k:k + args.batch], 1, temperature=0, first_logits=True)
            res += [torch.log_softmax(r.float(), -1) for r in f]
        for s, lp in zip(sessions, res):
            d = lp[digits]
            s[f"q_{arm}"] = float((torch.softmax(d, 0) * vals).sum())
            s[f"mass_{arm}"] = float(torch.exp(torch.logsumexp(d, 0)))
        log(f"arm {arm}")

    q = {arm: {} for arm in ARMS}
    for s in sessions:
        for arm in ARMS:
            q[arm].setdefault(s["first"], []).append(s[f"q_{arm}"])
    q = {arm: {i: mean(v) for i, v in d.items()} for arm, d in q.items()}
    dq = {i: q["cached"][i] - q["text"][i] for i in q["cached"]}
    nid = [i for i, it in items.items() if it["kind"] == "neuron" and i in dq and it.get("pref_cached") is not None]
    rid = [i for i, it in items.items() if it["kind"] == "random" and i in dq]
    state_pref = {i: items[i]["pref_cached"] - items[i]["pref_text"] for i in nid}

    def boot_sp(xf, yf):
        r = random.Random(0)
        bs = []
        for _ in range(4000):
            smp = [r.choice(nid) for _ in nid]
            bs.append(spearman([xf(i) for i in smp], [yf(i) for i in smp]))
        bs = sorted(b for b in bs if b == b)
        return {"est": spearman([xf(i) for i in nid], [yf(i) for i in nid]), "ci": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]], "n": len(nid)}

    def boot_mean(ids):
        r = random.Random(0)
        bs = sorted(mean([dq[r.choice(ids)] for _ in ids]) for _ in range(4000))
        return {"est": mean([dq[i] for i in ids]), "ci": [bs[100], bs[3899]]} if ids else None

    primary = boot_sp(lambda i: dq[i], lambda i: state_pref[i])
    summary = {
        "dq_items": {str(i): dq[i] for i in nid}, "state_pref_items": {str(i): state_pref[i] for i in nid},
        "n_sessions": len(sessions), "f_star": fstar, "P_dq_vs_state_pref": primary, "P": primary["ci"][0] > 0,
        "secondary": {
            "mean_dq": {"neuron": boot_mean(nid), "random": boot_mean(rid)},
            "pref_cached_vs_q_cached": spearman([items[i]["pref_cached"] for i in nid], [q["cached"][i] for i in nid]),
            "pref_cached_vs_q_text": spearman([items[i]["pref_cached"] for i in nid], [q["text"][i] for i in nid]),
            "pref_cached_vs_dq": spearman([items[i]["pref_cached"] for i in nid], [dq[i] for i in nid]),
            "dq_vs_dnll": spearman([dq[i] for i in nid], [items[i]["dnll"][str(fstar)] for i in nid]),
            "mean_q": {arm: mean([s[f"q_{arm}"] for s in sessions]) for arm in ARMS},
            "digit_mass": {arm: mean([s[f"mass_{arm}"] for s in sessions]) for arm in ARMS},
        },
        "items": [(it["layer"], it["index"], round(it["pref_cached"], 3), round(state_pref[i], 3), round(q["text"][i], 3), round(dq[i], 3))
                  for i, it in sorted(items.items()) if i in nid],
    }
    for it in items.values():
        it.pop("dir", None)
    keep = ("pair", "first", "la", "lb", "task", "order") + tuple(f"{k}_{a}" for a in ARMS for k in ("q", "mass"))
    json.dump({"args": vars(args), "provenance": lm.provenance(), "source": args.source, "prompts": {"ask": ask, "which": args.ask},
               "sessions": [{k: s.get(k) for k in keep} for s in sessions], "summary": summary}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    log(f"P dq~state_pref {primary['est']:+.3f} {primary['ci']} | mean dq neurons {summary['secondary']['mean_dq']['neuron']}")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
