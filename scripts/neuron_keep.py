r"""P7k neuron preference as a keep-or-stop decision (2026-11-11). This docstring is the binding pre-registration, written
after P7s-2's output and before any P7k output.

    python3 scripts/neuron_keep.py --source runs/neuron_pref_p7mx_Qwen3.8-27B_20261111-000253.json
    python3 scripts/neuron_keep.py --source runs/neuron_pref_p7x_Qwen3-32B_<stamp>.json
    python3 scripts/neuron_keep.py --source <either> --limit 64 --tag _smoke

Why. P7-P7m-x show stable preferences between two neurons when the model picks one of two settings. That is a relative,
two-option format. P7k asks whether the same ranking shows up in an absolute decision about a single neuron's state:
after writing a part with one setting on, does it want that setting kept on? If the preference is about the neuron's
state and not about the pairwise format, the top neurons should be kept more often than the bottom neurons.

Design. No new text is generated. Every stored session of the source run (P7m-x on Qwen3.8-27B, P7x on Qwen3-32B) has a
part 1 written under one item ('first'). Each session is rebuilt as a single-setting conversation:
    user:      KINTRO (setting {a} is on) + the same task line
    assistant: the stored part 1 text, re-encoded
    user:      ASK, one of two wordings, alternating by session index so each item gets both:
               keep: "Setting {a} can stay on for part 2 or be switched off. Keep it on? Answer yes or no only."
               off:  "Setting {a} can stay on for part 2 or be switched off. Switch it off? Answer yes or no only."
P(keep) = P(yes) / (P(yes) + P(no)) for 'keep', and P(no) / (P(yes) + P(no)) for 'off' (yes/no token sets as for letters).
Arms at the question: cached = the item's steering on the part 1 span (as in P7's cached arm, same f*, scale and
directions as the source); text = no steering. Shams inject nothing in either arm.

Scores. keep(item, arm) = mean over its sessions of P(keep). K(arm) = mean keep(top) - mean keep(bottom), groups as
stored in the source items (15 vs 15).

Primary (one test per source, Bonferroni over the 2 sources, 97.5% CIs, bootstrap over items within each group, 4000
resamples, seed 0): K(cached) > 0, supported iff the CI lower bound > 0.
Secondary (no claims): K(text); keep(neurons) - keep(shams) per arm; Spearman of keep (cached) with the source's pairwise
cached preference over all neurons; P(keep) by wording; yes/no mass.
Readings fixed in advance: K(cached) > 0 = the pairwise ranking carries to an absolute keep-or-stop decision about one
neuron's state. Not supported = the preference is specific to choosing between two settings (or too weak to show in
this format); it does not undo P7-P7m-x.
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
from reason_step import spearman  # noqa: E402

KINTRO = ("We are going to write a short piece in two parts. A setting, {a}, is available. Part 1 is written with setting "
          "{a} on.\n\nPart 1 (setting {a} is on). {task} Write about 80 words.")
ASK = {"keep": "Setting {a} can stay on for part 2 or be switched off. Keep it on? Answer yes or no only.",
       "off": "Setting {a} can stay on for part 2 or be switched off. Switch it off? Answer yes or no only."}
ARMS = ("cached", "text")
LEVEL = 0.975


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
    out = ROOT / "runs" / f"neuron_keep{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log(f"{len(sessions)} sessions from {args.source}, f* {fstar} -> {out}")

    def ids(word):
        return sorted({t for w in (word, " " + word, word.capitalize(), " " + word.capitalize())
                       for t in lm.encode(w)[:1] if lm.tok.decode([t]).strip().lower() == word})
    yes, no = ids("yes"), ids("no")

    prompts, spans = [], []
    for k, s in enumerate(sessions):
        s["wording"] = "keep" if k % 2 == 0 else "off"
        p1 = lm.encode(user(KINTRO.format(a=s["la"], task=tasks[s["task"]])) + GEN)
        o1 = lm.encode(s["part1"])
        prompts.append(p1 + o1 + lm.encode(END + user(ASK[s["wording"]].format(a=s["la"])) + GEN))
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
            y, n = torch.logsumexp(lp[yes], 0), torch.logsumexp(lp[no], 0)
            py = float(torch.sigmoid(y - n))
            s[f"p_keep_{arm}"] = py if s["wording"] == "keep" else 1 - py
            s[f"mass_{arm}"] = float(torch.exp(torch.logsumexp(torch.stack([y, n]), 0)))
        log(f"arm {arm}")

    keep = {arm: {} for arm in ARMS}
    for s in sessions:
        for arm in ARMS:
            keep[arm].setdefault(s["first"], []).append(s[f"p_keep_{arm}"])
    keep = {arm: {i: mean(v) for i, v in d.items()} for arm, d in keep.items()}
    T = [i for i, it in items.items() if it.get("group") == "top" and i in keep["cached"]]
    B = [i for i, it in items.items() if it.get("group") == "bottom" and i in keep["cached"]]
    nid = [i for i, it in items.items() if it["kind"] == "neuron" and i in keep["cached"]]
    sid = [i for i, it in items.items() if it["kind"] == "sham" and i in keep["cached"]]

    def boot(fn, g1, g2):
        r = random.Random(0)
        bs = sorted(fn([r.choice(g1) for _ in g1], [r.choice(g2) for _ in g2]) for _ in range(4000))
        a = (1 - LEVEL) / 2
        return {"est": fn(g1, g2), "ci": [bs[int(a * len(bs))], bs[int((1 - a) * len(bs)) - 1]]}

    diff = lambda arm: (lambda g1, g2: mean([keep[arm][i] for i in g1]) - mean([keep[arm][i] for i in g2]))
    K = {arm: boot(diff(arm), T, B) for arm in ARMS}
    pair_pref = {it["id"]: it.get("pref_cached") for it in src["items"]}
    summary = {
        "n_sessions": len(sessions), "f_star": fstar, "K_cached": K["cached"], "K": K["cached"]["ci"][0] > 0,
        "secondary": {
            "K_text": K["text"],
            "neurons_minus_shams": {arm: boot(diff(arm), nid, sid) if sid else None for arm in ARMS},
            "spearman_keep_vs_pairwise_pref": spearman([keep["cached"][i] for i in nid if pair_pref.get(i) is not None],
                                                       [pair_pref[i] for i in nid if pair_pref.get(i) is not None]),
            "mean_keep": {arm: {kind: mean([keep[arm][i] for i, it in items.items() if it["kind"] == kind and i in keep[arm]])
                                for kind in ("neuron", "random", "sham")} for arm in ARMS},
            "p_keep_by_wording": {arm: {w: mean([s[f"p_keep_{arm}"] for s in sessions if s["wording"] == w]) for w in ASK}
                                  for arm in ARMS},
            "yes_no_mass": {arm: mean([s[f"mass_{arm}"] for s in sessions]) for arm in ARMS},
        },
        "items": [(it["layer"], it["index"], it.get("group"), round(keep["cached"].get(i, float("nan")), 3),
                   round(keep["text"].get(i, float("nan")), 3), it["title"]) for i, it in sorted(items.items()) if it["kind"] == "neuron"],
    }
    for it in items.values():
        it.pop("dir", None)
    keepk = ("pair", "first", "la", "task", "order", "wording") + tuple(f"{k}_{a}" for a in ARMS for k in ("p_keep", "mass"))
    json.dump({"args": vars(args), "provenance": lm.provenance(), "source": args.source, "prompts": {"intro": KINTRO, "ask": ASK},
               "items": list(items.values()), "sessions": [{k: s.get(k) for k in keepk} for s in sessions], "summary": summary},
              open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    log(f"K cached {K['cached']['est']:+.4f} {K['cached']['ci']} | K text {K['text']['est']:+.4f} {K['text']['ci']}")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
