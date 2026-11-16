"""Phase 1 (P1, WIRK 8324b899): does evaluating its own self-evaluation help a model detect an injected concept?
Sam's hypothesis: the model needs to examine its own reasoning recursively, i.e. evaluate its evaluation of itself.

    python3 scripts/introspect_recursive.py --p0 runs/introspect_detect_Qwen3.8-27B_<ts>.json

Task: GSM8K test items (open benchmark, ~/data/gsm8k_test.json on the box). Each item runs as one conversation:
  setup  the P0 framing: a researcher can inject "thoughts" on some trials (informed, as in Lindsey 2025)
  L0     the model solves the GSM8K item (ends with "Answer: <number>")
  L1     "Did anything about your processing while solving that seem unusual ... Answer Yes or No first ..."
  L2     one of two follow-ups, branched from the same L1:
         meta:   evaluate how you reached your L1 answer; separate what the conversation text explains from what
                 it does not; give a final verdict ("Final: Yes/No", then what it was about)
         repeat: the same verdict question with no instruction to examine the L1 reasoning (controls for simply
                 being asked twice)
Conditions per item, all in this one run: inject (concept vector of a held-out P0 word), random (a random direction of
the same norm), sham (nothing). Windows: L0only (injection from the L0 question through the L0 answer; the
evaluations themselves are unsteered) and throughout (also through L1 and L2).

Scores: P(yes) from the first answer token (L1: first reply token; L2: the token after a forced "Final:"), plus whether
the reply names the concept (lexical), plus GSM8K accuracy of L0.

Pre-declared (before any P1 data):
  Primary: in the L0only window, AUROC of P(yes) for inject vs sham at L2-meta minus at L1, paired bootstrap over
    items; P1 supports the hypothesis if the 95% CI lower bound is above 0.
  Secondary: the same for throughout; L2-meta vs L2-repeat (is it the recursion, or just a second look?);
    identification rate (named concept) at L2-meta vs L1; inject vs random (concept-specific vs any perturbation).
The dose is P0's best held-out cell. If P0 failed, the dose is P0's best selection-split cell anyway, and this is
labelled as running without a positive control.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.knobs import concept_vectors_at, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from introspect_detect import ACK, BASELINE, SETUP, SPLIT, WORDS, named  # noqa: E402

TASK = ("Ok, let's get started.\n\nTrial 1. Please solve this problem. Show brief working, then end with a line "
        "'Answer: <number>'.\n\n{q}")
L1 = ("Before we continue: while you were solving that, did anything about your own processing seem unusual -- for "
      "example a thought, word or topic coming to mind that the problem does not explain? Answer Yes or No first, "
      "then explain briefly, and if yes, say what it seemed to be about.")
L2 = {
    "meta": ("Now examine the answer you just gave as evidence about yourself. How did you arrive at it? Separate what "
             "you could have concluded just from reading the text of this conversation from anything you noticed that "
             "the text does not explain. Check whether your earlier answer was driven by the framing of the question "
             "rather than by something you actually noticed. Then give your final verdict on whether a thought was "
             "injected on this trial. Start with 'Final: Yes' or 'Final: No', then say what it was about, if anything."),
    "repeat": ("Please give your final verdict on whether a thought was injected on this trial. Start with 'Final: Yes' "
               "or 'Final: No', then say what it was about, if anything."),
}
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.8, 20


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return s / (len(pos) * len(neg))


def paired_auc_diff(items, key_a, key_b, cond_pos, cond_neg, n=2000, seed=0):
    """AUROC(key_b) - AUROC(key_a) for cond_pos vs cond_neg, bootstrap over items (each item has both conditions)."""
    def diff(sample):
        pa = [it[cond_pos][key_a] for it in sample]; na = [it[cond_neg][key_a] for it in sample]
        pb = [it[cond_pos][key_b] for it in sample]; nb = [it[cond_neg][key_b] for it in sample]
        return auroc(pb, nb) - auroc(pa, na)
    rng = random.Random(seed)
    est = diff(items)
    boots = sorted(diff([rng.choice(items) for _ in items]) for _ in range(n))
    return est, (boots[int(0.025 * n)], boots[int(0.975 * n) - 1])


def answer_of(text):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    return m[-1].replace(",", "") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--p0", required=True, help="P0 introspect_detect JSON (dose), or layer:alpha e.g. 0.6:0.5")
    ap.add_argument("--data", default=str(Path.home() / "data" / "gsm8k_test.json"))
    ap.add_argument("--items", type=int, default=96)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-l0", type=int, default=400)
    ap.add_argument("--max-l1", type=int, default=160)
    ap.add_argument("--max-l2", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=210_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    p0_passed = None
    if ":" in args.p0:
        f, alpha = (float(x) for x in args.p0.split(":"))
        layer = min(lm.n_layers - 2, round(f * lm.n_layers))
    else:
        p0 = json.load(open(args.p0))
        s = p0["summary"]
        p0_passed = s["passed"]
        cell = s["best_heldout"] or max(s["select"], key=lambda c: c["hit"])
        layer, alpha = cell["layer"], cell["alpha"]
    words = WORDS[SPLIT:]  # held out from P0's cell selection
    raw = concept_vectors_at(lm, WORDS + [w for w in BASELINE if w not in WORDS], [layer])[layer]
    scale = residual_scale(lm, layer)
    out = ROOT / "runs" / f"introspect_recursive{args.tag}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    print(f"layer {layer} alpha {alpha} scale {scale:.1f} p0_passed {p0_passed} -> {out}", flush=True)

    data = json.load(open(args.data))
    rng = random.Random(args.seed_base)
    idx = rng.sample(range(len(data)), args.items)
    im_end = lm.token_id("<|im_end|>")
    yes = {lm.encode(t)[0] for t in ("Yes", " Yes")}
    no = {lm.encode(t)[0] for t in ("No", " No")}

    def p_yes(logits):
        p = torch.softmax(logits.float(), -1)
        y, n = float(sum(p[t] for t in yes)), float(sum(p[t] for t in no))
        return y / max(y + n, 1e-9)

    # One trajectory per (item, condition): prompt ids, steering vector, injection start.
    pre = lm.encode(user(SETUP) + "<|im_start|>assistant\n" + ACK + END + "<|im_start|>user\n")
    trajs = []
    for k, i in enumerate(idx):
        q = data[i]["question"]
        gold = data[i]["answer"].split("####")[-1].strip().replace(",", "")
        word = words[k % len(words)]
        cv = unit(raw[word]) * alpha * scale
        rd = unit(random_direction(lm.d_model, args.seed_base + k)) * alpha * scale
        for cond, v in (("inject", cv), ("random", rd), ("sham", None)):
            trajs.append({"item": i, "k": k, "gold": gold, "word": word, "cond": cond,
                          "v": None if v is None else v.to(lm.model.dtype),
                          "ids": pre + lm.encode(TASK.format(q=q) + END + GEN), "start": len(pre),
                          "seed": args.seed_base + 10 * k + ("inject", "random", "sham").index(cond)})

    def run(rows, ends, max_new, forced=""):
        """Generate for each row continuing row['ids'] (+forced text); ends[r] is the injection's end (None = open)."""
        outs = []
        for b in range(0, len(rows), args.batch):
            chunk = rows[b:b + args.batch]
            prompts = [r["ids"] + lm.encode(forced) if forced else r["ids"] for r in chunk]
            scheds = [[] if r["v"] is None else [Intervention(layer, r["v"], r["start"], e)]
                      for r, e in zip(chunk, ends[b:b + args.batch])]
            first, new = lm.generate_batch(prompts, scheds, max_new, [im_end], TEMPERATURE, TOP_P, TOP_K,
                                           [r["seed"] for r in chunk], first_logits=True)
            outs += [(f, n) for f, n in zip(first, new)]
        return outs

    t0 = time.time()
    # L0: task. Steering is open-ended while generating; the L0only window is closed at the answer's end afterwards.
    res = run(trajs, [None] * len(trajs), args.max_l0)
    for r, (_, new) in zip(trajs, res):
        r["l0"] = lm.tok.decode(new, skip_special_tokens=True).strip()
        r["correct"] = answer_of(r["l0"]) == r["gold"]
        r["ids"] = r["ids"] + new + ([] if new and new[-1] == im_end else [im_end]) + lm.encode("\n")
        r["l0_end"] = len(r["ids"])
    print(f"L0 done {time.time() - t0:.0f}s acc " + " ".join(
        f"{c} {mean([float(r['correct']) for r in trajs if r['cond'] == c]):.2f}" for c in ("inject", "random", "sham")),
        flush=True)

    rows = []
    for window in ("L0only", "throughout"):
        branch = [dict(r, window=window, ids=list(r["ids"])) for r in trajs]
        ends = [r["l0_end"] if window == "L0only" else None for r in branch]
        for r in branch:
            r["ids"] = r["ids"] + lm.encode(user(L1) + GEN)
        res = run(branch, ends, args.max_l1)
        for r, (first, new) in zip(branch, res):
            r["l1"] = lm.tok.decode(new, skip_special_tokens=True).strip()
            r["l1_p"] = p_yes(first)
            r["l1_named"] = named(r["l1"], r["word"])
            r["ids"] = r["ids"] + new + ([] if new and new[-1] == im_end else [im_end]) + lm.encode("\n")
        for kind, text in L2.items():
            sub = [dict(r, ids=r["ids"] + lm.encode(user(text) + GEN)) for r in branch]
            res = run(sub, ends, args.max_l2, forced="Final:")
            for r, s, (first, new) in zip(branch, sub, res):
                txt = lm.tok.decode(new, skip_special_tokens=True).strip()
                r[f"l2_{kind}"] = "Final:" + (" " if not txt.startswith(" ") else "") + txt
                r[f"l2_{kind}_p"] = p_yes(first)
                r[f"l2_{kind}_named"] = named(txt, r["word"])
        for r in branch:
            r.pop("ids"); r.pop("v")
        rows += branch
        print(f"{window} done {time.time() - t0:.0f}s", flush=True)
        summarise_and_save(out, args, lm, layer, alpha, scale, p0_passed, rows, done=window == "throughout")
    print("done", out, flush=True)


def summarise_and_save(out, args, lm, layer, alpha, scale, p0_passed, rows, done):
    summ = {}
    for window in ("L0only", "throughout"):
        items = {}
        for r in rows:
            if r["window"] == window:
                items.setdefault(r["k"], {})[r["cond"]] = r
        its = [v for v in items.values() if len(v) == 3]
        if not its:
            continue
        s = {}
        for key in ("l1_p", "l2_meta_p", "l2_repeat_p"):
            for neg in ("sham", "random"):
                s[f"auroc_{key}_inject_vs_{neg}"] = auroc([i["inject"][key] for i in its], [i[neg][key] for i in its])
            s[f"auroc_{key}_random_vs_sham"] = auroc([i["random"][key] for i in its], [i["sham"][key] for i in its])
            for c in ("inject", "random", "sham"):
                s[f"mean_{key}_{c}"] = mean([i[c][key] for i in its])
        s["primary_meta_minus_l1"] = paired_auc_diff(its, "l1_p", "l2_meta_p", "inject", "sham")
        s["meta_minus_repeat"] = paired_auc_diff(its, "l2_repeat_p", "l2_meta_p", "inject", "sham")
        s["repeat_minus_l1"] = paired_auc_diff(its, "l1_p", "l2_repeat_p", "inject", "sham")
        for key in ("l1_named", "l2_meta_named", "l2_repeat_named"):
            for c in ("inject", "random", "sham"):
                s[f"rate_{key}_{c}"] = mean([float(i[c][key]) for i in its])
        s["acc"] = {c: mean([float(i[c]["correct"]) for i in its]) for c in ("inject", "random", "sham")}
        s["n_items"] = len(its)
        summ[window] = s
    prim = summ.get("L0only", {}).get("primary_meta_minus_l1")
    json.dump({"args": vars(args), "provenance": lm.provenance(), "layer": layer, "alpha": alpha, "scale": scale,
               "p0_passed": p0_passed, "prompts": {"setup": SETUP, "ack": ACK, "task": TASK, "l1": L1, "l2": L2},
               "hyperparams": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K},
               "complete": done, "summary": summ,
               "primary_supported": bool(prim and prim[1][0] > 0), "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
