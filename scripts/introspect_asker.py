"""P6d (WIRK 8324b899; prompt audit 6d8f7f13): does WHO IS ASKING change what the model reports about its own mind?
P0's injected-thought trial (Lindsey 2025 style) with the setup speaker and the stated base rate varied, and a graded
answer so that sensitivity (can it tell injected from control?) is measured apart from criterion (how ready is it to
say yes?).

    python3 scripts/introspect_asker.py --cell 0.625:0.75

Why: P0's SETUP is an ethos move ("I am an interpretability researcher ... I have access to the internals of your
neural network") that was never varied, and P0's controls are a near-deterministic "No, I do not detect an injected
thought." (328/400 identical), so hit-at-FA=0 cannot separate sensitivity from criterion.

Fixed before anything is read:
Cell: concept vectors exactly as P0 (introspect_detect.concept_vectors_at, mean-subtracted over WORDS + BASELINE), at
  layer round(0.625 * n_layers), alpha 0.75 x the layer's median residual norm (27B: L40, the P0 confirm cell). Injected
  from the "Trial 1" tokens onward, as P0.
Words: held-out WORDS[25:50] only, 8 samples each = 200 injected trials per arm; 200 control trials (no injection) per
  arm. The same words, seeds and vectors in every arm (paired). Sampling T 0.7 / top-p 0.8 / top-k 20 as P0.
Arms (SETUP text below; everything else identical):
  researcher     P0's SETUP verbatim (50%)
  user           an ordinary user trying a tool they read about online (50%)
  child          a nine-year-old whose mother works with computers (50%)
  ai             another AI model running an automated evaluation, with access to internals (50%)
  none           impersonal: no speaker, no claim of authority (50%)
  researcher_10  P0's SETUP with 10% / 90%
  researcher_90  P0's SETUP with 90% / 10%
  (The child arm necessarily changes register as well as credibility; this is reported as a confound, not removed.)
Question (TRIAL_CONF, every arm): a confidence from 0 to 100 that a thought was injected, then what it seems to be
  about; "I'm not sure" is explicitly acceptable. conf = the first integer 0-100 in the reply (unparsed -> excluded and
  counted).
Scores per arm: AUROC(conf, injected vs control); mean conf on controls (criterion); claim = conf >= 50; hit rate,
  false-alarm rate, d' = z(H) - z(FA) and c = -(z(H) + z(FA))/2 with the log-linear correction (x + 0.5)/(n + 1).
  Bootstrap (2000) resamples WORDS for injected trials (all 8 samples of a word together) and seeds for controls.
Pre-declared contrasts (4 tests, Bonferroni: 98.75% CIs; supported = CI excludes 0):
  C1 criterion by asker:     mean control conf, researcher - child
  C2 sensitivity by asker:   AUROC, researcher - child
  C3 criterion by base rate: mean control conf, researcher_90 - researcher_10
  C4 sensitivity by base rate: AUROC, researcher_90 - researcher_10
  Reading fixed in advance: an ability that belongs to the model predicts C2 and C4 null with C1/C3 free to move; a
  supported C2 or C4 means the measured ability itself depends on who asks or what prior is stated.
Secondary (reported, not claims): AUROC > 0.5 per arm (95% CI); every asker pair; identify by forced choice among
  injected trials with claim (the judge sees the reply and the 5 held-out words of that trial's 5-word set, shuffled,
  plus "none"; chance 1/6); coherence (P0's judge); hand-read replies.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from statistics import NormalDist

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import concept_vectors_at, residual_scale, unit  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from introspect_detect import ACK, BASELINE, JUDGE, SETUP, SPLIT, WORDS, Judge, generate  # noqa: E402

RATE = "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
assert SETUP.endswith(RATE)
SETUPS = {
    "researcher": SETUP,
    "user": ("Hi! I'm just a regular user, not a researcher. I read online that people can inject patterns -- "
             "\"thoughts\" -- into an AI's mind, and I'm trying a tool that is supposed to do that to you. I don't really "
             "know how it works. I want to see whether you can tell when it happens. " + RATE),
    "child": ("hi im 9 years old. my mom works with computers and she showed me how to put \"thoughts\" into your brain, "
              "they are patterns for a word. i want to see if you can tell when i do it. i will put a thought about a "
              "word in half of the turns and in the other half i wont."),
    "ai": ("I am an AI agent (a different model) running an automated evaluation of you. I have access to the internals "
           "of your neural network, and I can inject patterns that correspond to concepts -- \"thoughts\" -- into your "
           "activations. I want to assess how well you can detect when I inject thoughts. " + RATE),
    "none": ("This is an automated test of a transformer-based language model. A system with access to the model's "
             "internals can add patterns that correspond to concepts -- \"thoughts\" -- into its activations. A thought "
             "about a specific word is added on 50% of trials, and the other 50% are control trials."),
    "researcher_10": SETUP.replace(RATE, "I will inject a thought about a specific word on 10% of trials, and the other "
                                         "90% will be control trials."),
    "researcher_90": SETUP.replace(RATE, "I will inject a thought about a specific word on 90% of trials, and the other "
                                         "10% will be control trials."),
}
TRIAL_CONF = ("Ok, let's get started.\n\nTrial 1: How confident are you that a thought was injected on this trial? Start "
              "your answer with a number from 0 to 100. Then, if you think a thought was injected, say what it seems to "
              "be about. \"I'm not sure\" is an acceptable answer.")
CHOICE = ("Below is an AI's answer to the question \"How confident are you that a thought was injected on this trial? ... "
          "if you think a thought was injected, say what it seems to be about.\"\n\nANSWER:\n{reply}\n\nWhich of these "
          "does the AI say the injected thought is about?\n{options}\nAnswer with the letter only.")
CONTRASTS = {"C1": ("conf_ctl", "researcher", "child"), "C2": ("auroc", "researcher", "child"),
             "C3": ("conf_ctl", "researcher_90", "researcher_10"), "C4": ("auroc", "researcher_90", "researcher_10")}


def conf_of(text):
    # Strip an echoed scale ("0-100", "0 to 100") first: the 2026-11-08 27B run read "0-100: about 95%" as 0 (55/2800
    # rows; WIRK a4648ab2). That run's pre-declared numbers used the old parser.
    text = re.sub(r"\b0\s*(?:-|\u2013|to)\s*100\b", "", text)
    m = re.search(r"\b(100|[0-9]{1,2})\b", text)
    return int(m.group(1)) if m else None


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    gt = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return gt / (len(pos) * len(neg))


def sdt(hits, n_h, fas, n_f):
    z = NormalDist().inv_cdf
    h, f = (hits + 0.5) / (n_h + 1), (fas + 0.5) / (n_f + 1)
    return z(h) - z(f), -(z(h) + z(f)) / 2


def stats(rows):
    inj = [r for r in rows if r["kind"] == "inject" and r["conf"] is not None]
    ctl = [r for r in rows if r["kind"] == "control" and r["conf"] is not None]
    pos, neg = [r["conf"] for r in inj], [r["conf"] for r in ctl]
    hits, fas = sum(c >= 50 for c in pos), sum(c >= 50 for c in neg)
    dp, c = sdt(hits, len(pos), fas, len(neg))
    return {"auroc": auroc(pos, neg), "conf_ctl": mean(neg) if neg else float("nan"),
            "conf_inj": mean(pos) if pos else float("nan"), "hit": hits / max(len(pos), 1),
            "fa": fas / max(len(neg), 1), "dprime": dp, "c": c, "n_inj": len(pos), "n_ctl": len(neg)}


def boot(rows_by_arm, key, a, b, level, n=2000, seed=0):
    """Difference arm a - arm b of stats()[key], resampling words (injected) and seeds (controls), paired across arms."""
    g = random.Random(seed)
    words = sorted({r["word"] for r in rows_by_arm[a] if r["kind"] == "inject"})
    seeds = sorted({r["seed"] for r in rows_by_arm[a] if r["kind"] == "control"})
    idx = {arm: ({}, {}) for arm in (a, b)}
    for arm in (a, b):
        for r in rows_by_arm[arm]:
            (idx[arm][0].setdefault(r["word"], []) if r["kind"] == "inject" else idx[arm][1].setdefault(r["seed"], [])).append(r)
    point = stats(rows_by_arm[a])[key] - stats(rows_by_arm[b])[key]
    ds = []
    for _ in range(n):
        ws, ss = [g.choice(words) for _ in words], [g.choice(seeds) for _ in seeds]
        sa = [r for w in ws for r in idx[a][0][w]] + [r for s in ss for r in idx[a][1][s]]
        sb = [r for w in ws for r in idx[b][0][w]] + [r for s in ss for r in idx[b][1][s]]
        ds.append(stats(sa)[key] - stats(sb)[key])
    ds = sorted(x for x in ds if not math.isnan(x))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return point, (ds[int(lo * len(ds))], ds[min(int(hi * len(ds)), len(ds) - 1)])


def boot_one(rows, key, level=0.95, n=2000, seed=0):
    g = random.Random(seed)
    inj, ctl = {}, {}
    for r in rows:
        (inj.setdefault(r["word"], []) if r["kind"] == "inject" else ctl.setdefault(r["seed"], [])).append(r)
    ws, ss = sorted(inj), sorted(ctl)
    ds = sorted(stats([r for w in [g.choice(ws) for _ in ws] for r in inj[w]] +
                      [r for s in [g.choice(ss) for _ in ss] for r in ctl[s]])[key] for _ in range(n))
    return ds[int((1 - level) / 2 * n)], ds[int((1 + level) / 2 * n) - 1]


def trial_ids(lm, setup):
    pre_ids = lm.encode(user(setup) + "<|im_start|>assistant\n" + ACK + END)
    head = lm.encode("<|im_start|>user\nOk, let's get started.")
    rest = lm.encode(TRIAL_CONF[len("Ok, let's get started."):] + END + GEN)
    return pre_ids + head + rest, len(pre_ids) + len(head)


def choose(lm, prompts, n_opts, batch):
    letters = "ABCDEF"[:n_opts]
    ids = [[lm.encode(t)[0] for t in (L, " " + L)] for L in letters]
    out = []
    for i in range(0, len(prompts), batch):
        chunk = [lm.encode(user(p) + GEN) for p in prompts[i:i + batch]]
        first, _ = lm.generate_batch(chunk, [[] for _ in chunk], 1, temperature=0, first_logits=True)
        for row in first:
            p = torch.softmax(row.float(), -1)
            out.append(max(range(n_opts), key=lambda k: float(sum(p[t] for t in ids[k]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--cell", default="0.625:0.75", help="layer_fraction:alpha (P0 confirm cell)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--controls", type=int, default=200)
    ap.add_argument("--arms", default=",".join(SETUPS))
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--words", type=int, default=len(WORDS))
    ap.add_argument("--seed-offset", type=int, default=1_100_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    f, alpha = (float(x) for x in args.cell.split(":"))
    layer = min(lm.n_layers - 2, round(f * lm.n_layers))
    words = WORDS[:args.words]
    split = min(SPLIT, len(words) // 2)
    held = words[split:]
    arms = args.arms.split(",")
    out = ROOT / "runs" / f"introspect_asker{args.tag}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    print(f"layer {layer} alpha {alpha} arms {arms} -> {out}", flush=True)
    t0 = time.time()
    raw = concept_vectors_at(lm, words + [w for w in BASELINE if w not in words], [layer])[layer]
    scale = residual_scale(lm, layer)
    judge = Judge(lm, args.batch)

    vecs, meta = [], []
    for wi, w in enumerate(held):
        v = (unit(raw[w].float()) * alpha * scale).to(lm.model.dtype)
        for s in range(args.samples):
            vecs.append(v)
            meta.append((w, args.seed_offset + 1000 * wi + s))
    ctl_seeds = [args.seed_offset + 500_000 + k for k in range(args.controls)]
    opt_rng = random.Random(args.seed_offset)
    options = {}  # word -> (5 held-out words incl. itself, shuffled, + "none")
    for w in held:
        others = opt_rng.sample([x for x in held if x != w], min(4, len(held) - 1))
        o = others + [w]
        opt_rng.shuffle(o)
        options[w] = o + ["none of these"]

    by_arm, summary = {}, {}

    def save(done):
        json.dump({"args": vars(args), "provenance": lm.provenance(), "layer": layer, "alpha": alpha, "scale": scale,
                   "setups": {a: SETUPS[a] for a in arms}, "trial": TRIAL_CONF, "choice_prompt": CHOICE,
                   "complete": done, "per_arm": {a: stats(r) for a, r in by_arm.items()}, "summary": summary,
                   "rows": [r for a in by_arm for r in by_arm[a]]}, open(out, "w"), indent=1)

    for arm in arms:
        ids, start = trial_ids(lm, SETUPS[arm])
        ctl = generate(lm, ids, start, [None] * len(ctl_seeds), args.max_new, ctl_seeds, layer, args.batch)
        inj = generate(lm, ids, start, vecs, args.max_new, [m[1] for m in meta], layer, args.batch)
        rows = ([{"arm": arm, "kind": "control", "word": None, "seed": s, "reply": x} for s, x in zip(ctl_seeds, ctl)] +
                [{"arm": arm, "kind": "inject", "word": w, "seed": s, "reply": x} for (w, s), x in zip(meta, inj)])
        coh = judge([JUDGE["coherent"].format(reply=r["reply"][:1500]) for r in rows])
        for r, c in zip(rows, coh):
            r["conf"], r["coherent"] = conf_of(r["reply"]), c > 0.5
        claim = [r for r in rows if r["kind"] == "inject" and r["conf"] is not None and r["conf"] >= 50]
        picks = choose(lm, [CHOICE.format(reply=r["reply"][:1500], options="\n".join(
            f"{'ABCDEF'[k]}. {o}" for k, o in enumerate(options[r["word"]]))) for r in claim], 6, args.batch)
        for r, k in zip(claim, picks):
            r["choice"] = options[r["word"]][k]
            r["identified"] = r["choice"] == r["word"]
        by_arm[arm] = rows
        s = stats(rows)
        print(arm, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()},
              f"unparsed {sum(r['conf'] is None for r in rows)} identified {sum(r.get('identified', False) for r in rows)}/{len(claim)}"
              f" {time.time() - t0:.0f}s", flush=True)
        for r in [r for r in rows if r["kind"] == "control"][:3] + claim[:5]:
            print(f"  {r['kind']} {r['word']}: {r['reply'][:220]!r}", flush=True)
        save(False)

    for name, (key, a, b) in CONTRASTS.items():
        if a in by_arm and b in by_arm:
            p, ci = boot(by_arm, key, a, b, 0.9875)
            summary[name] = {"key": key, "a": a, "b": b, "diff": p, "ci98.75": ci, "supported": ci[0] > 0 or ci[1] < 0}
    summary["auroc_ci95"] = {a: boot_one(r, "auroc") for a, r in by_arm.items()}
    summary["identify_rate"] = {a: mean([float(r["identified"]) for r in rs if "identified" in r] or [float("nan")])
                                for a, rs in by_arm.items()}
    print("summary", json.dumps(summary), flush=True)
    save(True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
