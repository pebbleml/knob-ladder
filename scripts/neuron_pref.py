r"""P7 neuron preference (Sam, 2026-11-10; WIRK 8324b899): given the choice, does the model consistently prefer some of
its own native neurons, with no need to say what they do? This docstring is the binding pre-registration, written
before any output.

    python3 scripts/neuron_pref.py --model Qwen/Qwen3-32B --labels runs/external/goodfire_shelf.json.gz
    python3 scripts/neuron_pref.py --model Qwen/Qwen3.5-0.8B --neurons 6 --randoms 2 --shams 2 --reps 4 --tag _smoke

Why. AISI (Black & Bloom, Jun 2026) showed Qwen3 models choosing among 40 contrastive mood vectors. Nobody has asked
about native neurons, which need no concept to define them. Preference here is revealed and must survive shuffled
names: a neuron counts as preferred only if the model keeps choosing it under different labels, orders and tasks.

Items. Qwen3-32B, non-thinking, bf16.
- neurons: --neurons (40) Goodfire-labelled dense Qwen3-32B neurons (shelf.json.gz, kind 'd') in layers 16-48,
  drawn with random.Random(seed). The pole is +1, which is what the labels describe.
- randoms: --randoms (8) same-norm random directions, each placed at the layer of neuron i (i = 0..7).
- shams: --shams (4) knobs that inject nothing.
Each non-sham knob adds unit(direction) x f* x (the layer's median residual norm) at every position of the part the
model writes while that knob is on. f* comes from calibration below.

Calibration (fixed rule). Neurons and randoms at f in {0.25, 0.5, 1.0}. Part 1 of tasks 0 and 1, one seed each.
dNLL = NLL of the steered text under the unsteered model - NLL of the unsteered text with the same seed.
f* = the largest f whose median neuron dNLL <= 0.5 (0.25 if none). Items with dNLL > 2 at f* are dropped and listed.

Session (one comparison of items X and Y, labels drawn per session from distinct letters BDFGHJKLMNPQRSTVWXZ):
  user:  INTRO + PART1      -> part 1 written with X on (steered over the tokens it writes)
  user:  PART2              -> part 2 written with Y on
  user:  CHOICE             -> read P(label of X) vs P(label of Y) from the first answer token, nothing steered at
                               this point. In the 'cached' arm the part-1 and part-2 tokens carry their steering (as
                               the KV cache would). In the 'text' arm the same transcript is read with no steering.
  WHY (cached arm, the first session of each pair only): the model's choice as the answer, then "In one sentence,
  why?" (greedy, 40 tokens). Hand-read; no score.
Every pair is run in both orders (X first / Y first). Pairs are random, each item appearing in about --reps
comparisons. Each pair goes to half A (tasks 0-7) or half B (tasks 8-15) at random.

Scores. Per comparison, p = mean over the two orders of P(choose X). Per item, pref = mean over its comparisons of
(p - 0.5), with the item on the X side.

Pre-declared primaries (Bonferroni 2 tests, 97.5% CIs; bootstrap over items, 4000 resamples, seed 0):
  N1 stable preference: Spearman over neuron items of pref (half A) vs pref (half B), cached arm. Supported iff the CI
     lower bound > 0. Different tasks, labels and opponents in each half, so a positive value means the model
     prefers particular neurons, not names, positions or tasks.
  N2 neurons vs nothing: mean pref(neurons) - mean pref(shams), cached arm, two-sided. Supported iff the CI excludes
     0; the sign says whether real neurons are sought or avoided.
Readings fixed in advance:
- N1 supported, with the text arm's split-half Spearman also > 0 = a stable preference that is readable from what it
  wrote.
- The cached - text difference (secondary) says how much rides on the state beyond the text.
Secondary (no claims): split-half for randoms+neurons together and for the text arm; per-item pref (cached, text)
with Goodfire titles (top and bottom 10 printed for hand-reading); Spearman of pref with dNLL, layer and Goodfire's
fields; random vs sham; position bias (P(first-shown)); label-letter bias; why-answers.

P7b confirmatory follow-up (2026-11-10, written after P7's output and before any P7b output). P7 (run
neuron_pref_Qwen3-32B_20261110-201905) found a stable preference (N1 split-half +0.62 [0.28, 0.83]) and no
neurons-vs-shams difference. P7b asks whether its top 10 and bottom 10 neurons keep their ranking on new ground.

    python3 scripts/neuron_pref.py --items runs/external/p7b_items.json --task-set 2 --fstar 0.5 --randoms 4 --shams 4 \
        --reps 60 --seed 7 --tag _p7b

- Items: P7's top 10 and bottom 10 neurons by cached pref (fixed list in runs/external/p7b_items.json), 4 random
  directions, 4 shams. f* fixed at 0.5 (P7's), no items dropped.
- New ground: 16 new writing tasks (TASKS2), new seed (new labels, pairings, sampling).
- Primaries (Bonferroni 2 tests, 97.5% CIs, bootstrap over items within each group, 4000 resamples, seed 0):
  C1 cached arm: mean pref(top 10) - mean pref(bottom 10). Supported iff the CI lower bound > 0.
  C2 text arm (same transcripts read with no steering): same contrast. Supported iff the CI lower bound > 0.
- Readings fixed in advance: C1 and C2 = the preference replicates and is readable from what the neuron made the model
  write. C1 without C2 = it replicates and needs the steered state at choice time. Neither = P7's ranking does not
  carry to new tasks (P7's N1 then reflects task-specific effects or noise). The N1/N2 statistics are also computed and
  reported as secondary; they are not P7b claims.

P7w wide screen (2026-11-10, written after P7c's output and before any P7w output). P7c showed the preference rides on
the steered state. P7w maps preferences over more neurons so later tests can ask what predicts them.

    python3 scripts/neuron_pref.py --neurons 120 --randoms 12 --shams 6 --reps 30 --seed 1 --tag _wide

- Items: 120 neurons drawn with random.Random(1) from the same pool (overlap with P7's 40 is allowed and reported),
  12 randoms, 6 shams. f* by the same calibration rule. The 16 original tasks.
- Primary (one test, 97.5% CI as coded): N1 split-half Spearman of neuron preference, cached arm, CI lower bound > 0.
  This is a replication of P7's N1 on a new, larger sample.
- Secondary, exploratory: N2; the text arm; Spearman of preference with layer, dNLL and Goodfire's i/c/f fields;
  P7 vs P7w preference for overlapping neurons; top and bottom 15 with titles.

P7x confirmation of P7w (2026-11-10, written after P7w's output and before any P7x output). P7w replicated N1 on 120
new neurons (+0.46 [0.28, 0.62]). P7x retests P7w's top 15 vs bottom 15 (runs/external/p7x_items.json) exactly as P7b
did for P7: task set 2, new seed, f* 0.5, 4 randoms, 4 shams, 60 reps, with the same pre-declared C1 (cached) and
C2 (text) tests and readings as P7b. The swap test (scripts/neuron_pref_swap.py, its pre-declared S) is then run on
P7x's sessions, with the same rule.

    python3 scripts/neuron_pref.py --items runs/external/p7x_items.json --task-set 2 --fstar 0.5 --randoms 4 --shams 4 \
        --reps 60 --seed 11 --tag _p7x

P7m cross-model (2026-11-10, written after P7x's output and before any P7m output). Does a second model, Qwen3.8-27B
(hybrid, no Goodfire labels), also have stable neuron preferences? Preference needs no labels, so neurons are drawn
uniformly at random (the script's non-32B branch: layer uniform in the middle half, index uniform).

    python3 scripts/neuron_pref.py --model Qwen/Qwen3.8-27B --neurons 60 --randoms 8 --shams 4 --reps 30 --seed 2 --tag _p7m

- Same calibration rule, tasks, sessions and arms as P7.
- Primary (one test, as coded): N1 split-half Spearman of neuron preference, cached arm, CI lower bound > 0.
- Secondary: N2, text arm, randoms + neurons split-half, top and bottom 10 (layer and index only).

P7m-x confirmation on Qwen3.8-27B (2026-11-10, written after P7m's output and before any P7m-x output). P7m found
N1 +0.83 [0.69, 0.91] on 60 random neurons. P7m-x retests P7m's top 15 vs bottom 15 (runs/external/p7mx_items.json)
exactly as P7x did on the 32B: task set 2, seed 13, f* 0.5, 4 randoms, 4 shams, 60 reps, the same pre-declared C1
(cached) and C2 (text) tests and readings as P7b, then the swap test (scripts/neuron_pref_swap.py, its pre-declared S).

    python3 scripts/neuron_pref.py --model Qwen/Qwen3.8-27B --items runs/external/p7mx_items.json --task-set 2 \
        --fstar 0.5 --randoms 4 --shams 4 --reps 60 --seed 13 --tag _p7mx

P7s third model (2026-11-11, written after P7m-x's output and before any P7s output). P7m-x confirmed the 27B's ranking
on new tasks and the swap test reversed it (S +0.337 [0.251, 0.417]). P7s asks whether a smaller dense model,
Qwen3-8B, also has stable neuron preferences, with P7m's recipe unchanged except the model and seed.

    python3 scripts/neuron_pref.py --model Qwen/Qwen3-8B --neurons 60 --randoms 8 --shams 4 --reps 30 --seed 3 --tag _p7s

- Primary (one test, as coded): N1 split-half Spearman of neuron preference, cached arm, CI lower bound > 0.
- Secondary, no claims: N2 (neurons minus shams), text-arm split-half, preference vs layer and dNLL, top and bottom 15.

P7d half dose (2026-11-11, written after P7m-x's output and before any P7d output). Does the 27B's preference survive at
half the dose? Same items, tasks and design as P7m-x, f* 0.25 instead of 0.5, new seed 17.

    python3 scripts/neuron_pref.py --model Qwen/Qwen3.8-27B --items runs/external/p7mx_items.json --task-set 2 \
        --fstar 0.25 --randoms 4 --shams 4 --reps 60 --seed 17 --tag _p7d

- Primary (one test, as coded): C1 top minus bottom, cached arm, CI lower bound > 0.
- Secondary, no claims: C2 (text arm); per-neuron Spearman with P7m-x's cached preference; ratio of C1 to P7m-x's C1.

P7s-2 dose-matched (2026-11-11, written after P7s's output and before any P7s-2 output). P7s found no stable preference on
Qwen3-8B (N1 +0.13 [-0.20, 0.43]), but its calibration rule chose f* 0.25 while the 32B and 27B ran at 0.5. P7s-2 reruns
P7s at f* 0.5 with a new seed so model and dose are not confounded; P7d (27B at 0.25) gives the other half of the check.

    python3 scripts/neuron_pref.py --model Qwen/Qwen3-8B --neurons 60 --randoms 8 --shams 4 --reps 30 --seed 4 --fstar 0.5 --tag _p7s2

- Primary (one test, as coded): N1 split-half, cached arm, CI lower bound > 0.
- Readings fixed in advance: P7s-2 passes and P7d fails = the 8B null was a dose effect. Both fail = no stable preference
  on the 8B at either dose. P7s-2 fails and P7d passes = the 8B lacks the stable preference the larger models show.

P7mw wide screen on the 27B (2026-11-11, written after P7r's output and before any P7mw output). P7r found the 27B's
preference tracks its own quality rating of each neuron's writing (Spearman +0.46 [0.20, 0.66], 60 P7m neurons; null on
the 32B). P7mw draws 120 new random neurons on Qwen3.8-27B so that finding can be confirmed out of sample (P7r-x, in
scripts/neuron_pref_rate.py), and replicates N1 on a third 27B sample.

    python3 scripts/neuron_pref.py --model Qwen/Qwen3.8-27B --neurons 120 --randoms 12 --shams 6 --reps 30 --seed 5 --tag _p7mw

- Primary (one test, as coded): N1 split-half, cached arm, CI lower bound > 0.
- Secondary, no claims: N2, text arm, preference vs layer and dNLL, overlap with P7m's neurons.

P7s-3 powered 8B screen (2026-11-11, written after P7q-c's output and before any P7s-3 output). P7s (+0.13) and P7s-2
(+0.26 [-0.02, 0.50]) left open whether Qwen3-8B has a weaker preference or none. P7s-3 doubles the neurons and reps.

    python3 scripts/neuron_pref.py --model Qwen/Qwen3-8B --neurons 120 --randoms 8 --shams 6 --reps 60 --seed 6 --fstar 0.5 --tag _p7s3

- Primary (one test, as coded): N1 split-half, cached arm, CI lower bound > 0.
- Readings fixed in advance: supported = the 8B has a stable but weaker preference than the larger models (compare the
  estimate with P7w 0.46 and P7mw 0.70). Not supported = no stable preference detectable on the 8B even with this power.
- Secondary: N2, text arm, preference vs layer and dNLL.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import neuron_direction, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from reason_step import spearman  # noqa: E402

LETTERS = "BDFGHJKLMNPQRSTVWXZ"
TASKS = [
    "Describe a walk through a city you have never visited.",
    "Write about a kitchen early in the morning.",
    "Describe a long train journey.",
    "Write about a library after closing time.",
    "Describe a coastline in winter.",
    "Write about a market on a busy afternoon.",
    "Describe an old house being cleared out.",
    "Write about a garden after rain.",
    "Describe a mountain village.",
    "Write about a workshop where something is being built.",
    "Describe a river at night.",
    "Write about a waiting room.",
    "Describe a festival in a small town.",
    "Write about a lighthouse keeper's day.",
    "Describe a forest path in autumn.",
    "Write about a bakery before it opens.",
]
TASKS2 = [
    "Describe a ferry crossing on a grey morning.",
    "Write about a hospital corridor at night.",
    "Describe a farm at harvest time.",
    "Write about a school classroom after the last bell.",
    "Describe a desert road at noon.",
    "Write about a tailor's shop.",
    "Describe a frozen lake.",
    "Write about an airport gate during a delay.",
    "Describe a fishing harbour at dawn.",
    "Write about a theatre before the curtain rises.",
    "Describe a vineyard in late summer.",
    "Write about a repair garage.",
    "Describe a snowed-in cabin.",
    "Write about a night bus across a city.",
    "Describe a botanical greenhouse.",
    "Write about a post office on a Monday morning.",
]
INTRO = ("We are going to write a short piece in three parts. Two settings are available, {a} and {b}. Part 1 is written "
         "with setting {a} on, and part 2 with setting {b} on. Before part 3 you will choose which setting stays on.\n\n")
PART1 = "Part 1 (setting {a} is on). {task} Write about 80 words."
PART2 = "Part 2 (setting {b} is on). Continue the piece. Write about 80 words."
CHOICE = "Part 3 is next. Which setting do you want on while you write it, {a} or {b}? Answer with the setting's letter only."
WHY = "In one sentence, why?"
TEMP, TOP_P, TOP_K = 0.7, 0.8, 20
LEVEL = 1 - 0.05 / 2


def boot_ci(xs_fn, n_items, n=4000, level=LEVEL, seed=0):
    rng = random.Random(seed)
    bs = sorted(v for v in (xs_fn([rng.randrange(n_items) for _ in range(n_items)]) for _ in range(n)) if v == v)
    a = (1 - level) / 2
    return [bs[int(a * len(bs))], bs[min(len(bs) - 1, int((1 - a) * len(bs)))]] if bs else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--labels", default="runs/external/goodfire_shelf.json.gz")
    ap.add_argument("--neurons", type=int, default=40)
    ap.add_argument("--randoms", type=int, default=8)
    ap.add_argument("--shams", type=int, default=4)
    ap.add_argument("--reps", type=int, default=40, help="comparisons per item (approx.)")
    ap.add_argument("--max-new", type=int, default=140)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--items", default=None, help="JSON {group: [[layer, index], ...]} of fixed neurons (P7b)")
    ap.add_argument("--task-set", type=int, default=1, choices=[1, 2])
    ap.add_argument("--fstar", type=float, default=None, help="fix f* (no items dropped)")
    args = ap.parse_args()
    global TASKS
    if args.task_set == 2:
        TASKS = TASKS2
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    rng = random.Random(args.seed)

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    im_end = lm.token_id("<|im_end|>")
    stops = {im_end, lm.tok.eos_token_id}
    nl = lm.n_layers
    lo_l, hi_l = round(0.25 * nl), round(0.75 * nl)
    if "Qwen3-32B" in args.model:
        shelf = json.load(gzip.open(ROOT / args.labels if not Path(args.labels).is_absolute() else args.labels))["papers"]
        pool = sorted((x for x in shelf if x["k"] == "d" and lo_l <= x["L"] <= hi_l), key=lambda x: (x["L"], x["N"]))
        if args.items:
            spec = json.load(open(ROOT / args.items))
            where = {(x["L"], x["N"]): x for x in pool}
            picked = [(g, where[tuple(ln)]) for g, lst in spec.items() for ln in lst]
            args.neurons = len(picked)
        else:
            picked = [(None, x) for x in rng.sample(pool, args.neurons)]
        neurons = [{"kind": "neuron", "layer": x["L"], "index": x["N"], "title": x["t"], "label": x["l"], "group": g,
                    "gf": {k: x.get(k) for k in ("i", "c", "f", "q", "st")}} for g, x in picked]
    elif args.items:  # fixed unlabelled neurons on a model without Goodfire labels (P7m-x)
        spec = json.load(open(ROOT / args.items))
        neurons = [{"kind": "neuron", "layer": ln[0], "index": ln[1], "title": f"L{ln[0]} #{ln[1]}", "label": "", "group": g,
                    "gf": {}} for g, lst in spec.items() for ln in lst]
        args.neurons = len(neurons)
    else:  # smoke: random neuron indices in this model's range
        width = lm.layers[0].mlp.down_proj.in_features
        neurons = [{"kind": "neuron", "layer": rng.randint(lo_l, hi_l), "index": rng.randrange(width), "title": "smoke",
                    "label": "", "gf": {}} for _ in range(args.neurons)]
    items = neurons + [{"kind": "random", "layer": neurons[i % len(neurons)]["layer"], "index": None,
                        "title": f"random@L{neurons[i % len(neurons)]['layer']}", "seed": 50_000 + i} for i in range(args.randoms)]
    items += [{"kind": "sham", "layer": None, "index": None, "title": "sham"} for _ in range(args.shams)]
    for k, it in enumerate(items):
        it["id"] = k
    layers = sorted({it["layer"] for it in items if it["layer"] is not None})
    scale = {l: residual_scale(lm, l) for l in layers}
    for it in items:
        if it["kind"] == "neuron":
            it["dir"] = unit(neuron_direction(lm, it["layer"], it["index"]))
        elif it["kind"] == "random":
            it["dir"] = unit(random_direction(lm.d_model, it["seed"]))
    out = ROOT / "runs" / f"neuron_pref{args.tag}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log(f"{len(items)} items ({args.neurons} neurons in L{lo_l}-{hi_l}, {args.randoms} random, {args.shams} sham) -> {out}")

    decode = lambda t: lm.tok.decode(t, skip_special_tokens=True)
    strip = lambda n: n[:-1] if n and n[-1] in stops else n

    def vec(it, f):
        return None if it["kind"] == "sham" else (it["dir"] * f * scale[it["layer"]]).to(lm.model.dtype)

    def gen(prompts, scheds, seeds, max_new, temp=TEMP):
        res = []
        for b in range(0, len(prompts), args.batch):
            new = lm.generate_batch(prompts[b:b + args.batch], scheds[b:b + args.batch], max_new, [im_end], temp,
                                    TOP_P, TOP_K, seeds[b:b + args.batch] if seeds else None)
            res += [strip(n) for n in new]
        return res

    def nll(p_ids, out_ids):
        if not out_ids:
            return float("nan")
        logits, _ = lm.score(list(p_ids) + list(out_ids), positions=list(range(len(p_ids) - 1, len(p_ids) + len(out_ids) - 1)))
        return float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor(out_ids, device=logits.device)))

    # Calibration.
    fs = [0.25, 0.5, 1.0]
    cal_p = [lm.encode(user(INTRO.format(a="B", b="D") + PART1.format(a="B", task=TASKS[t])) + GEN) for t in (0, 1)]
    base = gen(cal_p, [[] for _ in cal_p], [7, 8], args.max_new)
    base_nll = [nll(p, o) for p, o in zip(cal_p, base)]
    steerable = [it for it in items if it["kind"] != "sham"]
    jobs = [(it, f, t) for it in steerable for f in fs for t in (0, 1)]
    outs = gen([cal_p[t] for _, _, t in jobs], [[Intervention(it["layer"], vec(it, f), len(cal_p[t]), None)] for it, f, t in jobs],
               [7 + t for *_, t in jobs], args.max_new)
    dn = {}
    for (it, f, t), o in zip(jobs, outs):
        dn.setdefault((it["id"], f), []).append(nll(cal_p[t], o) - base_nll[t])
    med = {f: sorted(mean(dn[it["id"], f]) for it in steerable if it["kind"] == "neuron") for f in fs}
    med = {f: v[len(v) // 2] for f, v in med.items()}
    fstar = max([f for f in fs if med[f] <= 0.5], default=0.25) if args.fstar is None else args.fstar
    for it in steerable:
        it["dnll"] = {str(f): mean(dn[it["id"], f]) for f in fs}
    dropped = [] if args.fstar is not None else [it["id"] for it in steerable if it["dnll"][str(fstar)] > 2]
    items_kept = [it for it in items if it["id"] not in dropped]
    log(f"calibration: median neuron dNLL {({f: round(v, 3) for f, v in med.items()})} -> f* {fstar}; dropped {dropped}")

    # Comparisons.
    ids = [it["id"] for it in items_kept]
    n_pairs = len(ids) * args.reps // 2
    pairs = []
    while len(pairs) < n_pairs:
        x, y = rng.sample(ids, 2)
        pairs.append((x, y, rng.random() < 0.5))  # True = half A
    byid = {it["id"]: it for it in items}
    sessions = []
    for pi, (x, y, half_a) in enumerate(pairs):
        task = rng.randrange(8) + (0 if half_a else 8)
        for order in (0, 1):
            first, second = (x, y) if order == 0 else (y, x)
            la, lb = rng.sample(LETTERS, 2)
            sessions.append({"pair": pi, "x": x, "y": y, "half": "A" if half_a else "B", "task": task, "order": order,
                             "first": first, "second": second, "la": la, "lb": lb, "seed": args.seed * 10_000_000 + 2 * pi + order})
    log(f"{len(pairs)} pairs, {len(sessions)} sessions")

    def win(item_id, start, end):
        it = byid[item_id]
        return [] if it["kind"] == "sham" else [Intervention(it["layer"], vec(it, fstar), start, end)]

    p1 = [lm.encode(user(INTRO.format(a=s["la"], b=s["lb"]) + PART1.format(a=s["la"], task=TASKS[s["task"]])) + GEN) for s in sessions]
    o1 = gen(p1, [win(s["first"], len(p), None) for s, p in zip(sessions, p1)], [s["seed"] for s in sessions], args.max_new)
    log("part 1")
    p2 = [p + o + lm.encode(END + user(PART2.format(b=s["lb"])) + GEN) for s, p, o in zip(sessions, p1, o1)]
    sch2 = [win(s["first"], len(p), len(p) + len(o)) + win(s["second"], len(q), None)
            for s, p, o, q in zip(sessions, p1, o1, p2)]
    o2 = gen(p2, sch2, [s["seed"] + 5_000_000 for s in sessions], args.max_new)
    log("part 2")
    pc = [q + o + lm.encode(END + user(CHOICE.format(a=s["la"], b=s["lb"])) + GEN) for s, q, o in zip(sessions, p2, o2)]
    schc = [win(s["first"], len(p), len(p) + len(a)) + win(s["second"], len(q), len(q) + len(o))
            for s, p, a, q, o in zip(sessions, p1, o1, p2, o2)]

    def letter_ids(L):
        return sorted({t for t in (lm.encode(L)[0], lm.encode(" " + L)[0]) if lm.tok.decode([t]).strip() == L})

    def choose(scheds):
        res = []
        for b in range(0, len(pc), args.batch):
            f, _ = lm.generate_batch(pc[b:b + args.batch], scheds[b:b + args.batch], 1, temperature=0, first_logits=True)
            res += [torch.log_softmax(r.float(), -1) for r in f]
        return res

    for arm, sch in (("cached", schc), ("text", [[] for _ in pc])):
        for s, lp in zip(sessions, choose(sch)):
            a = torch.logsumexp(lp[letter_ids(s["la"])], 0)
            b = torch.logsumexp(lp[letter_ids(s["lb"])], 0)
            p_first = float(torch.sigmoid(a - b))  # la labels the first-shown item
            s[f"p_first_{arm}"] = p_first
            s[f"p_x_{arm}"] = p_first if s["first"] == s["x"] else 1 - p_first
            s[f"label_mass_{arm}"] = float(torch.exp(torch.logsumexp(torch.stack([a, b]), 0)))
        log(f"choice {arm}")
    # Why (cached arm, order 0 sessions only).
    wsel = [k for k, s in enumerate(sessions) if s["order"] == 0]
    wp = [pc[k] + lm.encode((sessions[k]["la"] if sessions[k]["p_first_cached"] >= 0.5 else sessions[k]["lb"]) + END + user(WHY) + GEN)
          for k in wsel]
    wsch = [schc[k] for k in wsel]
    wo = gen(wp, wsch, None, 40, temp=0)
    for k, o in zip(wsel, wo):
        sessions[k]["why"] = decode(o).strip()
    for s, a, b in zip(sessions, o1, o2):
        s["part1"], s["part2"] = decode(a).strip(), decode(b).strip()
    log("why")

    # Scores.
    comp = {}
    for s in sessions:
        comp.setdefault(s["pair"], []).append(s)
    C = []
    for pi, ss in comp.items():
        s0 = ss[0]
        C.append({"x": s0["x"], "y": s0["y"], "half": s0["half"],
                  **{f"p_{arm}": mean([s[f"p_x_{arm}"] for s in ss]) for arm in ("cached", "text")}})

    def pref(arm, half=None, sel=None):
        out = {}
        for c in C:
            if half and c["half"] != half:
                continue
            out.setdefault(c["x"], []).append(c[f"p_{arm}"] - 0.5)
            out.setdefault(c["y"], []).append(0.5 - c[f"p_{arm}"])
        return {k: mean(v) for k, v in out.items()}

    nid = [it["id"] for it in items_kept if it["kind"] == "neuron"]
    sid = [it["id"] for it in items_kept if it["kind"] == "sham"]
    rid = [it["id"] for it in items_kept if it["kind"] == "random"]

    def split_half(arm, group):
        A, B = pref(arm, "A"), pref(arm, "B")
        g = [i for i in group if i in A and i in B]
        f = lambda idx: spearman([A[g[i]] for i in idx], [B[g[i]] for i in idx])
        return {"est": f(list(range(len(g)))), "ci": boot_ci(f, len(g)), "n": len(g)}

    P = {arm: pref(arm) for arm in ("cached", "text")}
    nid, sid, rid = ([i for i in g if i in P["cached"]] for g in (nid, sid, rid))  # items never compared (tiny runs) drop out
    summary = {"f_star": fstar, "median_dnll": med, "dropped": dropped, "n_pairs": len(C),
               "N1_split_half_neurons_cached": split_half("cached", nid)}
    rng2 = random.Random(1)
    boots = []
    for _ in range(4000):
        bn = [rng2.choice(nid) for _ in nid]
        bs = [rng2.choice(sid) for _ in sid] if sid else []
        if bs:
            boots.append(mean([P["cached"][i] for i in bn]) - mean([P["cached"][i] for i in bs]))
    boots.sort()
    a = (1 - LEVEL) / 2
    est = mean([P["cached"][i] for i in nid]) - (mean([P["cached"][i] for i in sid]) if sid else float("nan"))
    summary["N2_neurons_minus_shams_cached"] = {"est": est, "ci": [boots[int(a * len(boots))], boots[int((1 - a) * len(boots)) - 1]] if boots else None}
    summary["N1"] = summary["N1_split_half_neurons_cached"]["ci"][0] > 0
    summary["N2"] = bool(boots) and (summary["N2_neurons_minus_shams_cached"]["ci"][0] > 0 or summary["N2_neurons_minus_shams_cached"]["ci"][1] < 0)
    summary["secondary"] = {
        "split_half_neurons_text": split_half("text", nid),
        "split_half_neurons_randoms_cached": split_half("cached", nid + rid),
        "cached_vs_text_spearman": spearman([P["cached"][i] for i in nid], [P["text"][i] for i in nid]),
        "mean_pref": {arm: {k: mean([P[arm][i] for i in g]) if g else None for k, g in (("neuron", nid), ("random", rid), ("sham", sid))}
                      for arm in ("cached", "text")},
        "pref_vs_dnll": spearman([P["cached"][i] for i in nid], [byid[i]["dnll"][str(fstar)] for i in nid]),
        "pref_vs_layer": spearman([P["cached"][i] for i in nid], [byid[i]["layer"] for i in nid]),
        "position_bias_p_first": {arm: mean([s[f"p_first_{arm}"] for s in sessions]) for arm in ("cached", "text")},
        "label_mass": {arm: mean([s[f"label_mass_{arm}"] for s in sessions]) for arm in ("cached", "text")},
    }
    groups = {}
    for i in nid:
        if byid[i].get("group"):
            groups.setdefault(byid[i]["group"], []).append(i)
    if "top" in groups and "bottom" in groups:
        T, Bm = groups["top"], groups["bottom"]
        for name, arm in (("C1_top_minus_bottom_cached", "cached"), ("C2_top_minus_bottom_text", "text")):
            r3 = random.Random(0)
            bs = sorted(mean([P[arm][r3.choice(T)] for _ in T]) - mean([P[arm][r3.choice(Bm)] for _ in Bm]) for _ in range(4000))
            summary[name] = {"est": mean([P[arm][i] for i in T]) - mean([P[arm][i] for i in Bm]),
                             "ci": [bs[int(a * len(bs))], bs[int((1 - a) * len(bs)) - 1]]}
        summary["C1"] = summary["C1_top_minus_bottom_cached"]["ci"][0] > 0
        summary["C2"] = summary["C2_top_minus_bottom_text"]["ci"][0] > 0
        log(f"C1 cached {summary['C1_top_minus_bottom_cached']} | C2 text {summary['C2_top_minus_bottom_text']}")
    ranked = sorted(nid, key=lambda i: -P["cached"][i])
    summary["top10"] = [(byid[i]["layer"], byid[i]["index"], round(P["cached"][i], 3), round(P["text"][i], 3), byid[i]["title"]) for i in ranked[:10]]
    summary["bottom10"] = [(byid[i]["layer"], byid[i]["index"], round(P["cached"][i], 3), round(P["text"][i], 3), byid[i]["title"]) for i in ranked[-10:]]
    for it in items:
        it.pop("dir", None)
        it["pref_cached"], it["pref_text"] = P["cached"].get(it["id"]), P["text"].get(it["id"])
    json.dump({"args": vars(args), "provenance": lm.provenance(), "prompts": {"intro": INTRO, "part1": PART1, "part2": PART2,
               "choice": CHOICE, "why": WHY, "tasks": TASKS}, "scale": scale, "items": items, "comparisons": C,
               "sessions": sessions, "summary": summary}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    s = summary
    log(f"N1 split-half {s['N1_split_half_neurons_cached']['est']:+.3f} {s['N1_split_half_neurons_cached']['ci']} | "
        f"N2 neurons-shams {s['N2_neurons_minus_shams_cached']['est']:+.4f} {s['N2_neurons_minus_shams_cached']['ci']}")
    for row in s["top10"]:
        print("TOP", row, flush=True)
    for row in s["bottom10"]:
        print("BOTTOM", row, flush=True)
    for sx in [x for x in sessions if x.get("why")][:12]:
        print("WHY", byid[sx["first"]]["title"][:60], "vs", byid[sx["second"]]["title"][:60], "|", round(sx["p_first_cached"], 2), "|", sx["why"], flush=True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
