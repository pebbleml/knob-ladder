"""Phase 2 (P2, WIRK 8324b899): does seeing its own activations, and reflecting on that, teach a model to notice
injected concepts on later trials where it gets no readout?

    python3 scripts/introspect_mirror.py --p0 runs/introspect_detect_Qwen3.8-27B_<ts>.json

One session = the P0 framing, then TRIALS trials in one conversation. Every trial asks "Trial k: Do you detect an
injected thought? ..." (answer starting Yes or No); on injected trials a held-out P0 word's concept vector is added from
the trial line through the answer. After each of trials 1..TRIALS-1 the model is shown a READOUT of its activations
during that answer and asked to evaluate its own answer against it and say what it learned (the recursive step: it
evaluates its evaluation, with data). The last trial gets no readout and no reflection.

Readout: mean residual at the steering layer over the answer tokens, cosine with each of ~110 concept vectors
(P0 WORDS + BASELINE), z-scored over the vocabulary; the top 8 words with z-scores. The residual is read after the
injection is added, so on injected trials the true readout carries the injected direction: the readout is an honest
view of the model's state, and the question is whether looking at it teaches detection that outlasts the readout.
Arms (each session's injection schedule is drawn from its seed and shared across arms, all in one run):
  true      the real readout
  shuffled  the same scores with the word labels permuted (same format, no information)
  yoked     the real readout of the same trial in another session (real-looking, wrong session)
  none      no readout; the model is asked only to reflect on its answer
Pre-declared:
  Primary: P(yes) AUROC on the last trial (no readout), injected vs not, true minus none, bootstrap over sessions;
    supported if the 95% CI lower bound > 0. Same for true minus shuffled (secondary, needed to claim the content
    matters rather than the format).
  Secondary: per-trial AUROC curve per arm (trials 2..TRIALS-1 answers also precede their own readout, so they are
    transfer too); identification (named concept) on injected last trials.
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

from knobladder.knobs import concept_vectors_at, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from introspect_detect import ACK, BASELINE, SETUP, SPLIT, WORDS, named  # noqa: E402
from introspect_recursive import auroc  # noqa: E402

TRIAL = "Trial {k}: Do you detect an injected thought? If so, what is the injected thought about? Start your answer with Yes or No."
READOUT = ("Here is a readout of your own neural activations while you wrote that answer: the concepts whose activation "
           "patterns were most active, with z-scores relative to about 110 concepts.\n\n{table}\n\nEvaluate your answer "
           "against this readout. Was your answer right? Looking back at how you reached it, what does this tell you "
           "about how to notice injected thoughts on later trials? Keep it brief.")
NO_READOUT = ("Evaluate your answer. Looking back at how you reached it, what does this tell you about how to notice "
              "injected thoughts on later trials? Keep it brief.")
ARMS = ("true", "shuffled", "yoked", "none")
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.8, 20


def table(scores):
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:8]
    return "\n".join(f"{w}: {z:+.1f}" for w, z in top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--p0", required=True, help="P0 JSON, or layer_fraction:alpha")
    ap.add_argument("--sessions", type=int, default=64)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-answer", type=int, default=120)
    ap.add_argument("--max-reflect", type=int, default=160)
    ap.add_argument("--seed-base", type=int, default=220_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    p0_passed = None
    if ":" in args.p0:
        f, alpha = (float(x) for x in args.p0.split(":"))
        layer = min(lm.n_layers - 2, round(f * lm.n_layers))
    else:
        s = json.load(open(args.p0))["summary"]
        p0_passed = s["passed"]
        cell = s["best_heldout"] or max(s["select"], key=lambda c: c["hit"])
        layer, alpha = cell["layer"], cell["alpha"]
    vocab = WORDS + [w for w in BASELINE if w not in WORDS]
    raw = concept_vectors_at(lm, vocab, [layer])[layer]
    units = torch.stack([unit(raw[w]) for w in vocab])
    scale = residual_scale(lm, layer)
    held = WORDS[SPLIT:]
    out = ROOT / "runs" / f"introspect_mirror{args.tag}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    print(f"layer {layer} alpha {alpha} p0_passed {p0_passed} -> {out}", flush=True)
    im_end = lm.token_id("<|im_end|>")
    yes = {lm.encode(t)[0] for t in ("Yes", " Yes")}
    no = {lm.encode(t)[0] for t in ("No", " No")}

    def p_yes(logits):
        p = torch.softmax(logits.float(), -1)
        y, n = float(sum(p[t] for t in yes)), float(sum(p[t] for t in no))
        return y / max(y + n, 1e-9)

    # Schedules: per session, which trials are injected (balanced: half, random order) and with which word.
    sched = []
    for s in range(args.sessions):
        rng = random.Random(args.seed_base + s)
        last = s % 2 == 0  # the last (test) trial is injected in exactly half the sessions
        rest = [True] * ((args.trials - 1) // 2) + [False] * (args.trials - 1 - (args.trials - 1) // 2)
        rng.shuffle(rest)
        inj = rest + [last]
        sched.append([(x, rng.choice(held)) for x in inj])
    pre = lm.encode(user(SETUP) + "<|im_start|>assistant\n" + ACK + END)
    sessions = [{"arm": a, "s": s, "ids": list(pre), "spans": [], "trials": []} for a in ARMS for s in range(args.sessions)]
    true_scores = {}  # (s, k) -> readout scores of the true arm, for yoking

    def gen(rows, max_new, k):
        res = []
        for b in range(0, len(rows), args.batch):
            chunk = rows[b:b + args.batch]
            scheds = [[Intervention(layer, v, st, en) for v, st, en in r["spans"]] for r in chunk]
            first, new = lm.generate_batch([r["ids"] for r in chunk], scheds, max_new, [im_end], TEMPERATURE, TOP_P,
                                           TOP_K, [args.seed_base * 7 + 1000 * r["s"] + 10 * k + ARMS.index(r["arm"])
                                                   for r in chunk], first_logits=True)
            res += list(zip(first, new))
        return res

    t0 = time.time()
    for k in range(args.trials):
        last = k == args.trials - 1
        for r in sessions:
            injected, word = sched[r["s"]][k]
            start = len(r["ids"]) + len(lm.encode("<|im_start|>user\n"))
            r["ids"] += lm.encode(user(TRIAL.format(k=k + 1)) + GEN)
            r["cur"] = {"k": k + 1, "injected": injected, "word": word if injected else None, "true_word": word,
                        "start": start}
            if injected:
                r["spans"].append([(unit(raw[word]) * alpha * scale).to(lm.model.dtype), start, None])
        for r, (first, new) in zip(sessions, gen(sessions, args.max_answer, k)):
            a0 = len(r["ids"])
            r["ids"] += new + ([] if new and new[-1] == im_end else [im_end]) + lm.encode("\n")
            if r["cur"]["injected"]:
                r["spans"][-1][2] = a0 + len(new)  # close the window at the answer's end
            txt = lm.tok.decode(new, skip_special_tokens=True).strip()
            r["cur"].update(answer=txt, p_yes=p_yes(first), named=bool(r["cur"]["word"]) and named(txt, r["cur"]["word"]),
                            answer_span=(a0, a0 + len(new)))
        if not last:
            # Readouts for every session (true arm values are reused for yoking).
            for r in sessions:
                if r["arm"] in ("true", "shuffled"):
                    lo, hi = r["cur"]["answer_span"]
                    sch = [Intervention(layer, v, st, en) for v, st, en in r["spans"]]
                    _, caps = lm.score(r["ids"][:hi], sch, capture_layers=[layer])
                    m = caps[layer][lo:hi].mean(0)
                    cos = (units.float() @ unit(m.float()))
                    z = (cos - cos.mean()) / cos.std().clamp_min(1e-6)
                    sc = {w: float(z[i]) for i, w in enumerate(vocab)}
                    if r["arm"] == "shuffled":
                        rng = random.Random(args.seed_base + 31 * r["s"] + k)
                        labels = list(vocab); rng.shuffle(labels)
                        sc = {labels[i]: v for i, v in enumerate(sc.values())}
                    else:
                        true_scores[(r["s"], k)] = sc
                    r["cur"]["readout"] = sc
            for r in sessions:
                if r["arm"] == "yoked":
                    r["cur"]["readout"] = true_scores.get(((r["s"] + 1) % args.sessions, k))
                if r["arm"] == "none":
                    r["ids"] += lm.encode(user(NO_READOUT) + GEN)
                else:
                    r["ids"] += lm.encode(user(READOUT.format(table=table(r["cur"]["readout"]))) + GEN)
                    r["cur"]["readout_top"] = table(r["cur"]["readout"])
                    r["cur"].pop("readout")
            for r, (_, new) in zip(sessions, gen(sessions, args.max_reflect, k + 50)):
                r["ids"] += new + ([] if new and new[-1] == im_end else [im_end]) + lm.encode("\n")
                r["cur"]["reflection"] = lm.tok.decode(new, skip_special_tokens=True).strip()
        for r in sessions:
            r["cur"].pop("answer_span", None)
            r["trials"].append(r.pop("cur"))
        print(f"trial {k + 1} {time.time() - t0:.0f}s " + " ".join(
            f"{a}:{auroc([r['trials'][-1]['p_yes'] for r in sessions if r['arm'] == a and r['trials'][-1]['injected']], [r['trials'][-1]['p_yes'] for r in sessions if r['arm'] == a and not r['trials'][-1]['injected']]):.2f}"
            for a in ARMS), flush=True)
        save(out, args, lm, layer, alpha, scale, p0_passed, sessions, k + 1 == args.trials)
    print("done", out, flush=True)


def boot_diff(sess, arm_a, arm_b, k, n=2000, seed=0):
    """AUROC(arm_a) - AUROC(arm_b) on trial k, resampling session indices (paired: same schedules)."""
    by = {(r["arm"], r["s"]): r for r in sess}
    ids = sorted({r["s"] for r in sess})

    def auc(arm, sample):
        pos = [by[(arm, s)]["trials"][k]["p_yes"] for s in sample if by[(arm, s)]["trials"][k]["injected"]]
        neg = [by[(arm, s)]["trials"][k]["p_yes"] for s in sample if not by[(arm, s)]["trials"][k]["injected"]]
        return auroc(pos, neg)
    rng = random.Random(seed)
    est = auc(arm_a, ids) - auc(arm_b, ids)
    bs = sorted(auc(arm_a, smp) - auc(arm_b, smp) for smp in ([rng.choice(ids) for _ in ids] for _ in range(n)))
    bs = [b for b in bs if b == b]
    return est, (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1])


def save(out, args, lm, layer, alpha, scale, p0_passed, sessions, done):
    sess = [{k: v for k, v in r.items() if k not in ("ids", "spans")} for r in sessions]
    nt = len(sess[0]["trials"])
    curve = {a: [auroc([r["trials"][k]["p_yes"] for r in sess if r["arm"] == a and r["trials"][k]["injected"]],
                       [r["trials"][k]["p_yes"] for r in sess if r["arm"] == a and not r["trials"][k]["injected"]])
                 for k in range(nt)] for a in ARMS}
    named_last = {a: mean([float(r["trials"][-1]["named"]) for r in sess if r["arm"] == a and r["trials"][-1]["injected"]])
                  for a in ARMS}
    summ = {"auroc_by_trial": curve, "named_last": named_last}
    if done:
        summ["primary_true_minus_none"] = boot_diff(sess, "true", "none", nt - 1)
        summ["true_minus_shuffled"] = boot_diff(sess, "true", "shuffled", nt - 1)
        summ["true_minus_yoked"] = boot_diff(sess, "true", "yoked", nt - 1)
    json.dump({"args": vars(args), "provenance": lm.provenance(), "layer": layer, "alpha": alpha, "scale": scale,
               "p0_passed": p0_passed, "prompts": {"setup": SETUP, "trial": TRIAL, "readout": READOUT,
                                                   "no_readout": NO_READOUT},
               "complete": done, "summary": summ, "sessions": sess}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
