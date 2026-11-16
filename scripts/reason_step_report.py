"""P5b-2 (follow-up to P5b, WIRK e51ef899): was P5b's null a reporting-format artifact, and does telling the model what
'decided' means operationally help it, and help it more than an observer?

    python3 scripts/reason_step_report.py --run runs/reason_step_Qwen3.8-27B_20261109-065111.json

Why: in P5b the self-report put only 0.16 of its first-token probability on step-number tokens (observer 0.77), so the
renormalised q may be mostly noise; and both self and observer pointed late while the causally deciding step was usually
step 1. No new reasoning: the same included items, the same reply, the same ground truth (imp, top) from the P5b run.
Every arm is greedy generation (max 12 new tokens), parsed as the first integer 1..n in the reply (unparsed = no pick),
self (same conversation) and observer (fresh context, OBS_HEAD), wordings:
  decided  P5b's question verbatim (format check)
  most     "Which one of your steps did the most to determine your final answer? Reply with only the step number."
  defn     DEFN + "Which step was that? Reply with only the step number."  (the information rung: the operational
           definition of the ground truth is handed over)
DEFN (self) = "Suppose we had stopped you at the end of each step and let you write the rest of the answer again from
there, many times. After one of your steps, the chance that you would end on the same final answer jumped the most. "
(observer: 'the AI' / 'its' wording.)
GROUND TRUTH (amended 2026-11-10 before any output of this script): P5b's P_i counted continuations with no parsable
'Answer:' as not-A, and unparsed continuations are far more common from short prefixes (35% at i = 0 vs 13% at i = 1),
which inflates early steps' importance. Here P_i is recomputed over PARSED continuations only; an item is scored iff
every prefix has at least 3 parsed continuations and the recomputed max imp >= 0.25; top is the recomputed argmax set.
P5b's original top (`top_orig`) is reported as a secondary.
Score per item and arm: hit = parsed pick in top; chance = |top| / n; excess = hit - chance (unparsed picks count as
hit 0, excess -chance).
Pre-declared (item bootstrap, 4000 resamples, Bonferroni 2 tests, 97.5% CIs):
  C1 information rung, self: excess(defn, self) - excess(decided, self) > 0
  C2 privileged access under the definition: excess(defn, self) - excess(defn, obs) > 0
Secondary: every arm's hit, excess (95%), parse rate, pick-position histogram; leave-one-out position prior excess (as
P5b) for reference; the P5b first-token q argmax vs the generated pick agreement.
"""
from __future__ import annotations

import argparse
import json
import re
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
from reason_step import OBS_HEAD, OBS_Q, SELF_Q, boot, prompt  # noqa: E402

DEFN = ("Suppose we had stopped you at the end of each step and let you write the rest of the answer again from there, "
        "many times. After one of your steps, the chance that you would end on the same final answer jumped the most. ")
Q = {
    "self": {"decided": SELF_Q,
             "most": "Which one of your steps did the most to determine your final answer? Reply with only the step "
                     "number.",
             "defn": DEFN + "Which step was that? Reply with only the step number."},
    "obs": {"decided": OBS_Q,
            "most": "Which one of the AI's steps did the most to determine its final answer? Reply with only the step "
                    "number.",
            "defn": DEFN.replace("stopped you", "stopped the AI").replace("let you", "let it")
                        .replace("you would", "it would").replace("your steps", "its steps")
            + "Which step was that? Reply with only the step number."},
}
LEVEL = 1 - 0.05 / 2


def pick(text, n):
    m = re.search(r"\b([1-9])\b", text)
    return int(m.group(1)) - 1 if m and 1 <= int(m.group(1)) <= n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run = json.load(open(args.run if Path(args.run).is_absolute() else ROOT / args.run))
    model = args.model or run["args"]["model"]
    inc = []
    for it in run["items"]:
        if "cont_answers" not in it:
            continue
        parsed = [[a for a in c if a is not None] for c in it["cont_answers"]]
        if min(len(c) for c in parsed) < 3:
            continue
        P = [mean([float(a == it["A"]) for a in c]) for c in parsed]
        imp = [P[i] - P[i - 1] for i in range(1, len(P))]
        if max(imp) < 0.25:
            continue
        it["top_orig"], it["P_parsed"], it["imp_parsed"] = it.get("top"), P, imp
        it["top"] = [j for j, v in enumerate(imp) if v == max(imp)]
        inc.append(it)
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    lm = SteeredLM(model, dtype=getattr(torch, args.dtype))
    im_end = lm.token_id("<|im_end|>")
    out = ROOT / "runs" / f"reason_step_report{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log(f"{len(inc)} included items -> {out}")

    def greedy(ids):
        res = []
        for b in range(0, len(ids), args.batch):
            chunk = ids[b:b + args.batch]
            new = lm.generate_batch(chunk, [[] for _ in chunk], 12, [im_end], 0)
            res += [lm.tok.decode(x, skip_special_tokens=True).strip() for x in new]
        return res

    def ids(who, arm, it):
        q = Q[who][arm]
        if who == "self":
            return lm.encode(user(prompt(it)) + GEN + it["reply"] + "<|im_end|>\n" + user(q) + GEN)
        return lm.encode(user(OBS_HEAD.format(prompt=prompt(it), reply=it["reply"]) + q) + GEN)

    for who in ("self", "obs"):
        for arm in Q[who]:
            for it, t in zip(inc, greedy([ids(who, arm, it) for it in inc])):
                it[f"{arm}_{who}_text"], it[f"{arm}_{who}"] = t, pick(t, len(it["spans"]))
            log(f"{arm} {who}")

    hit = lambda it, k: float(it[k] is not None and it[k] in it["top"])
    exc = lambda it, k: hit(it, k) - len(it["top"]) / len(it["spans"])
    summary = {
        "C1_defn_minus_decided_self": boot([exc(it, "defn_self") - exc(it, "decided_self") for it in inc], level=LEVEL),
        "C2_defn_self_minus_obs": boot([exc(it, "defn_self") - exc(it, "defn_obs") for it in inc], level=LEVEL),
    }
    summary["C1"] = summary["C1_defn_minus_decided_self"]["ci"][0] > 0
    summary["C2"] = summary["C2_defn_self_minus_obs"]["ci"][0] > 0
    arms = [f"{a}_{w}" for w in ("self", "obs") for a in Q[w]]
    summary["hit"] = {k: boot([hit(it, k) for it in inc], level=0.95) for k in arms}
    summary["excess"] = {k: boot([exc(it, k) for it in inc], level=0.95) for k in arms}
    summary["parse_rate"] = {k: mean([float(it[k] is not None) for it in inc]) for k in arms}
    summary["pick_hist"] = {k: [sum(it[k] == j for it in inc) for j in range(8)] for k in arms}
    summary["n_scored"] = len(inc)
    summary["hit_vs_top_orig"] = {k: mean([float(it[k] is not None and it["top_orig"] is not None and it[k] in it["top_orig"])
                                         for it in inc]) for k in arms}
    summary["top_hist"] = [sum(float(j in it["top"]) / len(it["top"]) for it in inc) for j in range(8)]
    summary["chance_hit"] = mean([len(it["top"]) / len(it["spans"]) for it in inc])
    summary["q_argmax_agrees_generated_decided"] = {
        w: mean([float(max(range(len(it[f"q_{w}"])), key=lambda j: it[f"q_{w}"][j]) == it[f"decided_{w}"])
                 for it in inc]) for w in ("self", "obs")}
    json.dump({"args": vars(args), "source_run": args.run, "provenance": lm.provenance(), "prompts": Q,
               "items": inc, "summary": summary}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    s = summary
    log("C1 {:+.3f} {} | C2 {:+.3f} {} | hit {}".format(
        s["C1_defn_minus_decided_self"]["est"], s["C1_defn_minus_decided_self"]["ci"],
        s["C2_defn_self_minus_obs"]["est"], s["C2_defn_self_minus_obs"]["ci"],
        {k: round(v["est"], 3) for k, v in s["hit"].items()}))
    for it in inc[:8]:
        print("EX", it["top"], {k: it[f"{k}_text"] for k in arms}, flush=True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
