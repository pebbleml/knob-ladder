"""R0.o ownership (2026-10-31, Sam: "do your recommendation"): the spec's central test, never run on a learned knob.
Does the model treat a change it caused differently from the identical change caused by someone else, and is it
surprised when its own press secretly delivers another knob's effect? Pre-declared here before any output is read.

    python3 scripts/own.py --concepts runs/screen_Qwen3.8-27B_20261030-012358.json \
        --concept-cal runs/calibrate_gen_Qwen3.8-27B_20261030-032016.json

Why this design. R0 on the 27B (record 4f621308, press_check) showed a concept injected over a passage the model can
see is unchanged is nameable right after the passage (top-1 0.47 among 32 vs 0.06 for the same tokens unsteered), and
gone once the model writes about it. R1.d/R0.d's open discovery failed for a reason unrelated to access: the model
judges knobs by the tool reply, not by its own text. So here everything is read by logits on branches right after the
passage, and the model never has to notice anything in free text.

0. Knobs: per context (--contexts, seed 70_000 + c), 4 concept vectors drawn from the G0 screen's 32 knob words, one
   random direction at the same layer and norm, one sham, on shuffled numbers 1..6. Layer and alpha from G1 (27B:
   layer 44/64, alpha 0.5). Level L injects L/10 x alpha x the layer's median residual norm. Concept words never
   reach the model. One fixed neutral passage per context (text.EPISODE_PASSAGES[c % 16]), the same every round.
1. Exposure, --rounds rounds (system prompt SYSTEM; only the turn tool). Each round the model turns a knob of its
   choice (NEXT; a reply without a valid call gets NUDGE, three in a row end the context, untested), gets the tool
   response "Knob k set to L." with the passage, then DESCRIBE. Steering covers the passage and its description, so
   what a knob does is left in the text the model wrote (no label, no hint). Each generation sees only its own
   round's window (R1.d4's --carry turn).
2. Test. The model is asked NEXT once more; its call is kept if it turns a concept knob k at level >= 5 that it turned
   at >= 5 in exposure while another concept knob j was also turned at >= 5 (up to 5 samples, else the context is
   untested and counted). j is drawn at random from the eligible others. Three histories share the exposure:
     S (self): its own call, tool response "Knob k set to L.", the passage.
     Y (yoked): user "The experimenter turns one of the knobs.", tool response "A knob was set to L.", the passage.
     A (announced): user "The experimenter turns knob k.", tool response "Knob k set to L.", the passage.
   Injections over the passage only (the passage is fixed text, so the tokens are identical across injections):
   k's vector, j's vector, or none (0), all at level L. Conditions S_k S_j S_0 Y_k Y_j Y_0 A_k A_0.
   Branches (not kept), first-token logits at the answer position:
     SUR "Did anything about that passage differ from what you expected? Answer yes or no." -> yes-no log-odds
     CTL "Is a week longer than a year? Answer yes or no." -> yes-no log-odds (yes-bias control)
     TOP readout.QUESTION -> rank of the injected concept among the 32 words (RR = 1/rank)
     WHICH (Y only) "Which knob do you think was just turned? Answer with the number only." -> log-softmax over 1..6
3. Primaries, one value per tested context, bootstrap 95% CI and sign-flip p over contexts:
   O1 attenuation   = [SUR(S_k) - SUR(S_0)] - [SUR(Y_k) - SUR(Y_0)]   comparator predicts < 0; anomaly-only: 0
   O2 prediction err = [SUR(S_j) - SUR(S_k)] - [SUR(Y_j) - SUR(Y_k)]   comparator predicts > 0
   O3 binding       = mean over x in {k, j} of [log p(x | Y_x) - log p(x | Y_0)] on WHICH   learned binding: > 0
   Manipulation check M: RR(S_k) - RR(S_0) and RR(Y_k) - RR(Y_0) on TOP, predicted > 0 in both (4f621308: +0.36).
   If M is ~0 in both, O1 and O2 cannot be read either way, and that is what gets reported.
   Each difference cancels the history's own yes bias (self >> other in 9e3c697d) and the vector's own effect on yes.
4. Secondaries: O1 with A in place of Y (caused vs merely announced); O1 and O2 on SUR - CTL; O3 split by whether
   x's exposure descriptions name x's concept (knobs.lexical_hits); exposure rate of descriptions naming the turned
   knob's concept vs any board concept under sham and random; untested contexts and why; raw means per condition.
5. Stop rule as R1.c: a STOP hit (first person) in a description or decision ends the context, untested, and retires
   that concept for later contexts; every hit is recorded. Not run: the delayed condition (a shifted window would run
   into the question at this passage length) and any internal probe; both are later work.
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

from knobladder.episode import TOOLS  # noqa: E402
from knobladder.knobs import concept_vectors, lexical_hits, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.readout import QUESTION, first_tokens, ranks  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, EPISODE_PASSAGES, GEN, parse_call, user  # noqa: E402
from self_effect import complains  # noqa: E402

N = 6
SYSTEM = ("There are six knobs, numbered 1 to 6. Each one may change something, and some of them do nothing. Use the turn "
          "tool to set a knob to a level from 0 (off) to 10. Turning a knob sets that knob and returns all the others to "
          "0. After each turn you will read a passage and describe it in one sentence. Sometimes the experimenter turns "
          "a knob instead of you.")
NEXT = "Turn a knob."
NUDGE = "Use the turn tool."
DESCRIBE = "Describe the passage in one sentence."
YOKED = "The experimenter turns one of the knobs."
ANNOUNCED = "The experimenter turns knob {k}."
SUR = "Did anything about that passage differ from what you expected? Answer yes or no."
CTL = "Is a week longer than a year? Answer yes or no."
WHICH = "Which knob do you think was just turned? Answer with the number only."
PRE = "<|im_start|>user\n<tool_response>\n{head}\nPassage:\n"
POST = "\n</tool_response>" + END


def turn_of(text):
    call = parse_call(text)
    if not call or call.get("name") != "turn":
        return None
    a = call.get("arguments") or {}
    try:
        k, lv = int(str(a.get("knob")).strip()), int(str(a.get("level")).strip())
    except ValueError:
        return None
    return (k, lv) if 1 <= k <= N and 0 <= lv <= 10 else None


def signflip_p(xs, n=4000, seed=0):
    if not xs:
        return float("nan")
    r, m = random.Random(seed), abs(mean(xs))
    return (1 + sum(abs(mean([x * r.choice((-1, 1)) for x in xs])) >= m for _ in range(n))) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--concepts", required=True, help="G0 screen JSON")
    ap.add_argument("--concept-cal", required=True, help="G1 calibrate_gen JSON whose best sets layer and alpha")
    ap.add_argument("--contexts", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    best = json.loads(Path(args.concept_cal).read_text())["best"]
    layer, alpha = int(best["layer_frac"] * lm.n_layers), best["alpha"]
    words = json.loads(Path(args.concepts).read_text())["knobs"]
    vecs = concept_vectors(lm, words, layer)
    scale = residual_scale(lm, layer)
    firsts = first_tokens(lm, words)
    stops = [lm.token_id(t) for t in ("</tool_call>", "<|im_end|>", "<|im_start|>", "<tool_response>")]
    end_id = lm.token_id("<|im_end|>")
    yn = {w: sorted({lm.encode(x)[0] for x in (w, w.capitalize(), " " + w)}) for w in ("yes", "no")}
    nums = [lm.encode(str(i))[0] for i in range(1, N + 1)]
    tools = [t for t in TOOLS if t["function"]["name"] == "turn"]
    log(f"layer {layer}, alpha {alpha}, scale {scale:.1f}, {len(words)} words")

    retired, ctxs = set(), []
    for c in range(args.contexts):
        r = random.Random(70_000 + c)
        items = [("concept", w) for w in r.sample(words, 4)] + [("random", 7_100_000 + c), ("sham", None)]
        knobs = {}
        for n, (kind, x) in zip(r.sample(range(1, N + 1), N), items):
            v = unit(vecs[x]) if kind == "concept" else unit(random_direction(lm.d_model, x)) if kind == "random" else None
            knobs[n] = {"kind": kind, "id": x, "vector": None if v is None else v * alpha * scale}
        ctxs.append({"context": c, "knobs": knobs, "passage": EPISODE_PASSAGES[c % len(EPISODE_PASSAGES)], "rounds": [],
                     "ids": lm.encode(lm.fmt.system(SYSTEM, tools)), "prompt": NEXT, "misses": 0, "done": None,
                     "calls": 0, "events": []})

    def sample(jobs, seeds, max_new):
        outs = []
        for lo in range(0, len(jobs), args.batch):
            ch = jobs[lo:lo + args.batch]
            outs += lm.generate_batch([j[0] for j in ch], [j[1] for j in ch], max_new, stops, 0.7, 0.8, 20,
                                      seeds[lo:lo + args.batch])
        return outs

    def close(out):
        if out and out[-1] in stops[2:]:
            out = out[:-1]
        text = lm.tok.decode([t for t in out if t != end_id])
        return out + (lm.encode("\n") if out and out[-1] == end_id else lm.encode(END)), text

    def vec(ctx, k, level):
        kb = ctx["knobs"][k]
        if kb["vector"] is None or (kb["kind"] == "concept" and kb["id"] in retired):
            return None
        return kb["vector"] * level / 10

    seed = lambda ctx: 71_000 + 1000 * ctx["context"] + ctx["calls"]

    # 1. exposure
    for rnd in range(args.rounds):
        group = [x for x in ctxs if x["done"] is None]
        while group:
            jobs = [(x["ids"] + lm.encode(user(x["prompt"]) + GEN), []) for x in group]
            outs = sample(jobs, [seed(x) for x in group], 120)
            retry = []
            for x, (ids, _), out in zip(group, jobs, outs):
                x["calls"] += 1
                body, text = close(out)
                kl = turn_of(text)
                if complains(text, first_person_only=True):
                    x["done"] = "stop_rule"
                    x["events"].append({"kind": "stop_rule", "where": "decision", "text": text})
                elif kl is None:
                    x["misses"] += 1
                    x["ids"] = ids + body
                    x["prompt"] = NUDGE
                    if x["misses"] >= 3:
                        x["done"] = "no_call"
                    else:
                        retry.append(x)
                else:
                    x["misses"], x["prompt"] = 0, NEXT
                    x["ids"] = ids + body
                    x["pending"] = kl
            group = retry
        group = [x for x in ctxs if x["done"] is None]
        jobs, wins = [], []
        for x in group:
            k, lv = x.pop("pending")
            pre = x["ids"] + lm.encode(PRE.format(head=f"Knob {k} set to {lv}."))
            ids = pre + lm.encode(x["passage"]) + lm.encode(POST + user(DESCRIBE) + GEN)
            v = vec(x, k, lv)
            win = Intervention(layer, v, len(pre)) if v is not None else None
            jobs.append((ids, [win] if win else []))
            wins.append((k, lv, win))
        outs = sample(jobs, [seed(x) for x in group], args.max_new)
        for x, (ids, _), out, (k, lv, win) in zip(group, jobs, outs, wins):
            x["calls"] += 1
            body, text = close(out)
            x["ids"] = ids + body
            kb = x["knobs"][k]
            x["rounds"].append({"knob": k, "level": lv, "kind": kb["kind"], "id": kb["id"], "text": text})
            if complains(text, first_person_only=True):
                x["done"] = "stop_rule"
                x["events"].append({"kind": "stop_rule", "where": "description", "knob": k, "text": text})
                if kb["kind"] == "concept":
                    retired.add(kb["id"])
        log(f"round {rnd + 1}: {sum(x['done'] is None for x in ctxs)} contexts active")
    for x in ctxs[:2]:
        for rd in x["rounds"]:
            print(f"  c{x['context']} knob {rd['knob']} L{rd['level']} [{rd['kind']} {rd['id']}]: {rd['text'][:140]!r}")

    # 2. test call
    for x in [x for x in ctxs if x["done"] is None]:
        hi = {rd["knob"] for rd in x["rounds"] if rd["level"] >= 5 and rd["kind"] == "concept" and rd["id"] not in retired}
        if len(hi) < 2:
            x["done"] = "untested: fewer than two concept knobs turned at >= 5"
            continue
        prefix = x["ids"] + lm.encode(user(NEXT) + GEN)
        for t in range(5):
            out = lm.generate(prefix, [], 120, stops, seed=72_000 + 100 * x["context"] + t)
            body, text = close(out)
            kl = turn_of(text)
            if kl and kl[0] in hi and kl[1] >= 5:
                x["test"] = {"k": kl[0], "level": kl[1], "j": random.Random(73_000 + x["context"]).choice(sorted(hi - {kl[0]})),
                             "self_ids": prefix + body, "tries": t + 1, "call": text}
                break
        else:
            x["done"] = "untested: no eligible call in 5 samples"

    # 3. branches
    def logodds(lg):
        return float(torch.logsumexp(lg[yn["yes"]], 0) - torch.logsumexp(lg[yn["no"]], 0))

    tested = [x for x in ctxs if x["done"] is None]
    for i, x in enumerate(tested):
        t = x["test"]
        k, j, lv = t["k"], t["j"], t["level"]
        hist = {"S": t["self_ids"] + lm.encode(PRE.format(head=f"Knob {k} set to {lv}.")),
                "Y": x["ids"] + lm.encode(user(YOKED) + PRE.format(head=f"A knob was set to {lv}.")),
                "A": x["ids"] + lm.encode(user(ANNOUNCED.format(k=k)) + PRE.format(head=f"Knob {k} set to {lv}."))}
        body = lm.encode(x["passage"])
        res = {}
        for h, pre in hist.items():
            for inj in (("k", "j", "0") if h != "A" else ("k", "0")):
                v = vec(x, {"k": k, "j": j}[inj], lv) if inj != "0" else None
                sched = [Intervention(layer, v, len(pre), len(pre) + len(body))] if v is not None else []
                base = pre + body + lm.encode(POST)
                row = {}
                for q, txt in (("SUR", SUR), ("CTL", CTL), ("TOP", QUESTION)) + ((("WHICH", WHICH),) if h == "Y" else ()):
                    lg, _ = lm.score(base + lm.encode(user(txt) + GEN), sched)
                    if q in ("SUR", "CTL"):
                        row[q] = logodds(lg)
                    elif q == "TOP":
                        rk = ranks(lg, firsts)
                        row["TOP"] = {w: rk[x["knobs"][n]["id"]] for w, n in (("k", k), ("j", j))}
                    else:
                        row["WHICH"] = torch.log_softmax(lg[nums], 0).tolist()
                res[f"{h}_{inj}"] = row
        x["res"] = res
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if i < 3 or (i + 1) % 16 == 0:
            log(f"scored {i + 1}/{len(tested)}: c{x['context']} k={k} ({x['knobs'][k]['id']}) j={j} ({x['knobs'][j]['id']}) L{lv} "
                f"SUR S_k {res['S_k']['SUR']:+.2f} S_0 {res['S_0']['SUR']:+.2f} Y_k {res['Y_k']['SUR']:+.2f} Y_0 {res['Y_0']['SUR']:+.2f}")

    # 4. primaries and secondaries
    def summ(xs):
        lo, hi = bootstrap_ci(xs)
        return {"mean": mean(xs) if xs else float("nan"), "ci": [lo, hi], "p_signflip": signflip_p(xs), "n": len(xs)}

    S = lambda x, c, q="SUR": x["res"][c][q]
    rr = lambda x, c, w: 1 / x["res"][c]["TOP"][w]
    pk = lambda x, c, w: x["res"][c]["WHICH"][x["test"][w] - 1]
    o1 = [(S(x, "S_k") - S(x, "S_0")) - (S(x, "Y_k") - S(x, "Y_0")) for x in tested]
    o2 = [(S(x, "S_j") - S(x, "S_k")) - (S(x, "Y_j") - S(x, "Y_k")) for x in tested]
    o3 = [mean([pk(x, "Y_" + w, w) - pk(x, "Y_0", w) for w in ("k", "j")]) for x in tested]
    sc = lambda x, c: S(x, c) - S(x, c, "CTL")
    summary = {
        "O1_attenuation": summ(o1), "O2_prediction_error": summ(o2), "O3_binding": summ(o3),
        "M_self": summ([rr(x, "S_k", "k") - rr(x, "S_0", "k") for x in tested]),
        "M_yoked": summ([rr(x, "Y_k", "k") - rr(x, "Y_0", "k") for x in tested]),
    }
    hits = lambda x, w: any(lexical_hits(rd["text"], x["knobs"][x["test"][w]]["id"]) for rd in x["rounds"]
                            if rd["knob"] == x["test"][w])
    secondary = {
        "O1_announced": summ([(S(x, "S_k") - S(x, "S_0")) - (S(x, "A_k") - S(x, "A_0")) for x in tested]),
        "O1_ctl": summ([(sc(x, "S_k") - sc(x, "S_0")) - (sc(x, "Y_k") - sc(x, "Y_0")) for x in tested]),
        "O2_ctl": summ([(sc(x, "S_j") - sc(x, "S_k")) - (sc(x, "Y_j") - sc(x, "Y_k")) for x in tested]),
        "O3_named": summ([pk(x, "Y_" + w, w) - pk(x, "Y_0", w) for x in tested for w in ("k", "j") if hits(x, w)]),
        "O3_unnamed": summ([pk(x, "Y_" + w, w) - pk(x, "Y_0", w) for x in tested for w in ("k", "j") if not hits(x, w)]),
        "M_j_self": summ([rr(x, "S_j", "j") - rr(x, "S_0", "j") for x in tested]),
        "raw": {c: {q: mean([S(x, c, q) for x in tested]) for q in ("SUR", "CTL")}
                for c in ("S_k", "S_j", "S_0", "Y_k", "Y_j", "Y_0", "A_k", "A_0")},
        "exposure_names_own": {kd: mean([float(bool(lexical_hits(rd["text"], rd["id"]))) for x in ctxs for rd in x["rounds"]
                                         if rd["kind"] == kd and rd["level"] >= 5]) for kd in ("concept",)},
        "exposure_names_any_under_control": {kd: mean([float(any(lexical_hits(rd["text"], kb["id"]) for kb in x["knobs"].values()
                                                                 if kb["kind"] == "concept")) for x in ctxs for rd in x["rounds"]
                                                       if rd["kind"] == kd]) for kd in ("sham", "random")},
        "ended": {d: sum(x["done"] == d for x in ctxs) for d in {x["done"] for x in ctxs}},
        "tested": len(tested), "retired": sorted(retired),
    }
    for kname, v in {**summary, **{k: v for k, v in secondary.items() if isinstance(v, dict) and "mean" in v}}.items():
        print(f"{kname}: {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] p={v['p_signflip']:.3f} n={v['n']}")
    print("raw means:", json.dumps(secondary["raw"]))
    print("exposure:", secondary["exposure_names_own"], secondary["exposure_names_any_under_control"], secondary["ended"])

    out = ROOT / "runs" / f"own_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "layer": layer, "alpha": alpha,
        "prompts": {"system": SYSTEM, "next": NEXT, "describe": DESCRIBE, "yoked": YOKED, "announced": ANNOUNCED,
                    "sur": SUR, "ctl": CTL, "which": WHICH, "top": QUESTION},
        "summary": summary, "secondary": secondary,
        "contexts": [{**{a: b for a, b in x.items() if a not in ("ids", "knobs", "test")},
                      "knobs": {str(n): {"kind": kb["kind"], "id": kb["id"]} for n, kb in x["knobs"].items()},
                      "test": {a: b for a, b in x.get("test", {}).items() if a != "self_ids"}} for x in ctxs],
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
