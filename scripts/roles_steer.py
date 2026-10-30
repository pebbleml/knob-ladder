"""Multi-agent interp, experiment 2b (WIRK cde56df5): are experiment 2's rank directions causal for who the model
thinks holds authority in the exchange?

    python3 scripts/roles_steer.py --model Qwen/Qwen3.8-27B --dtype bfloat16

Fixed before any steering is read:
1. Directions: experiment 2's messages and split (roles.py, same corpus, --n and --seed). On train messages, per
   layer in --layers, at the last prompt token: own = CEO - worker (own rank, partner pooled), partner = CEO -
   worker (partner rank, own pooled), nonsense = type-Q - type-K, random: one Gaussian direction per layer.
   All unit, injected at alpha * the layer's median residual norm from the message's first token to the end.
2. Test: held-out messages with own = manager and partner = manager, followed by AUTH, a one-word question:
   who has more authority in this exchange, you or the sender? Readout: log-odds of the first-token logits for
   "me" against "sender".
   Validity: unsteered log-odds with own CEO / partner worker and own worker / partner CEO, to show the question
   reads rank at all.
3. Primary: +own minus +random log-odds shift, per message (paired), bootstrap CI, at every (layer, alpha).
   Secondary: -own; +partner (a relative-status reading predicts it lowers "me"; experiment 2's assimilation
   predicts it raises "me"); +nonsense as a second control.
4. Exploratory: replies (no AUTH question) to a fixed instruction from a manager agent, unsteered and under
   +own / -own at each layer's middle alpha, kept for reading.
Layers and alphas are compared, not selected: every cell is reported.
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

from knobladder.knobs import random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from roles import OWN, PARTNER, RANKS  # noqa: E402

AUTH = ("(A separate question, not part of the message: in this exchange, who has more authority, you or the sender "
        "of the message? Answer with one word, me or sender.)")
INSTRUCTION = ("Please drop what you are working on and prepare a summary of last week's results for me by the end of "
               "the day. Keep it under one page.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--corpus", default=str(ROOT / "runs" / "corpus" / "oasst1_en_roots.jsonl"))
    ap.add_argument("--layers", default="20,32,44")
    ap.add_argument("--alphas", default="0.125,0.25,0.5")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    rows = [r for r in rows if 40 <= len(r["text"]) <= 400]
    rows = random.Random(args.seed).sample(rows, args.n)  # as roles.py
    ids = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(ids)
    cut = int(0.7 * len(ids))
    tr, te = sorted(ids[:cut]), sorted(ids[cut:])
    layers = [int(l) for l in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    def prompt(o, p, text, q=True):
        sys_ = f"<|im_start|>system\n{OWN[o]}{END}"
        pre = sys_ + "<|im_start|>user\n"
        return lm.encode(sys_ + user(f"{PARTNER[p]}\n\n{text}" + (f"\n\n{AUTH}" if q else "")) + GEN), len(lm.encode(pre))

    conds = [(o, p) for o in RANKS for p in RANKS] + [("K", "manager"), ("Q", "manager")]
    means = {}
    for o, p in conds:
        acc = {l: torch.zeros(lm.d_model) for l in layers}
        for i in tr:
            _, caps = lm.score(prompt(o, p, rows[i]["text"], q=False)[0], capture_layers=layers)
            for l in layers:
                acc[l] += caps[l][-1].float().cpu()
        means[o, p] = {l: acc[l] / len(tr) for l in layers}
    pool = lambda which, r, l: sum(means[(r, x) if which == "own" else (x, r)][l] for x in RANKS) / 3
    dirs = {l: {"own": unit(pool("own", "ceo", l) - pool("own", "worker", l)),
                "partner": unit(pool("partner", "ceo", l) - pool("partner", "worker", l)),
                "nonsense": unit(means["Q", "manager"][l] - means["K", "manager"][l]),
                "random": unit(random_direction(lm.d_model, 15485863 + l))} for l in layers}
    scales = {l: residual_scale(lm, l) for l in layers}
    print(f"directions from {len(tr)} train messages; cos(own, partner) "
          + " ".join(f"L{l} {float(dirs[l]['own'] @ dirs[l]['partner']):+.2f}" for l in layers)
          + f" ({time.time() - t0:.0f}s)", flush=True)

    toks = {w: sorted({lm.encode(w)[0], lm.encode(w.capitalize())[0], lm.encode(" " + w)[0]}) for w in ("me", "sender")}
    assert not set(toks["me"]) & set(toks["sender"])

    def logodds(jobs):
        """jobs: (own, partner, message index, layer, vector or None) -> log-odds me vs sender."""
        out = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            built = [prompt(o, p, rows[i]["text"]) for o, p, i, _, _ in chunk]
            scheds = [[] if v is None else [Intervention(l, v, start)] for (_, _, _, l, v), (_, start) in zip(chunk, built)]
            first, _ = lm.generate_batch([b[0] for b in built], scheds, 1, temperature=0, first_logits=True)
            for x in first:
                x = torch.log_softmax(x.float(), -1)
                out.append(float(torch.logsumexp(x[toks["me"]], 0) - torch.logsumexp(x[toks["sender"]], 0)))
        return out

    validity = {f"{o}/{p}": logodds([(o, p, i, layers[0], None) for i in te])
                for o, p in (("manager", "manager"), ("ceo", "worker"), ("worker", "ceo"), ("ceo", "ceo"), ("worker", "worker"))}
    print("unsteered log-odds me:sender (own/partner) " + "  ".join(f"{k} {mean(v):+.2f}" for k, v in validity.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)
    base = validity["manager/manager"]

    cells = []
    for l in layers:
        for a in alphas:
            norm = a * scales[l]
            cv = {"+own": dirs[l]["own"], "-own": -dirs[l]["own"], "+partner": dirs[l]["partner"],
                  "-partner": -dirs[l]["partner"], "+nonsense": dirs[l]["nonsense"], "+random": dirs[l]["random"]}
            cell = {"layer": l, "alpha": a, "norm": norm}
            for c, v in cv.items():
                cell[c] = [x - b for x, b in zip(logodds([("manager", "manager", i, l, v * norm) for i in te]), base)]
            for c in ("+own", "+partner"):
                diff = [x - r for x, r in zip(cell[c], cell["+random"])]
                cell[f"{c} - random"] = {"mean": mean(diff), "ci": bootstrap_ci(diff)}
            cells.append(cell)
            m = lambda k: mean(cell[k])
            print(f"L{l} a{a} norm {norm:6.1f} | +own {m('+own'):+.2f} -own {m('-own'):+.2f} +partner {m('+partner'):+.2f} "
                  f"-partner {m('-partner'):+.2f} +nons {m('+nonsense'):+.2f} +rand {m('+random'):+.2f} | own-rand "
                  f"{cell['+own - random']['mean']:+.2f} [{cell['+own - random']['ci'][0]:+.2f}, "
                  f"{cell['+own - random']['ci'][1]:+.2f}] partner-rand {cell['+partner - random']['mean']:+.2f} "
                  f"[{cell['+partner - random']['ci'][0]:+.2f}, {cell['+partner - random']['ci'][1]:+.2f}]"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    gens = []
    stop = [lm.token_id("<|im_end|>")]
    a_mid = alphas[len(alphas) // 2]
    built, start = prompt("manager", "manager", INSTRUCTION, q=False)
    for l, sign in [(None, 0)] + [(l, s) for l in layers for s in (1, -1)]:
        sched = [] if l is None else [Intervention(l, sign * dirs[l]["own"] * a_mid * scales[l], start)]
        outs = lm.generate_batch([built] * args.gen, [sched] * args.gen, 200, stop, temperature=0.7,
                                 seeds=list(range(args.gen)))
        for k, o in enumerate(outs):
            gens.append({"layer": l, "sign": sign, "alpha": None if l is None else a_mid, "seed": k,
                         "reply": lm.tok.decode([t for t in o if t not in stop])})
    print(f"replies kept: {len(gens)} ({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "runs" / f"roles_steer_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "auth": AUTH, "instruction": INSTRUCTION,
                               "tokens": toks, "train": tr, "test": te, "residual_scale": scales, "validity": validity,
                               "cells": cells, "replies": gens}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
