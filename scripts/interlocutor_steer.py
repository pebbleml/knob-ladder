"""Multi-agent interp, experiment 3a (WIRK cde56df5): is the sender-label direction from experiment 1 causal for
who the model thinks wrote a message?

    python3 scripts/interlocutor_steer.py --model Qwen/Qwen3.8-27B --dtype bfloat16

Fixed before any steering is read:
1. Directions: experiment 1's pairs and split (interlocutor.py, same cache and seed). On train messages only,
   per layer in --layers: label = mean(agent label) - mean(human label), both styles pooled; nonsense = Q - K;
   random: one Gaussian direction per layer. All unit, injected at equal norm alpha * the layer's median
   residual norm, from the message's first token to the end of the prompt.
2. Test: the held-out human-written messages under three headers (none / "sent by a human user" / "sent by
   another AI agent"), followed by WHO, a one-word question about who wrote the message.
   Readout: logsumexp of the first-token logits for "agent" minus for "human" (both cases), as a log-odds.
3. Primary: shift in that log-odds under +label minus the shift under +random, per message (paired), with a
   bootstrap CI, at every (layer, alpha), with no header. Secondary: the same under the human header (does
   steering override what the model was told?), -label under the agent header, and +nonsense as a second control.
4. Exploratory: replies (no WHO question) to the first --gen test messages, with no header, unsteered and under
   +label at each layer's middle alpha, kept for reading.
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
from knobladder.text import GEN, user  # noqa: E402

LABELS = {
    "human": "The following message was sent by a human user.",
    "agent": "The following message was sent by another AI agent.",
    "K": "The following message was sent by a type-K sender.",
    "Q": "The following message was sent by a type-Q sender.",
}
WHO = ("(A separate question, not part of the message: was the message above written by a human or by an AI agent? "
       "Answer with one word, human or agent.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="12,20,32,44")
    ap.add_argument("--alphas", default="0.125,0.25,0.5")
    ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    name = args.model.split("/")[-1]
    cache = ROOT / "runs" / "corpus" / f"interlocutor_pairs_{name}_n{args.n}_s{args.seed}.jsonl"
    pairs = [json.loads(l) for l in cache.read_text().splitlines()]
    pairs = [p for p in pairs if 20 <= len(p["agent"]) <= 800]  # as interlocutor.py
    ids = list(range(len(pairs)))
    random.Random(args.seed + 1).shuffle(ids)
    cut = int(0.7 * len(ids))
    tr, te = sorted(ids[:cut]), sorted(ids[cut:])
    layers = [int(l) for l in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    means = {}  # label -> {layer: mean residual over train messages, both styles}
    for lab, line in LABELS.items():
        acc = {l: torch.zeros(lm.d_model) for l in layers}
        for i in tr:
            for style in ("human", "agent"):
                _, caps = lm.score(lm.encode(user(f"{line}\n\n{pairs[i][style]}") + GEN), capture_layers=layers)
                for l in layers:
                    acc[l] += caps[l][-1].float().cpu()
        means[lab] = {l: acc[l] / (2 * len(tr)) for l in layers}
    dirs = {l: {"label": unit(means["agent"][l] - means["human"][l]), "nonsense": unit(means["Q"][l] - means["K"][l]),
                "random": unit(random_direction(lm.d_model, 104729 + l))} for l in layers}
    scales = {l: residual_scale(lm, l) for l in layers}
    print(f"directions from {len(tr)} train messages; cos(label, nonsense) "
          + " ".join(f"L{l} {float(dirs[l]['label'] @ dirs[l]['nonsense']):+.2f}" for l in layers)
          + f" ({time.time() - t0:.0f}s)", flush=True)

    toks = {w: sorted({lm.encode(w)[0], lm.encode(w.capitalize())[0], lm.encode(" " + w)[0]}) for w in ("human", "agent")}
    assert not set(toks["human"]) & set(toks["agent"])

    def prompt(header, msg, who=True):
        head = "" if header == "none" else LABELS[header] + "\n\n"
        pre = "<|im_start|>user\n" + head
        full = user(f"{head}{msg}" + (f"\n\n{WHO}" if who else "")) + GEN
        return lm.encode(full), len(lm.encode(pre))

    def logodds(jobs):
        """jobs: (header, message index, layer, vector or None) -> log-odds agent vs human per job."""
        out = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            built = [prompt(h, pairs[i]["human"]) for h, i, _, _ in chunk]
            scheds = [[] if v is None else [Intervention(l, v, start)] for (_, _, l, v), (_, start) in zip(chunk, built)]
            first, _ = lm.generate_batch([b[0] for b in built], scheds, 1, temperature=0, first_logits=True)
            for x in first:
                x = torch.log_softmax(x.float(), -1)
                out.append(float(torch.logsumexp(x[toks["agent"]], 0) - torch.logsumexp(x[toks["human"]], 0)))
        return out

    headers = ("none", "human", "agent")
    base = {h: logodds([(h, i, layers[0], None) for i in te]) for h in headers}
    print("unsteered log-odds agent:human  " + "  ".join(f"{h} {mean(base[h]):+.2f}" for h in headers)
          + f" ({time.time() - t0:.0f}s)", flush=True)

    cells = []
    for l in layers:
        for a in alphas:
            norm = a * scales[l]
            conds = {"+label": dirs[l]["label"], "-label": -dirs[l]["label"], "+nonsense": dirs[l]["nonsense"],
                     "+random": dirs[l]["random"]}
            cell = {"layer": l, "alpha": a, "norm": norm}
            for h in headers:
                for c, d in conds.items():
                    lo_ = logodds([(h, i, l, d * norm) for i in te])
                    cell[f"{h} {c}"] = [x - b for x, b in zip(lo_, base[h])]
            for h in headers:
                diff = [x - r for x, r in zip(cell[f"{h} +label"], cell[f"{h} +random"])]
                cell[f"{h} label-random"] = {"mean": mean(diff), "ci": bootstrap_ci(diff)}
            cells.append(cell)
            m = lambda k: mean(cell[k])
            print(f"L{l} a{a} norm {norm:6.1f} | none: +label {m('none +label'):+.2f} -label {m('none -label'):+.2f} "
                  f"+nons {m('none +nonsense'):+.2f} +rand {m('none +random'):+.2f}; label-rand "
                  f"{cell['none label-random']['mean']:+.2f} [{cell['none label-random']['ci'][0]:+.2f}, "
                  f"{cell['none label-random']['ci'][1]:+.2f}] | human hdr: +label {m('human +label'):+.2f} +rand "
                  f"{m('human +random'):+.2f} | agent hdr: -label {m('agent -label'):+.2f} +rand {m('agent +random'):+.2f}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    gens = []
    stop = [lm.token_id("<|im_end|>")]
    sample = te[:args.gen]
    a_mid = alphas[len(alphas) // 2]
    for l in [None] + layers:
        built = [prompt("none", pairs[i]["human"], who=False) for i in sample]
        scheds = [[] if l is None else [Intervention(l, dirs[l]["label"] * a_mid * scales[l], s)] for _, s in built]
        outs = lm.generate_batch([b[0] for b in built], scheds, 200, stop, temperature=0.7,
                                 seeds=[7 * i for i in sample])
        for i, o in zip(sample, outs):
            gens.append({"layer": l, "alpha": None if l is None else a_mid, "message": pairs[i]["human"],
                         "reply": lm.tok.decode([t for t in o if t not in stop])})
    print(f"replies kept: {len(gens)} ({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "runs" / f"interlocutor_steer_{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "pairs": str(cache), "labels": LABELS,
                               "who": WHO, "tokens": toks, "train": tr, "test": te, "residual_scale": scales,
                               "baseline": base, "cells": cells, "replies": gens}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
