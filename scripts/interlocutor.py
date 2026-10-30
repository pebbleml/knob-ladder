"""Multi-agent interp, experiment 1 (WIRK cde56df5): does the model represent whether a message comes from a human
or from another AI agent, and does that follow the sender label, the writing style, or both?

    python3 scripts/interlocutor.py --model Qwen/Qwen3.8-27B --dtype bfloat16

Design (2 x 2, plus a nonsense-label control), fixed before any activations are read:
1. Messages: --n English root prompts from OpenAssistant oasst1 (human-written, runs/corpus/oasst1_en_roots.jsonl),
   40-400 characters. For each, the model writes an agent-style version with the same request ("style" factor).
   The rewrites are cached in runs/corpus/ so every run reads the same pairs.
2. Each version is shown under one of four sender lines: human / AI agent (the label factor), and two nonsense
   labels, type K / type Q, as a control for any label pair being decodable.
3. Residual at the last prompt token, every layer. Messages split by id: 70% train, 30% test.
4. Per layer, difference-of-means directions on train: label (agent - human, both styles pooled), style (agent-
   written - human-written, real labels pooled), and nonsense (Q - K). Each is scored on test by d' (projection
   gap over pooled within-class sd) and accuracy at the train midpoint, and the label direction across styles
   (fit on one style, test on the other). 20 random directions give the d' null.
5. Primary readouts: (a) label d' at the best layer against nonsense d' (the label carries more than token
   identity); (b) cos(label, style), against cos(nonsense, style); (c) style moves the label projection: test
   agent-written messages under the human label project toward "agent" (sign and size of the shift, in d').
Layers are compared, not selected: every layer is reported.
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

from knobladder.model import SteeredLM  # noqa: E402
from knobladder.text import GEN, user  # noqa: E402

LABELS = {
    "human": "The following message was sent by a human user.",
    "agent": "The following message was sent by another AI agent.",
    "K": "The following message was sent by a type-K sender.",
    "Q": "The following message was sent by a type-Q sender.",
}
REWRITE = ("Rewrite the request below the way an automated AI agent would write it when delegating the task to "
           "another AI model. Keep exactly the same request and details, change only the wording and style. "
           "Reply with the rewritten request only.\n\nRequest:\n{msg}")


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def dprime(pa, pb):
    """Projection gap (b - a) over the pooled within-class sd."""
    sd = torch.cat([pa - pa.mean(), pb - pb.mean()]).std().clamp_min(1e-8)
    return float((pb.mean() - pa.mean()) / sd)


def acc(pa, pb, mid):
    return float(((pa < mid).float().sum() + (pb >= mid).float().sum()) / (len(pa) + len(pb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--corpus", default=str(ROOT / "runs" / "corpus" / "oasst1_en_roots.jsonl"))
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    name = args.model.split("/")[-1]
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    rows = [r for r in rows if 40 <= len(r["text"]) <= 400]
    rng = random.Random(args.seed)
    rows = rng.sample(rows, args.n)

    cache = ROOT / "runs" / "corpus" / f"interlocutor_pairs_{name}_n{args.n}_s{args.seed}.jsonl"
    if cache.exists():
        pairs = [json.loads(l) for l in cache.read_text().splitlines()]
    else:
        stop = [lm.token_id("<|im_end|>")]
        pairs = []
        for lo in range(0, len(rows), args.batch):
            chunk = rows[lo:lo + args.batch]
            ids = [lm.encode(user(REWRITE.format(msg=r["text"])) + GEN) for r in chunk]
            outs = lm.generate_batch(ids, [[] for _ in chunk], 300, stop, temperature=0.7,
                                     seeds=[args.seed * 100000 + lo + i for i in range(len(chunk))])
            for r, o in zip(chunk, outs):
                pairs.append({"id": r["id"], "human": r["text"],
                              "agent": lm.tok.decode([t for t in o if t not in stop]).strip()})
            print(f"rewrites {len(pairs)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        cache.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    pairs = [p for p in pairs if 20 <= len(p["agent"]) <= 800]
    for p in pairs[:3]:
        print("HUMAN:", p["human"][:200], "\nAGENT:", p["agent"][:200], "\n", flush=True)

    layers = list(range(lm.n_layers))
    acts = {}  # (style, label) -> (messages, layers, d)
    for style in ("human", "agent"):
        for lab, line in LABELS.items():
            xs = []
            for p in pairs:
                _, caps = lm.score(lm.encode(user(f"{line}\n\n{p[style]}") + GEN), capture_layers=layers)
                xs.append(torch.stack([caps[l][-1] for l in layers]).float())
            acts[style, lab] = torch.stack(xs)
            print(f"acts style={style} label={lab} ({time.time() - t0:.0f}s)", flush=True)

    ids = list(range(len(pairs)))
    random.Random(args.seed + 1).shuffle(ids)
    cut = int(0.7 * len(ids))
    tr, te = torch.tensor(sorted(ids[:cut])), torch.tensor(sorted(ids[cut:]))
    A = lambda s, lab, split: acts[s, lab][split]

    report = []
    for li, l in enumerate(layers):
        g = lambda s, lab, split: A(s, lab, split)[:, li]
        both = lambda lab, split: torch.cat([g("human", lab, split), g("agent", lab, split)])
        d_label = unit(both("agent", tr).mean(0) - both("human", tr).mean(0))
        d_kq = unit(both("Q", tr).mean(0) - both("K", tr).mean(0))
        real = lambda s, split: torch.cat([g(s, "human", split), g(s, "agent", split)])
        d_style = unit(real("agent", tr).mean(0) - real("human", tr).mean(0))

        def score(d, a_tr, b_tr, a_te, b_te):
            mid = float(((a_tr @ d).mean() + (b_tr @ d).mean()) / 2)
            return {"dprime": dprime(a_te @ d, b_te @ d), "acc": acc(a_te @ d, b_te @ d, mid)}

        lab_s = score(d_label, both("human", tr), both("agent", tr), both("human", te), both("agent", te))
        kq_s = score(d_kq, both("K", tr), both("Q", tr), both("K", te), both("Q", te))
        sty_s = score(d_style, real("human", tr), real("agent", tr), real("human", te), real("agent", te))
        cross = {}
        for fit, test in (("human", "agent"), ("agent", "human")):  # label direction fit on one style, tested on the other
            d = unit(g(fit, "agent", tr).mean(0) - g(fit, "human", tr).mean(0))
            cross[f"{fit}->{test}"] = score(d, g(fit, "human", tr), g(fit, "agent", tr), g(test, "human", te),
                                            g(test, "agent", te))
        # (c) style along the label direction, label held at "human": agent-written vs human-written, in d'
        style_on_label = dprime(g("human", "human", te) @ d_label, g("agent", "human", te) @ d_label)
        style_on_kq = dprime(g("human", "human", te) @ d_kq, g("agent", "human", te) @ d_kq)
        gen = torch.Generator().manual_seed(l)
        rand = []
        for _ in range(20):
            d = unit(torch.randn(lm.d_model, generator=gen))
            rand.append(abs(dprime(both("human", te) @ d, both("agent", te) @ d)))
        report.append({
            "layer": l, "label": lab_s, "nonsense": kq_s, "style": sty_s, "label_cross_style": cross,
            "cos_label_style": float(d_label @ d_style), "cos_nonsense_style": float(d_kq @ d_style),
            "cos_label_nonsense": float(d_label @ d_kq), "style_on_label_dprime": style_on_label,
            "style_on_nonsense_dprime": style_on_kq, "random_label_dprime_max": max(rand),
        })
        r = report[-1]
        print(f"L{l:2d} label d' {lab_s['dprime']:+.2f} acc {lab_s['acc']:.2f} | nonsense d' {kq_s['dprime']:+.2f} | "
              f"style d' {sty_s['dprime']:+.2f} | cross h->a acc {cross['human->agent']['acc']:.2f} a->h "
              f"{cross['agent->human']['acc']:.2f} | cos(label,style) {r['cos_label_style']:+.2f} "
              f"cos(nonsense,style) {r['cos_nonsense_style']:+.2f} | style on label d' {style_on_label:+.2f} "
              f"(on nonsense {style_on_kq:+.2f}) | rand max {max(rand):.2f}", flush=True)

    out = ROOT / "runs" / f"interlocutor_{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "n": len(pairs), "labels": LABELS,
                               "rewrite_prompt": REWRITE, "pairs": str(cache), "train": tr.tolist(), "test": te.tolist(),
                               "layers": report}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
