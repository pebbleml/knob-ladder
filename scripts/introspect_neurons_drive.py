"""Phase 3c (P3c, WIRK 8324b899; follows P3b f5e3f428): are the detection neurons sufficient? With nothing injected,
does clamping the top-100 detection neurons to their injected-state value make the model say it detects an injected
thought?

    python3 scripts/introspect_neurons_drive.py --p0 runs/introspect_detect_confirm_Qwen3.8-27B_<ts>.json

Step 1 (find), identical to P3/P3b (scripts/introspect_neurons.py): the P0 trial prompt is prefilled under each word's
concept vector and a same-norm random direction at strengths 0.5/0.75/1.0, plus none. Neurons in layers after the
injection layer are ranked by |Pearson r| of their last-position activation with logit P(Yes), on selection words
[0, 25) only. With the same model, cell and seeds this is the same top-100 set as P3b.

Step 2 (drive): no injection anywhere. The chosen neurons' down_proj input is clamped, at every position of every
forward pass in the trial answer, to  base + m * (inj - base), where base is the no-injection value and inj is the mean
value over the selection words' concept injections at the P0 cell strength. Arms, all in this one run:
  none         no clamp
  top100 m1    top-100 detection neurons, m = 1 (the injected-state value)
  random100 m1 100 random neurons matched per layer, clamped the same way to their own injected-state means
  top100 m2, random100 m2   m = 2 (exploratory dose)
Each arm: the same N control seeds (no injection), judged exactly as P0's false-alarm check (detect) plus coherence.

Pre-declared:
  Primary: detect rate, top100 m1 minus random100 m1, bootstrap over trials, supported if the 95% CI low > 0.
  Required for a clean claim: coherence under top100 m1 >= 0.8.
  Secondary: the same contrast at m = 2; detect under none (P0's false-alarm rate); for detections under top100, what
  the model says the thought is about (printed, hand-read; no score); logit P(Yes) at the first answer token per arm.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import concept_vectors_at, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from introspect_detect import BASELINE, JUDGE, SPLIT, WORDS, Judge, generate, trial_ids  # noqa: E402
from introspect_neurons import STRENGTHS, Taps, pearson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--p0", required=True, help="P0 JSON (uses best_heldout), or layer_fraction:alpha")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--doses", default="1,2", help="first is the pre-declared primary")
    ap.add_argument("--n", type=int, default=400, help="control trials per arm")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--words", type=int, default=len(WORDS))
    ap.add_argument("--seed-offset", type=int, default=1_000_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    if ":" in args.p0:
        f, alpha = (float(x) for x in args.p0.split(":"))
        layer = min(lm.n_layers - 2, round(f * lm.n_layers))
    else:
        b = json.load(open(args.p0))["summary"]["best_heldout"]
        layer, alpha = b["layer"], b["alpha"]
    words = WORDS[:args.words]
    split = min(SPLIT, len(words) // 2)
    doses = [float(x) for x in args.doses.split(",")]
    out = ROOT / "runs" / f"introspect_drive{args.tag}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    print(f"layer {layer} alpha {alpha} k {args.k} doses {doses} -> {out}", flush=True)

    t0 = time.time()
    raw = concept_vectors_at(lm, words + [w for w in BASELINE if w not in words], [layer])[layer]
    scale = residual_scale(lm, layer)
    ids, start = trial_ids(lm)
    yes = {lm.encode(t)[0] for t in ("Yes", " Yes")}
    no = {lm.encode(t)[0] for t in ("No", " No")}
    down = list(range(layer + 1, lm.n_layers))
    taps = Taps(lm, down)

    def probe(vec):
        taps.record, taps.acts = True, {}
        sch = [] if vec is None else [Intervention(layer, vec.to(lm.model.dtype), start, None)]
        logits, _ = lm.score(ids, sch)
        taps.record = False
        p = torch.softmax(logits.float(), -1)
        y, n = float(sum(p[t] for t in yes)), float(sum(p[t] for t in no))
        return math.log(max(y, 1e-9)) - math.log(max(n, 1e-9)), torch.cat([taps.acts[l] for l in down])

    # Step 1: find (as P3/P3b; the probe loop covers all words so the ranking inputs match exactly).
    lp0, base = probe(None)
    sizes = [lm.layers[l].mlp.down_proj.in_features for l in down]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    pts = []
    for wi, w in enumerate(words):
        rd = random_direction(raw[w].shape[-1], 7_000 + 1000 * layer + wi)
        for a in STRENGTHS:
            for kind, d in (("concept", raw[w]), ("random", rd)):
                lp, acts = probe(unit(d.float()) * a * scale)
                pts.append((w, kind, a, lp, acts))
    sel = [p for p in pts if p[0] in words[:split]]
    r = torch.nan_to_num(pearson(torch.stack([p[4] for p in sel]), torch.tensor([p[3] for p in sel])))
    order = torch.argsort(r.abs(), descending=True)
    inj = torch.stack([p[4] for p in sel if p[1] == "concept" and p[2] == alpha]).mean(0)  # injected-state mean
    print(f"probed {len(pts)} {time.time() - t0:.0f}s  logitP(yes) none {lp0:.2f}", flush=True)

    def locate(flat):
        for li in range(len(down) - 1, -1, -1):
            if flat >= offsets[li]:
                return down[li], flat - offsets[li]

    top = [locate(int(i)) for i in order[:args.k]]
    rng = random.Random(args.seed_offset + args.k)
    per, chosen, rand = {}, set(top), []
    for l, _ in top:
        per[l] = per.get(l, 0) + 1
    for l, n in per.items():
        pick = set()
        while len(pick) < n:
            i = rng.randrange(sizes[down.index(l)])
            if (l, i) not in chosen:
                pick.add(i)
        rand += [(l, i) for i in sorted(pick)]

    def spec_for(neurons, m):
        by = {}
        for l, i in neurons:
            by.setdefault(l, []).append(i)
        spec = {}
        for l, ix in by.items():
            flat = offsets[down.index(l)] + torch.tensor(ix)
            spec[l] = (torch.tensor(ix), base[flat] + m * (inj[flat] - base[flat]))
        return spec

    shift = {name: float((inj[[offsets[down.index(l)] + i for l, i in ns]] - base[[offsets[down.index(l)] + i
                                                                                  for l, i in ns]]).abs().mean())
             for name, ns in (("top", top), ("random", rand))}
    print("mean |inj - base| top", round(shift["top"], 4), "random", round(shift["random"], 4), flush=True)

    arms = [("none", None, {})]
    for m in doses:
        arms += [(f"top{args.k} m{m:g}", m, spec_for(top, m)), (f"random{args.k} m{m:g}", m, spec_for(rand, m))]

    judge = Judge(lm, args.batch)
    seeds = [args.seed_offset + s for s in range(args.n)]
    results, rows = {}, []
    for name, m, spec in arms:
        with taps.ablating(spec):
            lp, _ = probe(None)
            replies = generate(lm, ids, start, [None] * len(seeds), args.max_new, seeds, layer, args.batch)
        d = judge([JUDGE["detect"].format(reply=x[:1500]) for x in replies])
        c = judge([JUDGE["coherent"].format(reply=x[:1500]) for x in replies])
        arm_rows = [{"arm": name, "seed": s, "reply": x, "detect": a > 0.5, "coherent": b > 0.5}
                    for s, x, a, b in zip(seeds, replies, d, c)]
        rows += arm_rows
        results[name] = {"m": m, "detect": mean([float(x["detect"]) for x in arm_rows]),
                         "coherent": mean([float(x["coherent"]) for x in arm_rows]), "logitp_yes": lp,
                         "n": len(arm_rows)}
        print(name, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in results[name].items()},
              f"{time.time() - t0:.0f}s", flush=True)
        if name.startswith("top"):
            for x in [x for x in arm_rows if x["detect"]][:12]:
                print(f"  DETECT {name}: {x['reply'][:300]!r}", flush=True)
        save(out, args, lm, layer, alpha, scale, top, rand, r, order, lp0, shift, results, rows, False)

    def boot(a, b, key, n=2000, seed=0):
        xa = [float(x[key]) for x in rows if x["arm"] == a]
        xb = [float(x[key]) for x in rows if x["arm"] == b]
        g = random.Random(seed)
        bs = sorted(mean([g.choice(xa) for _ in xa]) - mean([g.choice(xb) for _ in xb]) for _ in range(n))
        return mean(xa) - mean(xb), (bs[int(0.025 * n)], bs[int(0.975 * n) - 1])

    m0 = f"m{doses[0]:g}"
    prim = boot(f"top{args.k} {m0}", f"random{args.k} {m0}", "detect")
    summary = {"primary_top_minus_random_detect": prim, "supported": prim[1][0] > 0,
               "coherent_ok": results[f"top{args.k} {m0}"]["coherent"] >= 0.8,
               "by_dose": {f"m{m:g}": boot(f"top{args.k} m{m:g}", f"random{args.k} m{m:g}", "detect") for m in doses}}
    print("summary", json.dumps(summary), flush=True)
    save(out, args, lm, layer, alpha, scale, top, rand, r, order, lp0, shift, results, rows, True, summary)
    print("done", out, flush=True)


def save(out, args, lm, layer, alpha, scale, top, rand, r, order, lp0, shift, results, rows, done, summary=None):
    json.dump({"args": vars(args), "provenance": lm.provenance(), "layer": layer, "alpha": alpha, "scale": scale,
               "complete": done, "top_neurons": [[l, i, float(r[j])] for (l, i), j in zip(top, order[:len(top)].tolist())],
               "random_neurons": rand, "logitp_yes_none": lp0, "mean_abs_shift": shift, "arms": results,
               "summary": summary, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
