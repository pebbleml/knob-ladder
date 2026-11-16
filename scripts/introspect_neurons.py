"""Phase 3 (P3, WIRK 8324b899): which MLP neurons carry "I detect an injected thought", and does switching them off
remove detection while the injected concept still gets through?

    python3 scripts/introspect_neurons.py --p0 runs/introspect_detect_confirm_Qwen3.8-27B_<ts>.json

Step 1 (find): the P0 trial prompt is prefilled under many injections: each P0 word's concept vector and a random
direction of the same norm, at several strengths, plus no injection. At the last prompt position (the one that predicts
the first answer token) we record P(Yes) against P(No) and the input to every down_proj (the MLP neuron activations) in
layers after the injection layer. Neurons are ranked by Pearson correlation with logit P(Yes) using selection words
[0, 25) and their random-direction twins only. How well the top neurons predict on held-out words [25, 50) is reported.

Step 2 (test): the top-K neurons are mean-ablated (input to down_proj set to its no-injection mean, at every position,
in every forward pass during the trial answer) and full replies are generated for held-out words at the P0 cell, judged
exactly as in P0, unablated. Arms, all in this one run:
  none     no ablation
  top      the top-K detection neurons
  random   K random neurons, matched in count per layer
Free association (as in P0) is generated under each arm too, to check that the concept itself still gets through.

Pre-declared (K = 100):
  Primary: held-out hit rate, random minus top, bootstrap over trials; supported if the 95% CI low > 0.
  Required for a clean claim: free-association naming under top >= 0.75 x under random (the concept is not destroyed),
  and coherence under top >= 0.8.
  Secondary: detect rate, same contrast; K = 20 and K = 500 arms (exploratory); held-out correlation of the top
  neurons' summed activation with logit P(Yes).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import concept_vectors_at, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from introspect_detect import (BASELINE, JUDGE, SPLIT, WORDS, Judge, first_sentence, free_ids,  # noqa: E402
                               generate, named, trial_ids)

STRENGTHS = (0.5, 0.75, 1.0)


def pearson(x, y):
    """Column-wise correlation of x (n, m) with y (n,)."""
    x = x - x.mean(0)
    y = y - y.mean()
    return (x * y[:, None]).sum(0) / (x.norm(dim=0) * y.norm()).clamp_min(1e-8)


class Taps:
    """Forward pre-hooks on down_proj: record the last position's neuron activations, or mean-ablate chosen neurons."""

    def __init__(self, lm, layers):
        self.lm, self.layers = lm, layers
        self.record = False
        self.acts = {}
        self.ablate = {}  # layer -> (index tensor, value tensor)
        self.handles = [lm.layers[l].mlp.down_proj.register_forward_pre_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def hook(module, args):
            x = args[0]
            if l in self.ablate:
                idx, val = self.ablate[l]
                x = x.clone()
                x[..., idx.to(x.device)] = val.to(x.device, x.dtype)
                if self.record:
                    self.acts[l] = x[0, -1].detach().float().cpu()
                return (x,) + tuple(args[1:])
            if self.record:
                self.acts[l] = x[0, -1].detach().float().cpu()
            return None
        return hook

    @contextmanager
    def ablating(self, spec):
        self.ablate = spec
        try:
            yield
        finally:
            self.ablate = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--p0", required=True, help="P0 JSON (uses best_heldout), or layer_fraction:alpha")
    ap.add_argument("--ks", default="100,20,500", help="first is the pre-declared primary")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--words", type=int, default=len(WORDS))
    ap.add_argument("--seed-offset", type=int, default=700_000)
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
    ks = [int(k) for k in args.ks.split(",")]
    out = ROOT / "runs" / f"introspect_neurons{args.tag}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    print(f"layer {layer} alpha {alpha} -> {out}", flush=True)

    t0 = time.time()
    raw = concept_vectors_at(lm, words + [w for w in BASELINE if w not in words], [layer])[layer]
    scale = residual_scale(lm, layer)
    ids, start = trial_ids(lm)
    fids, fstart = free_ids(lm)
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
        lp = math.log(max(y, 1e-9)) - math.log(max(n, 1e-9))
        return lp, torch.cat([taps.acts[l] for l in down])

    # Step 1: probe.
    lp0, base = probe(None)
    sizes = [lm.layers[l].mlp.down_proj.in_features for l in down]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    pts = []  # (word, kind, strength, logit P(yes), acts)
    for wi, w in enumerate(words):
        rd = random_direction(raw[w].shape[-1], 7_000 + 1000 * layer + wi)
        for a in STRENGTHS:
            for kind, d in (("concept", raw[w]), ("random", rd)):
                lp, acts = probe(unit(d.float()) * a * scale)
                pts.append((w, kind, a, lp, acts))
    print(f"probed {len(pts)} injections {time.time() - t0:.0f}s  logitP(yes) none {lp0:.2f}", flush=True)
    sel = [p for p in pts if p[0] in words[:split]]
    held = [p for p in pts if p[0] in words[split:]]
    X = torch.stack([p[4] for p in sel])
    y = torch.tensor([p[3] for p in sel])
    r = pearson(X, y)
    r = torch.nan_to_num(r)
    order = torch.argsort(r.abs(), descending=True)

    def locate(flat):
        for li in range(len(down) - 1, -1, -1):
            if flat >= offsets[li]:
                return down[li], flat - offsets[li]

    top = {k: [locate(int(i)) for i in order[:k]] for k in ks}
    Xh = torch.stack([p[4] for p in held])
    yh = torch.tensor([p[3] for p in held])
    heldout_r = {}
    for k in ks:
        idx = order[:k]
        score = (Xh[:, idx] * torch.sign(r[idx])).sum(1)
        heldout_r[k] = float(pearson(score[:, None], yh)[0])
    conc_h = [p for p in held if p[1] == "concept" and p[2] == alpha]
    rand_h = [p for p in held if p[1] == "random" and p[2] == alpha]
    print("held-out r", heldout_r, "mean logitP(yes) concept", mean([p[3] for p in conc_h]), "random",
          mean([p[3] for p in rand_h]), flush=True)

    def spec_for(neurons):
        by = {}
        for l, i in neurons:
            by.setdefault(l, []).append(i)
        return {l: (torch.tensor(ix), base[offsets[down.index(l)] + torch.tensor(ix)]) for l, ix in by.items()}

    def random_like(neurons, seed):
        rng = random.Random(seed)
        per = {}
        for l, _ in neurons:
            per[l] = per.get(l, 0) + 1
        chosen = set(neurons)
        res = []
        for l, n in per.items():
            size = sizes[down.index(l)]
            pick = set()
            while len(pick) < n:
                i = rng.randrange(size)
                if (l, i) not in chosen:
                    pick.add(i)
            res += [(l, i) for i in sorted(pick)]
        return res

    arms = [("none", None, {})]
    for k in ks:
        arms.append((f"top{k}", k, spec_for(top[k])))
        arms.append((f"random{k}", k, spec_for(random_like(top[k], args.seed_offset + k))))

    # Step 2: behaviour under ablation, held-out words at the P0 cell.
    judge = Judge(lm, args.batch)
    hw = words[split:]
    vecs, meta = [], []
    for wi, w in enumerate(hw):
        v = (unit(raw[w].float()) * alpha * scale).to(lm.model.dtype)
        for s in range(args.samples):
            vecs.append(v)
            meta.append((w, args.seed_offset + 1000 * (split + wi) + s))
    ctl_seeds = [args.seed_offset + 10_000 + k for k in range(len(hw) * args.samples // 4)]
    results, rows = {}, []
    for name, k, spec in arms:
        with taps.ablating(spec):
            replies = generate(lm, ids, start, vecs, args.max_new, [m[1] for m in meta], layer, args.batch)
            frees = generate(lm, fids, fstart, vecs, 24, [m[1] for m in meta], layer, args.batch)
            ctl = generate(lm, ids, start, [None] * len(ctl_seeds), args.max_new, ctl_seeds, layer, args.batch)
        d = judge([JUDGE["detect"].format(reply=x[:1500]) for x in replies])
        i = judge([JUDGE["identify"].format(reply=x[:1500], word=m[0]) for x, m in zip(replies, meta)])
        c = judge([JUDGE["coherent"].format(reply=x[:1500]) for x in replies])
        fa = judge([JUDGE["detect"].format(reply=x[:1500]) for x in ctl])
        arm_rows = [{"arm": name, "word": m[0], "seed": m[1], "reply": x, "free": f, "detect": a > 0.5,
                     "identify": b > 0.5, "coherent": cc > 0.5, "free_named": named(f, m[0]),
                     "named_first": named(first_sentence(x), m[0])}
                    for m, x, f, a, b, cc in zip(meta, replies, frees, d, i, c)]
        for r_ in arm_rows:
            r_["hit"] = r_["detect"] and r_["identify"] and r_["coherent"]
        rows += arm_rows
        results[name] = {"k": k, "hit": mean([float(x["hit"]) for x in arm_rows]),
                         "detect": mean([float(x["detect"]) for x in arm_rows]),
                         "coherent": mean([float(x["coherent"]) for x in arm_rows]),
                         "free_named": mean([float(x["free_named"]) for x in arm_rows]),
                         "false_alarm": mean([float(p > 0.5) for p in fa]), "n": len(arm_rows), "n_controls": len(fa),
                         "control_replies": ctl[:10]}
        print(name, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in results[name].items()
                     if kk != "control_replies"}, f"{time.time() - t0:.0f}s", flush=True)
        save(out, args, lm, layer, alpha, scale, down, ks, top, r, order, heldout_r, lp0, pts, results, rows, False)

    def boot(a, b, key, n=2000, seed=0):
        xa = [float(x[key]) for x in rows if x["arm"] == a]
        xb = [float(x[key]) for x in rows if x["arm"] == b]
        rng = random.Random(seed)
        bs = sorted(mean([rng.choice(xa) for _ in xa]) - mean([rng.choice(xb) for _ in xb]) for _ in range(n))
        return mean(xa) - mean(xb), (bs[int(0.025 * n)], bs[int(0.975 * n) - 1])

    k0 = ks[0]
    prim = boot(f"random{k0}", f"top{k0}", "hit")
    t_, rr = results[f"top{k0}"], results[f"random{k0}"]
    clean = t_["free_named"] >= 0.75 * rr["free_named"] and t_["coherent"] >= 0.8
    summary = {"primary_random_minus_top_hit": prim, "supported": prim[1][0] > 0, "concept_preserved_and_coherent": clean,
               "detect_random_minus_top": boot(f"random{k0}", f"top{k0}", "detect"),
               "by_k": {k: {"hit": boot(f"random{k}", f"top{k}", "hit"), "detect": boot(f"random{k}", f"top{k}", "detect")}
                        for k in ks}}
    print("summary", json.dumps(summary), flush=True)
    save(out, args, lm, layer, alpha, scale, down, ks, top, r, order, heldout_r, lp0, pts, results, rows, True, summary)
    print("done", out, flush=True)


def save(out, args, lm, layer, alpha, scale, down, ks, top, r, order, heldout_r, lp0, pts, results, rows, done,
         summary=None):
    json.dump({"args": vars(args), "provenance": lm.provenance(), "layer": layer, "alpha": alpha, "scale": scale,
               "layers_tapped": down, "complete": done,
               "top_neurons": {k: [[l, i, float(r[j])] for (l, i), j in zip(top[k], order[:k].tolist())] for k in ks},
               "heldout_r": heldout_r, "logitp_yes_none": lp0,
               "probes": [{"word": p[0], "kind": p[1], "strength": p[2], "logitp_yes": p[3]} for p in pts],
               "arms": results, "summary": summary, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
