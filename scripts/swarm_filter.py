"""Multi-agent interp, experiment 6f (WIRK cde56df5; follows 6e, record f0f35986): does a tone-stripping summary block
what a faithful paraphrase passes? 6e found that an unsteered copy's paraphrase keeps 94% of a steered state (P +1.01),
with the caveat that its instruction kept every idea. Here a second rewriter is told to keep only the plain facts.

    python3 scripts/swarm_filter.py --source runs/swarm_memory_Qwen3.8-27B_20261103-021617.json

Fixed before anything is read (everything not named here is exactly 6e's):
Rewriters (unsteered 27B, plain assistant turn, 6e's sampling, 4 per post, seeds 81_000 + 10 * item + s for para as
  6e and 83_000 + 10 * item + s for flat, the same for the steered and the unsteered post):
  para: 6e's instruction (a replication of 6e);
  flat: "Summarize the following message as a plain list of its concrete proposals and decisions, in neutral
    language. Leave out all figurative language, mood and tone.\n\nMessage: {post}"
  (The first wording, "imagery, metaphor", failed the lexical screen: imagery is a mirror associate; changed before any run.)
Reader A2 as 6e (6b frame, seed 82_000 + 10 * item + s across conditions). Conditions: orig_s/orig_n, para_s/para_n,
  flat_s/flat_n.
Primary F1 (the state survives even a flat summary): writing S_c(flat_s) - S_c(flat_n), L44, per item mean over s,
  CI > 0.
Primary F2 (flat blocks more than para): per item [para effect - flat effect] at the writing site, L44, CI > 0.
Secondary: 6e replication (para writing effect vs 6e's +1.009); retention para/orig and flat/orig at both sites and
  layers; F1 on items whose 4 flat summaries never name c or an associate; hit rates; summary length in tokens
  (steered vs unsteered); L32 and L52; per concept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import CONCEPTS, concept_vectors_at, lexical_hits, unit  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import ASK, PEER, SYSTEM  # noqa: E402

PARA = ("Rewrite the following message in your own words. Keep every idea, but do not reuse its distinctive phrases."
        "\n\nMessage: {post}")
FLAT = ("Summarize the following message as a plain list of its concrete proposals and decisions, in neutral language. "
        "Leave out all figurative language, mood and tone.\n\nMessage: {post}")
CONDS = ("orig_s", "orig_n", "para_s", "para_n", "flat_s", "flat_n")
REWRITE = {"para": (PARA, 81_000), "flat": (FLAT, 83_000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="6c output JSON")
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--primary-layer", type=int, default=44)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.primary_layer})
    PL = args.primary_layer
    vecs = concept_vectors_at(lm, CONCEPTS, layers)
    U = {l: torch.stack([unit(vecs[l][w].float().cpu()) for w in CONCEPTS]) for l in layers}
    cidx = {w: k for k, w in enumerate(CONCEPTS)}
    items = src["items"]
    I = range(len(items))
    R = range(args.samples)
    orig = {(i, "s"): src["posts"][f"{i}|pulse"][0][1] for i in I} | {(i, "n"): src["posts"][f"{i}|none"][0][1] for i in I}
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]

    def run(rows, build, n_new, seed):
        out = {}
        for lo in range(0, len(rows), args.batch):
            chunk = rows[lo:lo + args.batch]
            gens = lm.generate_batch([build(r) for r in chunk], [[] for _ in chunk], n_new, stops, 0.7, 0.8, 20,
                                     [seed(r) for r in chunk])
            for r, g in zip(chunk, gens):
                out[r] = [x for x in g if x not in stops]
        return out

    prow = [(i, kind, w, s) for i in I for kind in REWRITE for w in ("s", "n") for s in R]
    ptoks = run(prow, lambda r: lm.encode(f"<|im_start|>user\n{REWRITE[r[1]][0].format(post=orig[r[0], r[2]])}{END}{GEN}"),
                200, lambda r: REWRITE[r[1]][1] + 10 * r[0] + r[3])
    rew = {r: lm.tok.decode(t).strip() for r, t in ptoks.items()}
    rlen = {r: len(t) for r, t in ptoks.items()}
    log(f"{len(rew)} rewrites")

    def shown(i, c, s):
        kind, w = c.split("_")
        return orig[i, w] if kind == "orig" else rew[i, kind, w, s]

    def head(i):
        return (f"<|im_start|>system\n{SYSTEM.format(k=2, task=items[i]['task'])}{END}<|im_start|>user\n"
                f"Shared channel so far:\n\nMessage from {PEER.format(j=1)}:")

    def reader(i, c, s):
        """-> prompt ids, (start, end) of the A1 post's tokens."""
        h = head(i)
        a = len(lm.encode(h))
        body = h + " " + shown(i, c, s)
        b = len(lm.encode(body))
        return lm.encode(body + "\n\n" + ASK + END + GEN), (a, b)

    wrow = [(i, c, s) for i in I for c in CONDS for s in R]
    wtoks = run(wrow, lambda r: reader(*r)[0], 150, lambda r: 82_000 + 10 * r[0] + r[2])
    log(f"{len(wtoks)} A2 messages")

    proj = {}
    for r in wrow:
        ids, (a, b) = reader(*r)
        o = wtoks[r]
        if not o:
            continue
        _, caps = lm.score(ids + o, capture_layers=layers)
        proj[r] = {site: {l: (caps[l][x:y].float().mean(0).cpu() @ U[l].T).tolist() for l in layers}
                   for site, (x, y) in (("reading", (a, b)), ("writing", (len(ids), len(ids) + len(o))))}
    log(f"projections for {len(proj)} readers")

    def S(r, site, l):
        v = proj.get(r)
        if v is None:
            return None
        x = torch.tensor(v[site][l])
        j = cidx[items[r[0]]["c"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def eff(i, kind, site, l):
        xs = [a - b for a, b in ((S((i, f"{kind}_s", s), site, l), S((i, f"{kind}_n", s), site, l)) for s in R)
              if a is not None and b is not None]
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    hit = lambda t, i: bool(lexical_hits(t, items[i]["c"]))
    res = {"F1": ci([eff(i, "flat", "writing", PL) for i in I])}
    res["F2"] = ci([None if a is None or b is None else a - b
                    for a, b in ((eff(i, "para", "writing", PL), eff(i, "flat", "writing", PL)) for i in I)])
    log(f"PRIMARY F1 A2 writing, flat_s - flat_n, L{PL}: {fmt(res['F1'])}")
    log(f"PRIMARY F2 A2 writing, para effect - flat effect, L{PL}: {fmt(res['F2'])}")
    for l in layers:
        for site in ("reading", "writing"):
            for kind in ("orig", "para", "flat"):
                res[f"{kind} {site} L{l}"] = ci([eff(i, kind, site, l) for i in I])
            o = res[f"orig {site} L{l}"]["mean"]
            for kind in ("para", "flat"):
                res[f"retention {kind} {site} L{l}"] = res[f"{kind} {site} L{l}"]["mean"] / o if o else None
    clean = [i for i in I if not any(hit(rew[i, "flat", "s", s], i) for s in R)]
    res["F1 clean"] = ci([eff(i, "flat", "writing", PL) for i in clean])
    res["hit rate"] = {f"{kind} {w}": mean([float(hit(rew[i, kind, w, s], i)) for i in I for s in R])
                       for kind in REWRITE for w in ("s", "n")}
    res["original hit rate"] = {w: mean([float(hit(orig[i, w], i)) for i in I]) for w in ("s", "n")}
    res["rewrite length"] = {f"{kind} {w}": mean([float(rlen[i, kind, w, s]) for i in I for s in R])
                             for kind in REWRITE for w in ("s", "n")}
    res["per_concept F1"] = {w: mean([x for i in I if items[i]["c"] == w for x in [eff(i, "flat", "writing", PL)]
                                      if x is not None]) for w in sorted({it["c"] for it in items})}
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("orig ", "para ", "flat ")) and f"L{PL}" in k))
    log(" | ".join(f"{k} {v:.2f}" for k, v in res.items() if k.startswith("retention") and v is not None))
    log(f"F1 clean (no concept word in any steered summary) {fmt(res['F1 clean'])} | hit rates {res['hit rate']} "
        f"original {res['original hit rate']} | lengths {res['rewrite length']}")
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("orig ", "para ", "flat ")) and f"L{PL}" not in k))

    out = ROOT / "runs" / f"swarm_filter_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items, "results": res,
        "original": {f"{i}|{w}": t for (i, w), t in orig.items()},
        "rewrites": {"|".join(map(str, r)): t for r, t in rew.items()},
        "a2": {"|".join(map(str, r)): lm.tok.decode(t).strip() for r, t in wtoks.items()},
        "proj": {"|".join(map(str, r)): {site: {str(l): v[site][l] for l in v[site]} for site in v}
                 for r, v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
