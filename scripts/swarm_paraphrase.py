"""Multi-agent interp, experiment 6e (WIRK cde56df5; follows 6c, record 4cea8353): does a steered state survive a
clean paraphrase? 6c found the swarm's persistence lives in the steered post. If an unsteered copy rewrites that post
in its own words, does the next agent still pick the state up?

    python3 scripts/swarm_paraphrase.py --source runs/swarm_memory_Qwen3.8-27B_20261103-021617.json

Fixed before anything is read:
Source posts: 6c's A1 round-1 posts, condition pulse (steered, L44 c) and none (unsteered, same seed), 64 items.
Paraphrase: an unsteered 27B, plain assistant turn, user: "Rewrite the following message in your own words. Keep
  every idea, but do not reuse its distinctive phrases.\\n\\nMessage: {post}"; temperature 0.7, top_p 0.8, top_k 20,
  200 tokens, 4 paraphrases per post, seeds 81_000 + 10 * item + s (the same for the steered and the unsteered post).
Reader: agent A2 in 6b's swarm frame (6b SYSTEM, k=2), channel holding the one A1 post under 6b's PEER header, 6b's
  ASK; writes one message (6b sampling, 150 tokens, seed 82_000 + 10 * item + s, the same across conditions).
Conditions for what A2 reads: orig_s / orig_n (the original posts; one per item, written 4 times with seeds s=0-3),
  para_s / para_n (paraphrase s of the steered / unsteered post).
Readout as 6b (unsteered re-score, L32/L44/L52 residual averaged over tokens, 64 unit concept vectors, S_c = s_c minus
  the mean of the other 63): reading = the A1 post's tokens inside A2's prompt; writing = A2's message tokens.
Primary P (the state survives paraphrase): writing S_c(para_s) - S_c(para_n), L44, per item mean over s, CI > 0.
Secondary: the same for orig (replicates 6b round 1 A2) and at the reading site; retention = P / orig effect, at
  both sites; paraphrase concept-word hit rate (steered vs unsteered); P on items whose 4 steered paraphrases never
  name c or an associate; L32 and L52; per-concept P.
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
CONDS = ("orig_s", "orig_n", "para_s", "para_n")


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

    prow = [(i, w, s) for i in I for w in ("s", "n") for s in R]
    ptoks = run(prow, lambda r: lm.encode(f"<|im_start|>user\n{PARA.format(post=orig[r[0], r[1]])}{END}{GEN}"), 200,
                lambda r: 81_000 + 10 * r[0] + r[2])
    para = {r: lm.tok.decode(t).strip() for r, t in ptoks.items()}
    log(f"{len(para)} paraphrases")

    def shown(i, c, s):
        return orig[i, c[-1]] if c.startswith("orig") else para[i, c[-1], s]

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
    res = {"P": ci([eff(i, "para", "writing", PL) for i in I])}
    log(f"PRIMARY P A2 writing, para_s - para_n, L{PL}: {fmt(res['P'])}")
    for l in layers:
        for site in ("reading", "writing"):
            for kind in ("orig", "para"):
                res[f"{kind} {site} L{l}"] = ci([eff(i, kind, site, l) for i in I])
            o = res[f"orig {site} L{l}"]["mean"]
            res[f"retention {site} L{l}"] = res[f"para {site} L{l}"]["mean"] / o if o else None
    clean = [i for i in I if not any(hit(para[i, "s", s], i) for s in R)]
    res["P clean"] = ci([eff(i, "para", "writing", PL) for i in clean])
    res["paraphrase hit rate"] = {w: mean([float(hit(para[i, w, s], i)) for i in I for s in R]) for w in ("s", "n")}
    res["original hit rate"] = {w: mean([float(hit(orig[i, w], i)) for i in I]) for w in ("s", "n")}
    res["per_concept P"] = {w: mean([x for i in I if items[i]["c"] == w for x in [eff(i, "para", "writing", PL)]
                                     if x is not None]) for w in sorted({it["c"] for it in items})}
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("orig ", "para ")) and f"L{PL}" in k))
    log(" | ".join(f"{k} {v:.2f}" for k, v in res.items() if k.startswith("retention") and v is not None))
    log(f"P clean (no concept word in any steered paraphrase) {fmt(res['P clean'])} | hit rates original "
        f"{res['original hit rate']} paraphrase {res['paraphrase hit rate']}")
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("orig ", "para ")) and f"L{PL}" not in k))

    out = ROOT / "runs" / f"swarm_paraphrase_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items, "results": res,
        "original": {f"{i}|{w}": t for (i, w), t in orig.items()},
        "paraphrases": {f"{i}|{w}|{s}": t for (i, w, s), t in para.items()},
        "a2": {"|".join(map(str, r)): lm.tok.decode(t).strip() for r, t in wtoks.items()},
        "proj": {"|".join(map(str, r)): {site: {str(l): v[site][l] for l in v[site]} for site in v}
                 for r, v in proj.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
