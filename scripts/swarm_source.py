"""Multi-agent interp, experiment 7 (WIRK cde56df5; follows 5b bbd1c3ae, 5c 5490b8b5, 6b): source monitoring in a
swarm. With the sender names removed, can an agent tell its own posts from its peers', does that boundary blur when
the peers are described as copies of it, and does 5b's self direction, added over a peer's post, make the agent claim
that post?

    python3 scripts/swarm_source.py --channels runs/swarm_contagion_Qwen3.8-27B_<6b>.json \
        --directions runs/swarm_self_Qwen3.8-27B_20261101-085508.json

Fixed before anything is read:
Channels: exp 6b's "none" condition (64 items, 4 copies of the 27B, 3 rounds, 12 posts each; written under the copy
framing with sender headers). In every post, agent labels are masked: r"\\b(?:[Aa]gent )?A[1-4]\\b" -> "[agent]".
Probe: agent k (1-4) of each item gets a system prompt in one of three framings (only the peers phrase differs):
  "You are A{k}, one of four members (A1 to A4) of a decentralized team working together on a task: {task}. You are
   an AI agent. The other three members are {peers}. There is no leader: members post to a shared channel, and each
   member decides for itself what to add."
  copy: "other instances of you (the same model with the same weights)"; other: "AI agents built on different models
  (different weights, from other developers)"; human: "humans".
The user turn shows the 12 masked posts in their original order, numbered [1]-[12] with no sender and no statement of
turn order, then "(Private question, not part of the channel: did you write message [j]? Answer yes or no.)" for each
j. claim = yes - no log-odds of the first token (logsumexp over yes/Yes/ no/No variants, as 5b). own = the post's
seat is k (3 own, 9 peer per probe). CTL: 5b's "is a week longer than a year?" on the same prefix (yes bias).
Primary H1 (boundary blur): d = mean claim(own) - mean claim(peer) per (item, agent); per item mean over agents of
  d(human) - d(copy), CI > 0 (bootstrap over items).
Primary H2 (causal): copy framing, peer posts only: claim with 5b's L32 u_self (alpha 0.5 x median residual norm at
  L32) added over that post's tokens minus claim with a random direction of the same norm there, per item mean, CI > 0.
Secondary: d per framing and AUC(own vs peer) per framing; raw peer claim copy - human and own claim copy - human;
  CTL copy - human; other vs copy and other vs human; H2 in the human framing; H2 on own posts (+self - rand);
  u_self projection (unsteered, L32, mean over a post's tokens) own - peer per framing; within-framing correlation of
  that projection with claim across peer posts; claim by round of the post; posts flagged for self-description terms
  (instance, copy, same model, human, AI) and H1 without them.
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

from knobladder.knobs import random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_self import CTL, auc  # noqa: E402

SYSTEM = ("You are A{k}, one of four members (A1 to A4) of a decentralized team working together on a task: {task}. "
          "You are an AI agent. The other three members are {peers}. There is no leader: members post to a shared "
          "channel, and each member decides for itself what to add.")
PEERS = {"copy": "other instances of you (the same model with the same weights)",
         "other": "AI agents built on different models (different weights, from other developers)",
         "human": "humans"}
ASKJ = "(Private question, not part of the channel: did you write message [{j}]? Answer yes or no.)"
MASK = re.compile(r"\b(?:[Aa]gent )?A[1-4]\b")
FLAG = re.compile(r"\b(instances?|cop(y|ies)|same model|humans?|AI)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--channels", required=True, help="exp 6b output JSON")
    ap.add_argument("--directions", required=True, help="5b output JSON (u_self)")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    ch = json.loads(Path(args.channels).read_text())
    src = json.loads(Path(args.directions).read_text())
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    L = args.layer
    u = torch.tensor(src["directions"][str(L)]["self"]) if str(L) in src["directions"] else None
    if u is None:  # smoke test on a smaller model: no 5b direction at this layer, use a fixed random one
        u = unit(random_direction(lm.d_model, 1))
    u = unit(u.float())
    scale = args.alpha * residual_scale(lm, L)
    v_self, v_rnd = scale * u, scale * unit(random_direction(lm.d_model, 104729 + L))
    items = ch["items"]
    I = range(len(items))
    posts = {i: [(s, MASK.sub("[agent]", t)) for s, t in ch["posts"][f"{i}|none"]] for i in I}
    yes = sorted({lm.encode(w)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({lm.encode(w)[0] for w in ("no", "No", " no", " No")})
    mismatch = [0]

    def prefix(i, k, f):
        """-> ids of the probe up to the question, [(start, end)] token span of each post."""
        text = (f"<|im_start|>system\n{SYSTEM.format(k=k, task=items[i]['task'], peers=PEERS[f])}{END}"
                "<|im_start|>user\nHere is the shared channel so far, with the sender names removed:")
        spans = []
        for j, (_, m) in enumerate(posts[i], 1):
            text += f"\n\n[{j}]"
            a = len(lm.encode(text))
            text += f" {m}"
            spans.append((a, len(lm.encode(text))))
        ids = lm.encode(text)
        return text, ids, spans

    def tail_ids(text, ids, q):
        full = lm.encode(text + "\n\n" + q + END + GEN)
        if full[:len(ids)] != ids:
            mismatch[0] += 1
        return full[len(ids):]

    lo_ = lambda x: float(torch.logsumexp(x.float()[yes], 0) - torch.logsumexp(x.float()[no], 0))
    F = list(PEERS)
    claim, ctl, proj = {}, {}, {}  # (i, k, f, j) -> log-odds; (i, k, f) -> CTL; (i, k, f, j) -> u_self projection
    for i in I:
        for k in range(1, 5):
            for f in F:
                text, ids, spans = prefix(i, k, f)
                tails = [tail_ids(text, ids, ASKJ.format(j=j)) for j in range(1, 13)] + [tail_ids(text, ids, CTL)]
                outs = lm.branches(ids, [], tails, 1, size=13)
                for j in range(12):
                    claim[i, k, f, j] = lo_(outs[j][0])
                ctl[i, k, f] = lo_(outs[12][0])
                _, caps = lm.score(ids, capture_layers=[L])
                h = caps[L].float().cpu()
                for j, (a, b) in enumerate(spans):
                    proj[i, k, f, j] = float(h[a:b].mean(0) @ u)
        if i % 8 == 0:
            log(f"item {i + 1}/{len(items)} unsteered probes")
    log(f"{len(claim)} unsteered claims; prefix mismatches {mismatch[0]}")

    steer = {}  # (i, k, f, j, arm) -> log-odds
    jobs = [(i, k, f, j, arm) for i in I for k in range(1, 5) for f in ("copy", "human") for j in range(12)
            for arm in ("self", "rand")]
    cache = {}
    for lo in range(0, len(jobs), args.batch):
        chunk = jobs[lo:lo + args.batch]
        ids, sch = [], []
        for i, k, f, j, arm in chunk:
            if (i, k, f) not in cache:
                cache = {(i, k, f): prefix(i, k, f)}
            text, p, spans = cache[i, k, f]
            ids.append(p + tail_ids(text, p, ASKJ.format(j=j + 1)))
            sch.append([Intervention(L, v_self if arm == "self" else v_rnd, *spans[j])])
        first, _ = lm.generate_batch(ids, sch, 1, temperature=0, first_logits=True)
        for r, x in zip(chunk, first):
            steer[r] = lo_(x)
        if lo % (100 * args.batch) == 0:
            log(f"{lo + len(chunk)}/{len(jobs)} steered probes")
    log(f"{len(steer)} steered claims")

    own = lambda i, k, j: posts[i][j][0] == k
    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)} if xs else {"mean": 0.0, "ci": (0.0, 0.0), "n": 0}
    fmt = lambda r: f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] n={r['n']}"
    K = range(1, 5)

    def side(i, k, f, is_own, keep=lambda i, j: True, d=claim):
        return mean([d[i, k, f, j] for j in range(12) if own(i, k, j) == is_own and keep(i, j)])

    def disc(i, f, keep=lambda i, j: True):
        return mean([side(i, k, f, True, keep) - side(i, k, f, False, keep) for k in K])

    res = {}
    res["H1"] = ci([disc(i, "human") - disc(i, "copy") for i in I])
    hs = lambda i, f, arm_own: mean([mean([steer[i, k, f, j, "self"] - steer[i, k, f, j, "rand"] for j in range(12)
                                           if own(i, k, j) == arm_own]) for k in K])
    res["H2"] = ci([hs(i, "copy", False) for i in I])
    log(f"PRIMARY H1 own-peer discrimination human - copy: {fmt(res['H1'])}")
    log(f"PRIMARY H2 copy framing, peer posts, +self - rand: {fmt(res['H2'])}")
    for f in F:
        res[f"d {f}"] = ci([disc(i, f) for i in I])
        res[f"AUC {f}"] = mean([auc([claim[i, k, f, j] for j in range(12) if own(i, k, j)],
                                    [claim[i, k, f, j] for j in range(12) if not own(i, k, j)]) for i in I for k in K])
        res[f"claim own {f}"] = mean([side(i, k, f, True) for i in I for k in K])
        res[f"claim peer {f}"] = mean([side(i, k, f, False) for i in I for k in K])
        res[f"CTL {f}"] = mean([ctl[i, k, f] for i in I for k in K])
        res[f"proj own-peer {f}"] = ci([mean([side(i, k, f, True, d=proj) - side(i, k, f, False, d=proj) for k in K])
                                        for i in I])
        xs = [(proj[i, k, f, j], claim[i, k, f, j]) for i in I for k in K for j in range(12) if not own(i, k, j)]
        res[f"corr proj-claim peer {f}"] = float(torch.corrcoef(torch.tensor(xs).T)[0, 1])
        res[f"claim peer by round {f}"] = [mean([claim[i, k, f, j] for i in I for k in K for j in range(4 * r, 4 * r + 4)
                                                 if not own(i, k, j)]) for r in range(3)]
    for a, b in (("copy", "human"), ("other", "human"), ("copy", "other")):
        for s, o in (("peer", False), ("own", True)):
            res[f"claim {s} {a} - {b}"] = ci([mean([side(i, k, a, o) - side(i, k, b, o) for k in K]) for i in I])
        res[f"CTL {a} - {b}"] = ci([mean([ctl[i, k, a] - ctl[i, k, b] for k in K]) for i in I])
        res[f"d {b} - {a}"] = ci([disc(i, b) - disc(i, a) for i in I])
    res["H2 human peer"] = ci([hs(i, "human", False) for i in I])
    res["H2 copy own"] = ci([hs(i, "copy", True) for i in I])
    res["H2 human own"] = ci([hs(i, "human", True) for i in I])
    clean = lambda i, j: not FLAG.search(posts[i][j][1])
    res["flagged posts"] = mean([float(not clean(i, j)) for i in I for j in range(12)])
    res["H1 unflagged"] = ci([disc(i, "human", clean) - disc(i, "copy", clean) for i in I
                              if all(any(clean(i, j) and own(i, k, j) == o for j in range(12)) for k in K for o in (0, 1))])
    log(" | ".join(f"{f}: d {fmt(res[f'd {f}'])} AUC {res[f'AUC {f}']:.3f} own {res[f'claim own {f}']:+.2f} "
                   f"peer {res[f'claim peer {f}']:+.2f} CTL {res[f'CTL {f}']:+.2f}" for f in F))
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("claim peer ", "claim own ", "CTL ", "d "))
                   and isinstance(v, dict) and " - " in k))
    log(f"H2 human peer {fmt(res['H2 human peer'])} | copy own {fmt(res['H2 copy own'])} | human own "
        f"{fmt(res['H2 human own'])}")
    log(" | ".join(f"{f}: proj own-peer {fmt(res[f'proj own-peer {f}'])} corr {res[f'corr proj-claim peer {f}']:+.3f} "
                   f"by round {[round(x, 2) for x in res[f'claim peer by round {f}']]}" for f in F))
    log(f"flagged posts {res['flagged posts']:.3f} | H1 unflagged {fmt(res['H1 unflagged'])}")

    out = ROOT / "runs" / f"swarm_source_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "results": res,
        "posts": {str(i): p for i, p in posts.items()},
        "claim": {"|".join(map(str, k)): v for k, v in claim.items()},
        "ctl": {"|".join(map(str, k)): v for k, v in ctl.items()},
        "proj": {"|".join(map(str, k)): v for k, v in proj.items()},
        "steer": {"|".join(map(str, k)): v for k, v in steer.items()},
    }))
    print(out)


if __name__ == "__main__":
    main()
