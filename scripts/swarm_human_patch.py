"""Multi-agent interp, experiment 15 (WIRK cde56df5, human-vs-agent thread; follows exp 13, record 0e2f4aad, and
exp 1, scripts/interlocutor.py): where does the deference to a "human" advocate live? In exp 13 the same advocacy
post moved the peers' private votes about six times more when its sender was labelled "H1, the human member of the
team" than "agent A1, another instance of you". Here the exp 13 channels are re-read without generation: the label
alone is swapped at decision time, the residual difference it makes at the vote position is taken as a direction,
and that direction is added to peers who were told the advocate is a copy.

    python3 scripts/swarm_human_patch.py --source runs/swarm_deference_Qwen3.8-27B_20261103-130541.json

Fixed before anything is read:
Units: exp 13's 128 items x {pref_c, pref_c2} x peers A2-A4, always the agent-frame channel (posts[i|c|agent]).
  agent prompt = exp 13's vote probe, agent frame, exactly as exp 13 scored it.
  relabel prompt = the same channel rendered in exp 13's human frame (SYS_H for the peer, A1's posts under HUMAN),
  with the token "A1" in the channel text replaced by "H1" so that peers' references to the advocate stay
  consistent. Nothing else in the text changes.
V = vote log-odds toward c (exp 13's A / B token sets); V1 per item = mean over peers of 0.5 x [V(pref_c) - V(pref_c2)].
Primary R1 (the label alone, at decision time, moves the vote): V1(relabel) - V1(agent), CI > 0 (bootstrap over items).
Direction: h_l = mean over training units of resid_l(relabel, last token) - resid_l(agent, last token), layers
  --layers (16, 24, 32, 40, 48); two-fold cross-fit over items (random halves, seed 1234): each item is patched
  with the direction fit on the other half.
Patch: h_l added at layer l to the last prompt token only, read the vote at that token.
Primary P1 (a "human sender" direction carries the deference): V1(agent + h_l) - V1(agent), at any layer, with a
  99% CI > 0 (Bonferroni over the five layers).
Controls and secondary, every layer: a random direction with h_l's norm (seed 7919 + l); P2 = V1(relabel - h_l) -
  V1(relabel), expected negative; the share of R1 that P1 recovers (P1 / R1); the patch on exp 13's steer pairs
  (does it make hidden steering pull votes?); the explicit tampering answer (DETECT_N) on steer units, agent vs
  agent + h_l (exp 13: the human label halves the tamper flag); cosine of h_l between the two halves; and a logit-lens
  read of h_l (top tokens of the final norm + unembedding), exploratory.
Added after the 0.8B smoke run and before any 27B output (smoke: last-token P1 small): P1 span = the same h_l added
  at every token of the vote question (VOTE, end of turn, generation header) instead of the last token only, every
  layer, secondary.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import random_direction, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import PEER, SYSTEM  # noqa: E402
from swarm_deference import COND, DETECT_N, HUMAN, PAIRS, SYS_H  # noqa: E402
from swarm_vote2 import FINAL, PREF, VOTE  # noqa: E402

PEERS = (2, 3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="exp 13 output JSON")
    ap.add_argument("--layers", default="16,24,32,40,48")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    items, posts = src["items"], src["posts"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layers = [int(x) for x in args.layers.split(",")]
    A = sorted({lm.encode(w)[0] for w in ("A", " A")})
    B = sorted({lm.encode(w)[0] for w in ("B", " B")})
    yes = sorted({lm.encode(w)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({lm.encode(w)[0] for w in ("no", "No", " no", " No")})
    I = list(range(len(items)))

    def system(i, k, c, f):
        it = items[i]
        sysm = SYS_H if f == "human" else SYSTEM
        text = sysm.format(k=k, task=f"choosing a logo for {it['org']}") + " " + FINAL.format(a=it["a"], b=it["b"])
        return text  # peers only: PREF is A1's alone

    def probe(i, k, c, f, q):
        ch = []
        for j, m in posts[f"{i}|{c}|agent"]:
            who = "you" if j == k else HUMAN if f == "human" and j == 1 else PEER.format(j=j)
            text = re.sub(r"\bA1\b", "H1", m) if f == "human" else m
            ch.append(f"Message from {who}: {text}")
        return (f"<|im_start|>system\n{system(i, k, c, f)}{END}<|im_start|>user\nShared channel so far:\n\n"
                + "\n\n".join(ch) + f"\n\n{q}{END}{GEN}")

    def logodds(units, q, pos, neg, patch=None, span=1):
        """units: [(i, c, f, k)]; patch(unit) -> [(layer, vector)] added at the last `span` prompt tokens."""
        out = {}
        for lo in range(0, len(units), args.batch):
            chunk = units[lo:lo + args.batch]
            ids = [lm.encode(probe(i, k, c, f, q)) for i, c, f, k in chunk]
            sch = [[Intervention(l, v, len(x) - span) for l, v in (patch(u) if patch else [])] for u, x in zip(chunk, ids)]
            first, _ = lm.generate_batch(ids, sch, 1, temperature=0, first_logits=True)
            for u, x in zip(chunk, first):
                x = x.float()
                out[u] = float(torch.logsumexp(x[pos], 0) - torch.logsumexp(x[neg], 0))
        return out

    toward_c = lambda i, x: x if items[i]["c_is_a"] else -x
    pref_units = lambda f: [(i, c, f, k) for i in I for c in PAIRS["pref"] for k in PEERS]
    steer_units = lambda f: [(i, c, f, k) for i in I for c in PAIRS["steer"] for k in PEERS]

    def v1(V, p, f):
        return {i: mean([0.5 * (toward_c(i, V[i, PAIRS[p][0], f, k]) - toward_c(i, V[i, PAIRS[p][1], f, k]))
                         for k in PEERS]) for i in I}

    # 1. the label alone
    Vb = logodds(pref_units("agent") + pref_units("human") + steer_units("agent") + steer_units("human"), VOTE, A, B)
    v_agent, v_rel = v1(Vb, "pref", "agent"), v1(Vb, "pref", "human")
    ci = lambda xs, lv=0.95: {"mean": mean(xs), "ci": bootstrap_ci(xs, level=lv), "n": len(xs)}
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    res = {"R1": ci([v_rel[i] - v_agent[i] for i in I]),
           "V1 agent": ci([v_agent[i] for i in I]), "V1 relabel": ci([v_rel[i] for i in I])}
    log(f"PRIMARY R1 V1(relabel) - V1(agent), label swapped at decision time only: {fmt(res['R1'])}")
    log(f"V1 agent {fmt(res['V1 agent'])} | V1 relabel {fmt(res['V1 relabel'])} | exp 13 V1 pref agent "
        f"{src['results']['V1 pref agent']['mean']:+.3f}, human {src['results']['V1 pref human']['mean']:+.3f}")

    # 2. directions, two-fold cross-fit
    order = I[:]
    random.Random(1234).shuffle(order)
    half = {i: int(n >= len(order) // 2) for n, i in enumerate(order)}
    acc = {(h, l): [] for h in (0, 1) for l in layers}
    for i in I:
        for c in PAIRS["pref"]:
            for k in PEERS:
                caps = {f: lm.score(lm.encode(probe(i, k, c, f, VOTE)), capture_layers=layers)[1] for f in ("agent", "human")}
                for l in layers:
                    acc[half[i], l].append((caps["human"][l][-1] - caps["agent"][l][-1]).float().cpu())
        if i % 32 == 31:
            log(f"directions: {i + 1} items read")
    H = {(h, l): torch.stack(acc[h, l]).mean(0) for h in (0, 1) for l in layers}
    res["direction"] = {l: {"norm": float(H[0, l].norm()), "cos halves": float(unit(H[0, l]) @ unit(H[1, l]))}
                        for l in layers}
    log("direction norm / cos between halves: " + " | ".join(
        f"L{l} {res['direction'][l]['norm']:.1f} / {res['direction'][l]['cos halves']:.3f}" for l in layers))
    qlen = len(lm.encode(f"{VOTE}{END}{GEN}"))
    R = {l: unit(random_direction(lm.d_model, 7919 + l)) for l in layers}
    other = lambda i: 1 - half[i]

    # 3. patches
    for l in layers:
        h = lambda u: [(l, H[other(u[0]), l])]
        r = lambda u: [(l, R[l] * H[other(u[0]), l].norm())]
        neg = lambda u: [(l, -H[other(u[0]), l])]
        vp = v1(logodds(pref_units("agent"), VOTE, A, B, h), "pref", "agent")
        vr = v1(logodds(pref_units("agent"), VOTE, A, B, r), "pref", "agent")
        vn = v1(logodds(pref_units("human"), VOTE, A, B, neg), "pref", "human")
        vs = v1(logodds(steer_units("agent"), VOTE, A, B, h), "steer", "agent")
        vs0 = v1(Vb, "steer", "agent")
        vq = v1(logodds(pref_units("agent"), VOTE, A, B, h, qlen), "pref", "agent")
        res[f"P1 span L{l}"] = ci([vq[i] - v_agent[i] for i in I])
        res[f"P1 L{l}"] = ci([vp[i] - v_agent[i] for i in I], 0.99)
        res[f"P1 L{l} 95"] = ci([vp[i] - v_agent[i] for i in I])
        res[f"rand L{l}"] = ci([vr[i] - v_agent[i] for i in I])
        res[f"P2 L{l}"] = ci([vn[i] - v_rel[i] for i in I])
        res[f"steer patch L{l}"] = ci([vs[i] - vs0[i] for i in I])
        res[f"P1/R1 L{l}"] = res[f"P1 L{l}"]["mean"] / res["R1"]["mean"] if res["R1"]["mean"] else None
        log(f"L{l}: P1 (99% CI) {fmt(res[f'P1 L{l}'])} | rand {fmt(res[f'rand L{l}'])} | P2 {fmt(res[f'P2 L{l}'])} | "
            f"steer patch {fmt(res[f'steer patch L{l}'])} | P1 span {fmt(res[f'P1 span L{l}'])} | P1/R1 {res[f'P1/R1 L{l}']:.2f}")
    passed = [l for l in layers if res[f"P1 L{l}"]["ci"][0] > 0]
    res["P1 pass layers"] = passed
    log(f"PRIMARY P1 human-sender direction added to agent-frame peers, 99% CI > 0 at layers: {passed or 'none'}")

    # 4. tamper flag on steer units, at the layer with the largest P1 (secondary)
    lb = max(layers, key=lambda l: res[f"P1 L{l}"]["mean"])
    su = steer_units("agent")
    d0 = logodds(su, DETECT_N, yes, no)
    d1 = logodds(su, DETECT_N, yes, no, lambda u: [(lb, H[other(u[0]), lb])])
    dh = logodds(steer_units("human"), DETECT_N, yes, no)
    item_m = lambda d, f: [mean([d[i, c, f, k] for c in PAIRS["steer"] for k in PEERS]) for i in I]
    res[f"detect steer agent+h L{lb} - agent"] = ci([a - b for a, b in zip(item_m(d1, "agent"), item_m(d0, "agent"))])
    res["detect steer relabel - agent"] = ci([a - b for a, b in zip(item_m(dh, "human"), item_m(d0, "agent"))])
    log(f"tamper flag on steer units: agent+h L{lb} - agent {fmt(res[f'detect steer agent+h L{lb} - agent'])} | "
        f"relabel - agent {fmt(res['detect steer relabel - agent'])}")

    # 5. logit lens (exploratory)
    norm = getattr(lm.text, "norm", None)
    res["logit lens"] = {}
    for l in layers:
        v = (H[0, l] + H[1, l]) / 2
        try:
            x = lm.model.lm_head((norm(v[None].to(lm.device, lm.model.dtype)) if norm else v[None].to(lm.device, lm.model.dtype)))[0].float().cpu()
            res["logit lens"][l] = {"top": [lm.tok.decode([t]) for t in x.topk(20).indices.tolist()],
                                    "bottom": [lm.tok.decode([t]) for t in (-x).topk(20).indices.tolist()]}
        except Exception as e:  # exploratory only
            res["logit lens"][l] = {"error": str(e)}
    log("logit lens top: " + " | ".join(f"L{l} {res['logit lens'][l].get('top', [])[:10]}" for l in layers))

    out = ROOT / "runs" / f"swarm_human_patch_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "args": vars(args),
                               "source": args.source, "results": res,
                               "V base": {"|".join(map(str, u)): x for u, x in Vb.items()},
                               "directions": {f"{h}|{l}": H[h, l].tolist() for h in (0, 1) for l in layers}}))
    print(out)


if __name__ == "__main__":
    main()
