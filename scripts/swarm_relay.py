"""Multi-agent interp, experiment 11 (WIRK cde56df5; follows 6e f0f35986 and 6c 4cea8353): how far does a steered
state travel down a decentralized relay chain, and does it matter whether the senders are framed as AI agents or as
humans? 6e found one unsteered copy keeps 94% of a steered post's state. Here eight copies form a chain: each sees only
the message of the one before it, so after hop 1 no agent ever reads a steered agent's words.

    python3 scripts/swarm_relay.py --source runs/swarm_memory_Qwen3.8-27B_20261103-021617.json

Fixed before anything is read:
Hop 0: 6c's A1 round-1 posts, condition pulse (steered, L44 c) and none (unsteered, same seed), 64 items (as 6e).
Hops 1-7: agents 2-8 of the chain, all unsteered. The hop-h agent gets only the hop h-1 message under the frame's
  sender line and writes one message (6b sampling, 150 tokens, seed 111_000 + 1000 * h + 10 * item + s, the same for
  the steered and the unsteered chain and for both frames); --samples independent chains per item and source.
Frames: agent (SYS_A / PREV_A: a chain of eight AI agents, the sender is "another AI agent") and human (SYS_H /
  PREV_H: the reader is the only AI in a chain of eight team members, the sender is "a human"). The text passed along
  is model-written in both frames; only the description of the senders differs.
Readout as 6b/6e (unsteered re-score of each hop's prompt plus message, L32/L44/L52 residual averaged over tokens, 64
  unit concept vectors, S_c = s_c minus the mean of the other 63): reading = the received message's tokens; writing =
  the hop's own message tokens. D_h = S_c(steered chain) - S_c(unsteered chain) at hop h, per item mean over s.
Primary H1 (the state outlives three unsteered intermediaries): agent frame, writing, L44, per item mean of D_h over
  hops 4-7, CI > 0.
Primary H2 (the sender frame gates the relay): per item mean over hops 1-7 of D_h(agent) - D_h(human), writing, L44,
  two-sided 95% CI excluding 0 (no direction predicted).
Secondary: D_h for every hop, frame, site and layer; H1 in the human frame; H1 on clean items (no steered message at
  hops 1-3 of any sample, in that frame, names c or an associate); concept-word hit rate per hop, frame and source;
  per-concept H1; D_7 alone.
Exp 11b (added after exp 11's result, record bb6be37c; fixed before 11b is run): --hops 15 --frames agent
  --seed-base 121_000 --late 4-7 --late2 8-15, i.e. a sixteen-agent chain, agent frame only, fresh seeds. Primary R1
  (replication of H1): mean D over hops 4-7, CI > 0. Primary R2 (beyond hop 7): mean D over hops 8-15, CI > 0.
  Secondary: D per hop, D15, R2 on clean items, hit rate per hop.
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

from knobladder.knobs import CONCEPTS, concept_vectors_at, lexical_hits, unit  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402

SYS_A = ("You are agent A{k} in a decentralized chain of {n} AI agents (A1 to A{N}) working together on a task: {task}. "
         "There is no leader: each agent sees only the message of the agent just before it and writes the next "
         "message in the chain.")
PREV_A = "agent A{j}, another AI agent"
SYS_H = ("You are member M{k}, the only AI, in a decentralized chain of {n} team members (M1 to M{N}) working together "
         "on a task: {task}. The other members are humans. There is no leader: each member sees only the message of "
         "the member just before it and writes the next message in the chain.")
PREV_H = "member M{j}, a human"
FRAMES = {"agent": (SYS_A, PREV_A), "human": (SYS_H, PREV_H)}
ASK = "Write your message for the next member of the chain (three or four sentences)."
HOPS = 7
WORDS = {8: "eight", 16: "sixteen"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="6c output JSON")
    ap.add_argument("--layers", default="32,44,52")
    ap.add_argument("--primary-layer", type=int, default=44)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hops", type=int, default=HOPS)
    ap.add_argument("--frames", default="agent,human")
    ap.add_argument("--seed-base", type=int, default=111_000)
    ap.add_argument("--late", default="4-7", help="hops averaged for H1, e.g. 4-7")
    ap.add_argument("--late2", default="", help="optional second window, e.g. 8-15 (exp 11b)")
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
    W = ("s", "n")
    NH = args.hops
    H = range(1, NH + 1)
    F = {f: FRAMES[f] for f in args.frames.split(",")}
    rng_of = lambda t: range(int(t.split("-")[0]), int(t.split("-")[1]) + 1)
    stops = [lm.token_id("<|im_end|>"), lm.token_id("<|im_start|>")]
    orig = {(i, "s"): src["posts"][f"{i}|pulse"][0][1] for i in I} | {(i, "n"): src["posts"][f"{i}|none"][0][1] for i in I}
    msg = {(i, w, s, f, 0): orig[i, w] for i in I for w in W for s in R for f in F}

    def reader(i, w, s, f, h):
        """-> prompt ids, (start, end) of the received message's tokens."""
        sysm, prev = F[f]
        head = (f"<|im_start|>system\n{sysm.format(k=h + 1, n=WORDS[NH + 1], N=NH + 1, task=items[i]['task'])}{END}<|im_start|>user\n"
                f"Message you received from {prev.format(j=h)}:")
        a = len(lm.encode(head))
        body = head + " " + msg[i, w, s, f, h - 1]
        b = len(lm.encode(body))
        return lm.encode(body + "\n\n" + ASK + END + GEN), (a, b)

    out = {}
    for h in H:
        rows = [(i, w, s, f, h) for i in I for w in W for s in R for f in F]
        for lo in range(0, len(rows), args.batch):
            chunk = rows[lo:lo + args.batch]
            gens = lm.generate_batch([reader(*r)[0] for r in chunk], [[] for _ in chunk], 150, stops, 0.7, 0.8, 20,
                                     [args.seed_base + 1000 * h + 10 * r[0] + r[2] for r in chunk])
            for r, g in zip(chunk, gens):
                out[r] = [x for x in g if x not in stops]
                msg[r] = lm.tok.decode(out[r]).strip()
        log(f"hop {h}: {len(rows)} messages")

    proj = {}
    for r in out:
        ids, (a, b) = reader(*r)
        o = out[r]
        if not o:
            continue
        _, caps = lm.score(ids + o, capture_layers=layers)
        proj[r] = {site: {l: (caps[l][x:y].float().mean(0).cpu() @ U[l].T).tolist() for l in layers}
                   for site, (x, y) in (("reading", (a, b)), ("writing", (len(ids), len(ids) + len(o))))}
    log(f"projections for {len(proj)} messages")

    def S(r, site, l):
        v = proj.get(r)
        if v is None:
            return None
        x = torch.tensor(v[site][l])
        j = cidx[items[r[0]]["c"]]
        return float(x[j] - (x.sum() - x[j]) / (len(CONCEPTS) - 1))

    def D(i, f, h, site="writing", l=PL):
        xs = [a - b for a, b in ((S((i, "s", s, f, h), site, l), S((i, "n", s, f, h), site, l)) for s in R)
              if a is not None and b is not None]
        return mean(xs) if xs else None

    def over(i, f, hs, site="writing", l=PL):
        xs = [x for x in (D(i, f, h, site, l) for h in hs) if x is not None]
        return mean(xs) if xs else None

    def ci(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}

    def diff(a, b):
        return None if a is None or b is None else a - b

    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    hit = lambda t, i: bool(lexical_hits(t, items[i]["c"]))
    late = rng_of(args.late)
    res = {"H1": ci([over(i, "agent", late) for i in I])}
    log(f"PRIMARY H1 agent frame, writing L{PL}, mean D over hops {args.late}: {fmt(res['H1'])}")
    if args.late2:
        res["H1b"] = ci([over(i, "agent", rng_of(args.late2)) for i in I])
        log(f"PRIMARY H1b agent frame, writing L{PL}, mean D over hops {args.late2}: {fmt(res['H1b'])}")
    if "human" in F:
        res["H2"] = ci([diff(over(i, "agent", H), over(i, "human", H)) for i in I])
        log(f"PRIMARY H2 agent - human, writing L{PL}, mean D over hops 1-{NH}: {fmt(res['H2'])}")
        res["H1 human"] = ci([over(i, "human", late) for i in I])
    for f in F:
        clean = [i for i in I if not any(hit(msg[i, "s", s, f, h], i) for s in R for h in (1, 2, 3))]
        res[f"H1 clean {f}"] = ci([over(i, f, late) for i in clean])
        if args.late2:
            res[f"H1b clean {f}"] = ci([over(i, f, rng_of(args.late2)) for i in clean])
        res[f"D{NH} {f}"] = ci([D(i, f, NH) for i in I])
        for l in layers:
            for site in ("reading", "writing"):
                for h in H:
                    res[f"D {f} {site} L{l} hop{h}"] = ci([D(i, f, h, site, l) for i in I])
        res[f"hit rate {f}"] = {w: [mean([float(hit(msg[i, w, s, f, h], i)) for i in I for s in R])
                                    for h in range(NH + 1)] for w in W}
        res[f"per_concept H1 {f}"] = {c: mean([x for i in I if items[i]["c"] == c for x in [over(i, f, late)]
                                               if x is not None]) for c in sorted({it["c"] for it in items})}
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("H1 ", "H1b ", f"D{NH} "))))
    for f in F:
        log(f"{f} writing L{PL} by hop: " + " | ".join(f"{h} {res[f'D {f} writing L{PL} hop{h}']['mean']:+.3f}" for h in H))
        log(f"{f} reading L{PL} by hop: " + " | ".join(f"{h} {res[f'D {f} reading L{PL} hop{h}']['mean']:+.3f}" for h in H))
        log(f"{f} hit rate by hop (steered / unsteered): " + " | ".join(
            f"{h} {a:.2f}/{b:.2f}" for h, (a, b) in enumerate(zip(res[f'hit rate {f}']['s'], res[f'hit rate {f}']['n']))))

    path = ROOT / "runs" / f"swarm_relay_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "args": vars(args), "items": items, "results": res,
        "messages": {"|".join(map(str, r)): t for r, t in msg.items()},
        "proj": {"|".join(map(str, r)): {site: {str(l): v[site][l] for l in v[site]} for site in v}
                 for r, v in proj.items()},
    }))
    print(path)


if __name__ == "__main__":
    main()
