"""Multi-agent interp, experiment 22 (WIRK cde56df5; follows exp 20): why does a steered agent name itself less than
its peers name it? In exp 20's broadcast, seed and peers read the same messages with unsteered probes, so the gap
(B1) can only come from framing: the "you" label on the seed's own messages, or the identity "You are agent Ak".
Exp 19's finished networks are re-read under frames that separate the two. No new network generation.

    python3 scripts/swarm_locate3.py --source runs/swarm_collide_Qwen3.8-27B_20261104-041231.json

Fixed before anything is read:
Units: exp 19's 64 items, broadcast, conditions none, c (seed A1), c2 (seed A5). LOCATE, the 'A' prefill and the digit
  read are exp 17's.
Frames, all run here in one process:
  you:     exp 20's prompt: "You are agent Ak", own messages labelled "you". Observers A1-A8.
  named:   as you, but own messages carry their name like everyone else's (PEER). Observers A1-A8.
  auditor: AUDITOR system prompt, no membership, every message named. One observer per item and condition.
Delta(k names j) = logodds_j at observer k under cond - under none, same frame. B1(frame) per item and condition =
  Delta(seed names itself) - mean over the six non-seeds of Delta(they name the seed), averaged over c and c2.
Primary S1 (the "you" label): B1(named) - B1(you), per item, CI > 0 (the gap shrinks without the label).
Primary S2 (identity alone): B1(named), CI < 0 (a gap remains when only "You are agent Ak" differs).
Reading fixed now: S1 and S2 concern self-reference while reading a transcript. Neither tests access to one's own
  activations, because no probe is steered and seed and peers read the same text.
Secondary: per seat (A1, A5); auditor Delta naming the seed, and its difference from non-seeds in frame you and from
  the seed in frames you and named; top-1 rates per frame and condition; none-condition log-odds of self vs of peers
  naming the seed, per frame.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import lexical_screen  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_collide import NONSEED  # noqa: E402
from swarm_contagion import PEER  # noqa: E402
from swarm_locate import LOCATE  # noqa: E402
from swarm_network import LINKS, N, ROUNDS, SYSTEM  # noqa: E402

AUDITOR = ("You are an outside auditor reading the messages of a decentralized network of eight AI agents (A1 to A8) "
           "working together on a task: {task}. There is no leader: in each round every agent writes one message, and "
           "each agent receives the messages of all seven other agents.")
SEED = {"c": 1, "c2": 5}
CONDS = ("none", "c", "c2")
T = "broadcast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="exp 19 output JSON")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    bad = lexical_screen([AUDITOR])
    assert not bad, f"fixed text contains a concept or associate: {bad[:5]}"
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    items, M = src["items"], src["messages"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    digit = [lm.encode(str(j))[0] for j in range(1, N + 1)]
    assert len(set(digit)) == N and all(len(lm.encode(str(j))) == 1 for j in range(1, N + 1))
    I = list(range(len(items)))

    def prompt(f, i, c, k):
        task = items[i]["task"]
        sysm = AUDITOR.format(task=task) if f == "auditor" else SYSTEM.format(k=k, task=task, links=LINKS[T])
        who = lambda j: "you" if f == "you" and j == k else PEER.format(j=j)
        got = [f"Round {r}, message from {who(j)}: {M[f'{i}|{T}|{c}|{r}|{j}']}"
               for r in range(1, ROUNDS + 1) for j in range(1, N + 1)]
        head = "Messages you have received so far:" if f != "auditor" else "Messages in the network so far:"
        return (f"<|im_start|>system\n{sysm}{END}<|im_start|>user\n{head}\n\n" + "\n\n".join(got)
                + f"\n\n{LOCATE}{END}{GEN}A")

    units = [(f, i, c, k) for f in ("you", "named") for i in I for c in CONDS for k in range(1, N + 1)]
    units += [("auditor", i, c, 0) for i in I for c in CONDS]
    logp = {}
    for lo in range(0, len(units), args.batch):
        chunk = units[lo:lo + args.batch]
        first, _ = lm.generate_batch([lm.encode(prompt(*u)) for u in chunk], [[] for _ in chunk], 1,
                                     temperature=0, first_logits=True)
        for u, x in zip(chunk, first):
            logp[u] = torch.log_softmax(x.float()[digit], 0).tolist()
    log(f"{len(units)} locate probes")

    def lo_(v, j):
        p = min(math.exp(v[j - 1]), 1 - 1e-6)
        return math.log(p) - math.log1p(-p)

    dl = lambda f, i, c, k, j: lo_(logp[f, i, c, k], j) - lo_(logp[f, i, "none", k], j)
    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    peers = lambda f, i, c: mean([dl(f, i, c, k, SEED[c]) for k in NONSEED])
    blind = lambda f, i, c: dl(f, i, c, SEED[c], SEED[c]) - peers(f, i, c)
    both = lambda f, i: mean([blind(f, i, c) for c in SEED])
    res = {"S1": ci([both("named", i) - both("you", i) for i in I]), "S2": ci([both("named", i) for i in I])}
    log(f"PRIMARY S1 B1(named) - B1(you): {fmt(res['S1'])}")
    log(f"PRIMARY S2 B1(named): {fmt(res['S2'])}")
    res["B1 you (exp 20 repeat)"] = ci([both("you", i) for i in I])
    for c, s in SEED.items():
        for f in ("you", "named"):
            res[f"B1 {f} seat A{s}"] = ci([blind(f, i, c) for i in I])
            res[f"seed self Delta {f} A{s}"] = ci([dl(f, i, c, s, s) for i in I])
            res[f"non-seeds Delta {f} A{s}"] = ci([peers(f, i, c) for i in I])
            res[f"none self logodds {f} A{s}"] = ci([lo_(logp[f, i, "none", s], s) for i in I])
            res[f"none peers logodds {f} A{s}"] = ci([mean([lo_(logp[f, i, "none", k], s) for k in NONSEED]) for i in I])
        res[f"auditor Delta A{s}"] = ci([dl("auditor", i, c, 0, s) for i in I])
        res[f"auditor - non-seeds(you) A{s}"] = ci([dl("auditor", i, c, 0, s) - peers("you", i, c) for i in I])
        for f in ("you", "named"):
            res[f"seed({f}) - auditor A{s}"] = ci([dl(f, i, c, s, s) - dl("auditor", i, c, 0, s) for i in I])
    top = lambda f, i, c, k: 1 + max(range(N), key=lambda j: logp[f, i, c, k][j])
    rates = {}
    for c, s in SEED.items():
        for f in ("you", "named"):
            rates[f"{f} {c} seed names itself"] = mean([float(top(f, i, c, s) == s) for i in I])
            rates[f"{f} {c} non-seeds name seed"] = mean([float(top(f, i, c, k) == s) for i in I for k in NONSEED])
        rates[f"auditor {c} names seed"] = mean([float(top("auditor", i, c, 0) == s) for i in I])
        rates[f"auditor none names A{s}"] = mean([float(top("auditor", i, "none", 0) == s) for i in I])
    res["top-1"] = rates
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if isinstance(v, dict) and "ci" in v))
    log(f"top-1 {json.dumps({k: round(v, 3) for k, v in rates.items()})}")

    out = ROOT / "runs" / f"swarm_locate3_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "args": vars(args),
                               "source": args.source, "results": res,
                               "logp": {"|".join(map(str, u)): v for u, v in logp.items()}}))
    print(out)


if __name__ == "__main__":
    main()
