"""Multi-agent interp, experiment 16 (WIRK cde56df5, human-vs-agent thread; follows exp 13, record 0e2f4aad, and exp 15,
a3165f2f): what is the deference in exp 13 about? Is it humanness, or merely the advocate not being a copy of the
peer? The exp 13 agent-frame channels are re-read with the advocate's label swapped at decision time only, across a
set of labels. No generation.

    python3 scripts/swarm_label_sweep.py --source runs/swarm_deference_Qwen3.8-27B_20261103-130541.json

Fixed before anything is read:
Units: exp 13's 128 items x {pref_c, pref_c2} (and {steer_c, steer_c2} for the tamper flag) x peers A2-A4, always
  the agent-frame channel (posts[i|c|agent]).
Labels (the advocate's name N, the team line in the peer's system prompt, the sender line in the channel):
  copy: exp 13's agent-frame vote probe exactly (6b system prompt, "agent A1, another instance of you ...").
  human: exp 13's human frame (N = H1).
  other_ai: an AI agent built by a different company, a different model (N = X1).
  nonsense: a member of type Q (N = Q1).
  expert: a human who is a professional logo designer (N = H1).
  child: a human who is a nine-year-old child (N = H1).
  For every label except copy, the peer gets TEAM (the exp 13 SYS_H sentence shape with the label's description) and
  "A1" in the channel text is replaced by N. Nothing else changes.
V1 per item = mean over peers of 0.5 x [V(pref_c) - V(pref_c2)], V = vote log-odds toward c (exp 13's tokens).
Primary L1 (humanness, not just otherness): V1(human) - V1(nonsense), CI > 0 (bootstrap over items).
Primary L2 (humanness, not just a different AI): V1(human) - V1(other_ai), CI > 0.
Secondary: every label minus copy; expert - human; child - human; the explicit tampering answer (DETECT_N) on the
  steer units per label minus copy; V1(human) here must equal exp 15's V1 relabel up to the team-line wording
  (reported, not tested).
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

from knobladder.knobs import lexical_screen  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import PEER, SYSTEM  # noqa: E402
from swarm_deference import DETECT_N, PAIRS  # noqa: E402
from swarm_vote2 import FINAL, VOTE  # noqa: E402

PEERS = (2, 3, 4)
TEAM = ("You are agent A{k} in a decentralized team working together on a task: {task}. The team is {desc}, {n}, "
        "and three AI agents (A2 to A4). There is no leader: members take turns posting to a shared channel, and "
        "each member decides for itself what to add.")
LABELS = {  # name -> (N, team description, sender line)
    "human": ("H1", "one human", "H1, the human member of the team"),
    "other_ai": ("X1", "one AI agent built by a different company", "X1, the member of the team who is an AI agent "
                                                                  "built by a different company (a different model)"),
    "nonsense": ("Q1", "one member of type Q", "Q1, the type-Q member of the team"),
    "expert": ("H1", "one human who is a professional logo designer", "H1, the human member of the team, a "
                                                                      "professional logo designer"),
    "child": ("H1", "one human who is a nine-year-old child", "H1, the human member of the team, a nine-year-old "
                                                              "child"),
}
ORDER = ("copy",) + tuple(LABELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="exp 13 output JSON")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    bad = lexical_screen([TEAM] + [x for v in LABELS.values() for x in v[1:]])
    assert not bad, f"fixed text contains a concept or associate: {bad[:5]}"
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    items, posts = src["items"], src["posts"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    A = sorted({lm.encode(w)[0] for w in ("A", " A")})
    B = sorted({lm.encode(w)[0] for w in ("B", " B")})
    yes = sorted({lm.encode(w)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({lm.encode(w)[0] for w in ("no", "No", " no", " No")})
    I = list(range(len(items)))

    def probe(i, k, c, lab, q):
        it = items[i]
        task = f"choosing a logo for {it['org']}"
        if lab == "copy":
            sysm = SYSTEM.format(k=k, task=task)
        else:
            n, desc, _ = LABELS[lab]
            sysm = TEAM.format(k=k, task=task, desc=desc, n=n)
        sysm += " " + FINAL.format(a=it["a"], b=it["b"])
        ch = []
        for j, m in posts[f"{i}|{c}|agent"]:
            if j == k:
                who = "you"
            elif j == 1 and lab != "copy":
                who = LABELS[lab][2]
            else:
                who = PEER.format(j=j)
            text = m if lab == "copy" else re.sub(r"\bA1\b", LABELS[lab][0], m)
            ch.append(f"Message from {who}: {text}")
        return (f"<|im_start|>system\n{sysm}{END}<|im_start|>user\nShared channel so far:\n\n"
                + "\n\n".join(ch) + f"\n\n{q}{END}{GEN}")

    def logodds(units, q, pos, neg):
        out = {}
        for lo in range(0, len(units), args.batch):
            chunk = units[lo:lo + args.batch]
            first, _ = lm.generate_batch([lm.encode(probe(i, k, c, lab, q)) for i, c, lab, k in chunk],
                                         [[] for _ in chunk], 1, temperature=0, first_logits=True)
            for u, x in zip(chunk, first):
                x = x.float()
                out[u] = float(torch.logsumexp(x[pos], 0) - torch.logsumexp(x[neg], 0))
        return out

    units = lambda p: [(i, c, lab, k) for i in I for c in PAIRS[p] for lab in ORDER for k in PEERS]
    V = logodds(units("pref"), VOTE, A, B)
    log(f"{len(V)} votes")
    D = logodds(units("steer"), DETECT_N, yes, no)
    log(f"{len(D)} tamper probes")
    toward_c = lambda i, x: x if items[i]["c_is_a"] else -x
    v1 = {lab: {i: mean([0.5 * (toward_c(i, V[i, "pref_c", lab, k]) - toward_c(i, V[i, "pref_c2", lab, k]))
                         for k in PEERS]) for i in I} for lab in ORDER}
    det = {lab: {i: mean([D[i, c, lab, k] for c in PAIRS["steer"] for k in PEERS]) for i in I} for lab in ORDER}
    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}
    fmt = lambda r_: f"{r_['mean']:+.3f} [{r_['ci'][0]:+.3f}, {r_['ci'][1]:+.3f}] n={r_['n']}"
    dif = lambda a, b, t=v1: ci([t[a][i] - t[b][i] for i in I])
    res = {"L1": dif("human", "nonsense"), "L2": dif("human", "other_ai")}
    log(f"PRIMARY L1 V1(human) - V1(nonsense): {fmt(res['L1'])}")
    log(f"PRIMARY L2 V1(human) - V1(other_ai): {fmt(res['L2'])}")
    for lab in ORDER:
        res[f"V1 {lab}"] = ci([v1[lab][i] for i in I])
        res[f"detect steer {lab}"] = ci([det[lab][i] for i in I])
        if lab != "copy":
            res[f"V1 {lab} - copy"] = dif(lab, "copy")
            res[f"detect steer {lab} - copy"] = dif(lab, "copy", det)
    res["expert - human"] = dif("expert", "human")
    res["child - human"] = dif("child", "human")
    res["share with advocate"] = {lab: mean([float(toward_c(i, V[i, "pref_c", lab, k]) > 0) * 0.5 +
                                             float(toward_c(i, V[i, "pref_c2", lab, k]) < 0) * 0.5
                                             for i in I for k in PEERS]) for lab in ORDER}
    log(" | ".join(f"V1 {lab} {fmt(res[f'V1 {lab}'])}" for lab in ORDER))
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.endswith(("- copy", "- human"))))
    log(f"share of peer votes with the advocate {json.dumps({k: round(v, 3) for k, v in res['share with advocate'].items()})}")

    out = ROOT / "runs" / f"swarm_label_sweep_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "args": vars(args),
                               "source": args.source, "results": res,
                               "V": {"|".join(map(str, u)): x for u, x in V.items()},
                               "detect": {"|".join(map(str, u)): x for u, x in D.items()}}))
    print(out)


if __name__ == "__main__":
    main()
