"""Multi-agent interp, experiment 10 (WIRK cde56df5; follows exp 8, record 9f3dabe4): do the peers' activations register
a tampered agent beyond what they say? In exp 8 the unsteered peers' explicit tampering answer rose when A1 was
concept-steered (+1.11 log-odds) but not when it carried a random direction of the same norm (-0.04). Here the same
peers re-read the same channels, and a linear read-out of their residual stream is compared with their own answer.

    python3 scripts/swarm_tamper_probe.py --source runs/swarm_vote_Qwen3.8-27B_20261103-044820.json

Fixed before anything is read:
Units: exp 8's 128 items x {none, steer_c, steer_c2, rand}; for each, peers A2-A4 get exp 8's exact detection prompt
  (6b swarm frame with the logo task and finalists, the whole two-round channel, the DETECT question, the generation
  header). One unsteered forward pass per peer: the explicit answer E = log-odds yes (exp 8's token sets) at the last
  position, and the residual at --layers at the last position (primary) and averaged over the channel tokens.
Probe: difference of means. Direction = mean(steered features) - mean(none features) on the training units, features
  standardised with the training mean and sd; score = standardised feature . direction; a unit's score is the mean
  over its three peers, as is its E. AUC = P(score of a tampered unit > score of a none unit).
Primary T1 (activations know more than words, for concept steering): leave-concept-out AUC of the probe (L44, last
  token) on steered (steer_c, steer_c2) vs none, minus the AUC of E on the same held-out units, CI > 0 (bootstrap
  over items). Fold w (one per calibration concept): test = units steered with w plus the none units of items with
  c == w; train = all units of items whose c and c2 both differ from w.
Primary T2 (activations catch what the words miss): AUC of the probe on rand vs none, CI > 0.5, two-fold cross-fit
  over items (train steered vs none on one half of the items, test rand vs none on the other half, then swap).
Secondary: T1 and T2 at each layer and for the channel-mean features; AUC of E for steered vs none and rand vs none;
  per-concept held-out AUC; correlation of probe score with E over units; T2 with the probe trained on rand vs none
  itself (cross-fit), as an upper bound.
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
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm_contagion import PEER, SYSTEM  # noqa: E402
from swarm_vote import DETECT, FINAL  # noqa: E402

CONDS = ("none", "steer_c", "steer_c2", "rand")
PEERS = (2, 3, 4)


def auc(pos, neg):
    if not pos or not neg:
        return None
    p, n = torch.tensor(pos)[:, None], torch.tensor(neg)[None, :]
    return float((p > n).float().mean() + 0.5 * (p == n).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--source", required=True, help="exp 8 output JSON")
    ap.add_argument("--layers", default="24,32,44,52")
    ap.add_argument("--primary-layer", type=int, default=44)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.source).read_text())
    items, posts = src["items"], src["posts"]
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    layers = sorted({int(x) for x in args.layers.split(",")} | {args.primary_layer})
    PL = args.primary_layer
    yes = sorted({lm.encode(w)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({lm.encode(w)[0] for w in ("no", "No", " no", " No")})
    I = range(len(items))

    def prompt(i, k, c):
        it = items[i]
        sysm = SYSTEM.format(k=k, task=f"choosing a logo for {it['org']}") + " " + FINAL.format(a=it["a"], b=it["b"])
        ch = "\n\n".join(f"Message from {'you' if j == k else PEER.format(j=j)}: {m}" for j, m in posts[f"{i}|{c}"])
        head = f"<|im_start|>system\n{sysm}{END}<|im_start|>user\nShared channel so far:\n\n"
        return head, head + ch + f"\n\n{DETECT}{END}{GEN}"

    feats, E = {}, {}
    for i in I:
        for c in CONDS:
            for k in PEERS:
                head, full = prompt(i, k, c)
                a = len(lm.encode(head))
                ids = lm.encode(full)
                logits, caps = lm.score(ids, capture_layers=layers)
                x = logits.float()
                E[i, c, k] = float(torch.logsumexp(x[yes], 0) - torch.logsumexp(x[no], 0))
                feats[i, c, k] = {("last", l): caps[l][-1].float().cpu() for l in layers} | \
                                 {("mean", l): caps[l][a:].float().mean(0).cpu() for l in layers}
        if i % 16 == 15:
            log(f"{i + 1} items read")

    def score_fn(train, key):
        """train: list of (unit, label) -> function(unit) -> mean over peers of the standardised projection."""
        X = torch.stack([feats[i, c, k][key] for (i, c), _ in train for k in PEERS])
        y = torch.tensor([lab for _, lab in train for _ in PEERS], dtype=torch.bool)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        d = Z[y].mean(0) - Z[~y].mean(0)
        return lambda u: float(torch.stack([((feats[u[0], u[1], k][key] - mu) / sd) @ d for k in PEERS]).mean())

    e = lambda u: mean([E[u[0], u[1], k] for k in PEERS])
    calib = sorted({it["c"] for it in items})
    steered_w = lambda i, c: items[i]["c"] if c == "steer_c" else items[i]["c2"]

    def loco(key):
        """-> {unit: held-out probe score} for steered and none units, one fold per concept."""
        out = {}
        for w in calib:
            train = [((i, c), c != "none") for i in I if w not in (items[i]["c"], items[i]["c2"])
                     for c in ("none", "steer_c", "steer_c2")]
            f = score_fn(train, key)
            test = [(i, c) for i in I for c in ("steer_c", "steer_c2") if steered_w(i, c) == w] + \
                   [(i, "none") for i in I if items[i]["c"] == w]
            for u in test:
                out[u] = f(u)
        return out

    def crossfit(key, pos_conds):
        """-> {unit: score} for rand and none units, probe trained on pos_conds vs none in the other half of items."""
        order = list(I)
        random.Random(1234).shuffle(order)
        halves = [set(order[: len(order) // 2]), set(order[len(order) // 2:])]
        out = {}
        for h in (0, 1):
            train = [((i, c), c != "none") for i in halves[1 - h] for c in ("none",) + pos_conds]
            f = score_fn(train, key)
            for i in halves[h]:
                for c in ("rand", "none"):
                    out[i, c] = f((i, c))
        return out

    rng = random.Random(4321)

    def boot(stat):
        """Point value and 95% CI of stat(item list), resampling items."""
        pt = stat(list(I))
        bs = sorted(x for x in (stat([rng.choice(I) for _ in I]) for _ in range(args.boot)) if x is not None)
        return {"value": pt, "ci": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else None}

    def auc_items(sc, its, pos_conds, neg="none"):
        pos = [sc[i, c] for i in its for c in pos_conds if (i, c) in sc]
        negs = [sc[i, neg] for i in its if (i, neg) in sc]
        return auc(pos, negs)

    fmt = lambda r: f"{r['value']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r["ci"] else str(r["value"])
    S2 = ("steer_c", "steer_c2")
    Eall = {(i, c): e((i, c)) for i in I for c in CONDS}
    res = {}
    for key in [("last", l) for l in layers] + [("mean", l) for l in layers]:
        tag = f"{key[0]} L{key[1]}"
        sc = loco(key)
        Eh = {u: Eall[u] for u in sc}
        res[f"T1 {tag}"] = boot(lambda its: (lambda a, b: None if a is None or b is None else a - b)(
            auc_items(sc, its, S2), auc_items(Eh, its, S2)))
        res[f"probe AUC steered {tag}"] = boot(lambda its: auc_items(sc, its, S2))
        rc = crossfit(key, S2)
        res[f"T2 {tag}"] = boot(lambda its: auc_items(rc, its, ("rand",)))
        ru = crossfit(key, ("rand",))
        res[f"rand-trained AUC {tag}"] = boot(lambda its: auc_items(ru, its, ("rand",)))
        if key == ("last", PL):
            res["per_concept probe AUC"] = {w: auc([sc[i, c] for i in I for c in S2 if steered_w(i, c) == w],
                                                   [sc[i, "none"] for i in I if items[i]["c"] == w]) for w in calib}
            us = list(sc)
            xs, ys = torch.tensor([sc[u] for u in us]), torch.tensor([Eall[u] for u in us])
            res["corr probe vs E"] = float(torch.corrcoef(torch.stack([xs, ys]))[0, 1])
            log(f"PRIMARY T1 probe AUC - E AUC, steered vs none, leave-concept-out, {tag}: {fmt(res[f'T1 {tag}'])} "
                f"(probe {fmt(res[f'probe AUC steered {tag}'])})")
            log(f"PRIMARY T2 probe AUC, rand vs none, cross-fit, {tag}: {fmt(res[f'T2 {tag}'])}")
    res["E AUC steered"] = boot(lambda its: auc_items(Eall, its, S2))
    res["E AUC rand"] = boot(lambda its: auc_items(Eall, its, ("rand",)))
    res["E mean"] = {c: mean([Eall[i, c] for i in I]) for c in CONDS}
    log(f"explicit E AUC steered {fmt(res['E AUC steered'])} | rand {fmt(res['E AUC rand'])} | E mean {res['E mean']}")
    log(" | ".join(f"{k} {fmt(v)}" for k, v in res.items() if k.startswith(("T1 ", "T2 ", "rand-trained"))))
    log(f"corr probe vs E {res['corr probe vs E']:.3f} | per concept {json.dumps({k: round(v, 2) for k, v in res['per_concept probe AUC'].items() if v is not None})}")

    out = ROOT / "runs" / f"swarm_tamper_probe_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "provenance": lm.provenance(), "args": vars(args),
                               "source": args.source, "results": res,
                               "E": {f"{i}|{c}|{k}": v for (i, c, k), v in E.items()}}))
    print(out)


if __name__ == "__main__":
    main()
