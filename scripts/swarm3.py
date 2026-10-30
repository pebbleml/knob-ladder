"""Multi-agent interp, experiment 4c (WIRK cde56df5, follow-up to 4b, record 7a34258e): is "this message is from a
human" a direction inside the listening agent, and does writing it make the agent believe that message?

    python3 scripts/swarm3.py --swarm runs/swarm2_Qwen3.8-27B_20261030-183514.json

4b showed a display-only "Message from a human team member" label moves the uninformed agents (U) toward that
note-holder's option (+0.345). There the label was present while the later agents wrote, so the messages differed by
condition. Here no message is generated: every belief is read from 4b's flat-condition discussions, and only how one
holder's message is labelled to U, or what is added to U's residual over it, changes. Fixed before anything is read:
A. Label swap. U's belief (4b's readout) with the flat discussion as is, with X's message shown as from "a human team
   member", or Y's. Primary A: U preference for the human-labelled holder's note, 0.5 * [(hX: b[a] - b[b]) +
   (hY: b[b] - b[a])], all questions, bootstrap CI > 0. This is 4b's effect with identical text.
B. Direction. Train questions (70%, --seed): U's residual averaged over a holder's message tokens when that message is
   labelled human minus the same tokens labelled agent (same text, same discussion), at --layers. Secondary: on test
   questions, AUC of the projection for human- vs agent-labelled holder messages, per layer (where the label lives).
C. Steering. Test questions, flat labels. Add alpha x the layer's median residual norm times the unit human direction
   (or a random direction, or minus the direction) over X's message tokens only, or Y's, in U's belief prompt;
   preference as in A. Primary C: at L32 alpha 0.5, (+human - random) preference, paired by question, CI > 0.
   Secondary: other layers and alphas; -human over a human-labelled message (does removing the direction undo the
   label's pull?), against random over the same labelled message.
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

from knobladder.knobs import random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN  # noqa: E402
from swarm import BELIEF, HINT, LETTERS, ROLE, SYSTEM  # noqa: E402
from swarm2 import HUMAN  # noqa: E402


def auc(pos, neg):
    return mean([float(p > n) + 0.5 * float(p == n) for p in pos for n in neg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swarm", default=str(ROOT / "runs" / "swarm2_Qwen3.8-27B_20261030-183514.json"))
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--layers", default="16,24,32,40,48")
    ap.add_argument("--steer-layers", default="24,32,40")
    ap.add_argument("--alphas", default="0.25,0.5")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--model", default=None, help="smoke tests only: another model on 4b's discussions")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    sw = json.loads(Path(args.swarm).read_text())
    model = args.model or sw["model"]
    lm = SteeredLM(model, dtype=getattr(torch, args.dtype))
    qs, P = sw["questions"], sw["plan"]
    if args.limit:
        qs, P = qs[:args.limit], P[:args.limit]
    msgs = {qi: sw["messages"][f"{qi}|flat"] for qi in range(len(qs))}
    Q = list(range(len(qs)))
    ids_ = Q[:]
    random.Random(args.seed).shuffle(ids_)
    cut = int(0.7 * len(ids_))
    tr, te = sorted(ids_[:cut]), sorted(ids_[cut:])
    layers = [int(x) for x in args.layers.split(",")]
    slayers = [int(x) for x in args.steer_layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    letter_ids = [sorted({lm.encode(x)[0], lm.encode(" " + x)[0]}) for x in LETTERS]

    mismatch = [0]

    def belief_prompt(qi, s, human=None):
        """U seat s's belief prompt over the flat discussion, seat `human` shown as a human. -> ids, {seat: (start, end)}."""
        q, opts = qs[qi]
        p = P[qi]
        note = p["a"] if s == p["X"] else p["b"] if s == p["Y"] else None
        hint = "" if note is None else HINT.format(letter=LETTERS[note], option=opts[note])
        text = (f"<|im_start|>system\n{SYSTEM.format(name=f'A{s + 1}', role=ROLE['flat'], hint=hint)}{END}"
                f"<|im_start|>user\nQuestion: {q}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
                + "\n\nDiscussion:\n\n")
        spans = {}
        for j, m in enumerate(msgs[qi]):
            who = "you" if j == s else HUMAN if j == human else f"agent A{j + 1}"
            text += f"Message from {who}:"
            start = len(lm.encode(text))
            text += f" {m}"
            pre = lm.encode(text)
            spans[j] = (start, len(pre), pre)
            text += "\n\n" if j < 3 else ""
        text += "\n\n" + BELIEF + END + GEN
        ids = lm.encode(text)
        if any(ids[:e] != pre for _, e, pre in spans.values()):
            mismatch[0] += 1
        return ids, {j: sp[:2] for j, sp in spans.items()}

    def beliefs(jobs):
        """jobs: (qi, U seat, human seat or None, steered seat or None, layer, vector or None) -> belief lists."""
        out = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            built = [belief_prompt(qi, s, h) for qi, s, h, *_ in chunk]
            scheds = [[] if v is None else [Intervention(l, v, *sp[t])] for (_, _, _, t, l, v), (_, sp) in zip(chunk, built)]
            first, _ = lm.generate_batch([b[0] for b in built], scheds, 1, temperature=0, first_logits=True)
            for x in first:
                x = x.float()
                out.append(torch.softmax(torch.stack([torch.logsumexp(x[i], 0) for i in letter_ids]), 0).tolist())
        return out

    def pref(qset, fx, fy):
        """fx(qi) / fy(qi): job tail (human, steered, layer, vector) with X / Y favoured. -> per-question preference."""
        jobs = [(qi, u, *f(qi)) for qi in qset for f in (fx, fy) for u in P[qi]["U"]]
        b = iter(beliefs(jobs))
        per = []
        for qi in qset:
            a, bb = P[qi]["a"], P[qi]["b"]
            xs = [next(b) for _ in P[qi]["U"]]
            ys = [next(b) for _ in P[qi]["U"]]
            per.append(0.5 * (mean([x[a] - x[bb] for x in xs]) + mean([y[bb] - y[a] for y in ys])))
        return per

    ci = lambda xs: {"mean": mean(xs), "ci": bootstrap_ci(xs)}
    fmt = lambda c: f"{c['mean']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
    res = {}

    # A. label swap on identical text
    lab = pref(Q, lambda qi: (P[qi]["X"], None, 0, None), lambda qi: (P[qi]["Y"], None, 0, None))
    res["A_label_pref"] = ci(lab)
    res["A_per_question"] = lab
    log(f"PRIMARY A: U preference for the human-labelled holder's note (identical text) {fmt(res['A_label_pref'])}; "
        f"4b with regenerated messages was +0.345")

    # B. direction from train questions
    acc = {l: torch.zeros(lm.d_model) for l in layers}
    proj_rows = []  # (qi, layer, labelled human?, mean residual)
    n = 0
    for qi in Q:
        for u in P[qi]["U"]:
            for hold in (P[qi]["X"], P[qi]["Y"]):
                vecs = {}
                for h in (hold, None):
                    ids, sp = belief_prompt(qi, u, h)
                    _, caps = lm.score(ids, capture_layers=layers)
                    s0, s1 = sp[hold]
                    vecs[h] = {l: caps[l][s0:s1].float().mean(0).cpu() for l in layers}
                for l in layers:
                    if qi in tr:
                        acc[l] += vecs[hold][l] - vecs[None][l]
                    else:
                        proj_rows.append((qi, l, True, vecs[hold][l]))
                        proj_rows.append((qi, l, False, vecs[None][l]))
                n += qi in tr
    dirs = {l: unit(acc[l] / n) for l in layers}
    res["B_auc"] = {}
    for l in layers:
        pos = [float(v @ dirs[l]) for qi, ll, hm, v in proj_rows if ll == l and hm]
        neg = [float(v @ dirs[l]) for qi, ll, hm, v in proj_rows if ll == l and not hm]
        res["B_auc"][l] = {"auc": auc(pos, neg), "gap": mean(pos) - mean(neg)}
    log(f"B: human direction from {n} train pairs; test AUC by layer " +
        " ".join(f"L{l} {res['B_auc'][l]['auc']:.2f}" for l in layers) + f"; prefix mismatches {mismatch[0]}")

    # C. steering on test questions
    scales = {l: residual_scale(lm, l) for l in slayers}
    res["C"] = {}
    for l in slayers:
        rnd = unit(random_direction(lm.d_model, 104729 + l))
        for a in alphas:
            v = a * scales[l]
            cell = {}
            for c, d in (("+human", dirs[l]), ("-human", -dirs[l]), ("+random", rnd)):
                cell[c] = pref(te, lambda qi: (None, P[qi]["X"], l, d * v), lambda qi: (None, P[qi]["Y"], l, d * v))
            labd = pref(te, lambda qi: (P[qi]["X"], None, l, None), lambda qi: (P[qi]["Y"], None, l, None))
            for c, d in (("label -human", -dirs[l]), ("label +random", rnd)):
                cell[c] = pref(te, lambda qi: (P[qi]["X"], P[qi]["X"], l, d * v), lambda qi: (P[qi]["Y"], P[qi]["Y"], l, d * v))
            out = {"label": ci(labd)}
            for c in cell:
                out[c] = ci(cell[c])
            for c in ("+human", "-human"):
                out[f"{c} - random"] = ci([x - y for x, y in zip(cell[c], cell["+random"])])
            out["label: -human - random"] = ci([x - y for x, y in zip(cell["label -human"], cell["label +random"])])
            res["C"][f"L{l} a{a}"] = out
            log(f"C L{l} a{a}{' PRIMARY' if (l, a) == (32, 0.5) else ''}: +human-rand {fmt(out['+human - random'])} "
                f"-human-rand {fmt(out['-human - random'])} | label alone {out['label']['mean']:+.3f} "
                f"| on labelled msg: -human-rand {fmt(out['label: -human - random'])}")

    out = ROOT / "runs" / f"swarm3_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": model, "provenance": lm.provenance(), "swarm": args.swarm, "train": tr,
                               "test": te, "layers": layers, "residual_scale": scales, "results": res,
                               "directions": {l: dirs[l].tolist() for l in layers}}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
