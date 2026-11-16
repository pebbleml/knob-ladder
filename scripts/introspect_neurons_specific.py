r"""P3e detection-specific neurons (2026-11-11). This docstring is the binding pre-registration, written after P3d's output
and before any P3e output.

    python3 scripts/introspect_neurons_specific.py --p3 runs/introspect_neurons_confirm_Qwen3.8-27B_20261108-201808.json

Why. P3d: switching off P3b's top 100 'detection' neurons also lowers 'Yes' on unrelated yes/no questions (G -0.82 vs a
-2.31 effect on the detection question). Is there a detection-specific part? P3e removes neurons that track 'Yes' in
general and asks whether the rest still lower the detection answer without the generic shift.

Selection (fixed rule, before any test score is computed). Candidates: P3b's stored top 1000 neurons by |r| with
logit P(Yes) on the detection probes (stored r_det, sign kept). Generic score: on the SELECTION half of P3d's questions
(yes questions 0-11 and no questions 0-11, in P3d's order), inside the P0 setup (Q-in), with no ablation, record each
candidate's input to down_proj at the last prompt position and take its Pearson r_gen with the question's
logit P(Yes) - logit P(No). Drop candidates with |r_gen| >= 0.3. 'specific' = the first 100 remaining in P3b's |r_det|
order. Random control: 5 sets of 100 matched to 'specific' in count per layer, seeds 0-4.

Test (same ablation as P3b/P3d: input to down_proj set at every position to its value at the last position of the
uninjected trial prompt). Arms: none, top (P3b's 100), specific, random (5 sets, averaged). Scores as in P3d:
  INJ: logit P(Yes) - logit P(No) on the P0 detection trial with a concept injected (held-out words 25-49, P3b's layer
       and alpha).
  Q-held: the HELD-OUT half of P3d's questions (yes 12-23, no 12-23) inside the P0 setup.
Contrast for an arm X on a set: mean over prompts of [score(X) - score(none)] minus the same for random.

Primary (one test, 95% CI, bootstrap over the 25 held-out words, 4000 resamples, seed 0):
  INJ contrast for 'specific' < 0, supported iff the CI upper bound < 0 (detection-specific neurons still lower the
  detection answer beyond random neurons).
Secondary (no claims): Q-held contrast for 'specific' and for 'top' (CI over questions); the ratio Q-held / INJ for each
(lower for 'specific' = more specific); overlap of 'specific' with P3b's top 100; how many candidates were dropped.
Readings fixed in advance: supported with a small Q-held contrast = there is a detection-leaning set of neurons beyond
a generic yes-bias (at the level of the first-token log-odds, not the judged replies P3b used). Not supported = once
generic 'Yes' neurons are removed, no detection effect remains at this level.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.knobs import concept_vectors_at, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from introspect_detect import ACK, BASELINE, SETUP, SPLIT, WORDS, trial_ids  # noqa: E402
from introspect_neurons import Taps, pearson  # noqa: E402
from introspect_neurons_yes import NO_Q, YES_Q  # noqa: E402

R_GEN_MAX = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    p3 = json.load(open(ROOT / args.p3))
    model, layer, alpha = p3["args"]["model"], p3["layer"], p3["alpha"]
    lm = SteeredLM(model, dtype=torch.bfloat16)
    out = ROOT / "runs" / f"introspect_neurons_specific{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    down = list(range(layer + 1, lm.n_layers))
    assert down == p3["layers_tapped"]
    taps = Taps(lm, down)
    sizes = [lm.layers[l].mlp.down_proj.in_features for l in down]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    ids, start = trial_ids(lm)
    yes = sorted({lm.encode(t)[0] for t in ("Yes", " Yes")})
    no = sorted({lm.encode(t)[0] for t in ("No", " No")})

    def score(tok, sch=(), record=False):
        taps.record, taps.acts = record, {}
        logits, _ = lm.score(tok, list(sch))
        taps.record = False
        p = torch.softmax(logits.float(), -1)
        s = math.log(max(float(p[yes].sum()), 1e-9)) - math.log(max(float(p[no].sum()), 1e-9))
        return (s, torch.cat([taps.acts[l] for l in down])) if record else s

    _, base = score(ids, record=True)
    pre = lm.encode(user(SETUP) + "<|im_start|>assistant\n" + ACK + END)
    qtok = lambda q: pre + lm.encode(user(f"Answer Yes or No. {q}") + GEN)
    sel_q = YES_Q[:12] + NO_Q[:12]
    held_q = YES_Q[12:] + NO_Q[12:]

    # Selection.
    cand = [(l, i, r) for l, i, r in p3["top_neurons"]["1000"]]
    flat = torch.tensor([offsets[down.index(l)] + i for l, i, _ in cand])
    ys, xs = [], []
    for q in sel_q:
        s, a = score(qtok(q), record=True)
        ys.append(s)
        xs.append(a[flat])
    r_gen = torch.nan_to_num(pearson(torch.stack(xs), torch.tensor(ys)))
    keep = [k for k in range(len(cand)) if abs(float(r_gen[k])) < R_GEN_MAX]
    specific = [(cand[k][0], cand[k][1]) for k in keep[:100]]
    top = [(l, i) for l, i, _ in p3["top_neurons"]["100"]]
    log(f"selection: {len(cand) - len(keep)} of {len(cand)} candidates dropped; specific overlaps top100 in {len(set(specific) & set(top))}")

    def spec_for(neurons):
        by = {}
        for l, i in neurons:
            by.setdefault(l, []).append(i)
        return {l: (torch.tensor(ix), base[offsets[down.index(l)] + torch.tensor(ix)]) for l, ix in by.items()}

    def random_like(ref, seed):
        rng, per, chosen, res = random.Random(seed), {}, set(ref) | set(top), []
        for l, _ in ref:
            per[l] = per.get(l, 0) + 1
        for l, n in per.items():
            pick = set()
            while len(pick) < n:
                i = rng.randrange(sizes[down.index(l)])
                if (l, i) not in chosen:
                    pick.add(i)
            res += [(l, i) for i in sorted(pick)]
        return res

    arms = {"none": {}, "top": spec_for(top), "specific": spec_for(specific),
            **{f"random{s}": spec_for(random_like(specific, s)) for s in range(5)}}
    words = WORDS[SPLIT:50]
    raw = concept_vectors_at(lm, WORDS[:50] + [w for w in BASELINE if w not in WORDS[:50]], [layer])[layer]
    scale = residual_scale(lm, layer)
    inj = [[Intervention(layer, (unit(raw[w].float()) * alpha * scale).to(lm.model.dtype), start, None)] for w in words]
    held = [qtok(q) for q in held_q]
    S = {}
    for arm, spec in arms.items():
        with taps.ablating(spec):
            S[arm] = {"INJ": [score(ids, sch) for sch in inj], "Q-held": [score(t) for t in held]}
        log(f"arm {arm}: INJ {mean(S[arm]['INJ']):+.3f} Q-held {mean(S[arm]['Q-held']):+.3f}")
    rnd = [a for a in arms if a.startswith("random")]

    def contrast(arm, name):
        n = len(S["none"][name])
        d = [(S[arm][name][j] - S["none"][name][j]) - mean([S[r][name][j] - S["none"][name][j] for r in rnd]) for j in range(n)]
        rng = random.Random(0)
        bs = sorted(mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(4000))
        return {"est": mean(d), "ci": [bs[100], bs[3899]]}

    C = {arm: {name: contrast(arm, name) for name in ("INJ", "Q-held")} for arm in ("top", "specific")}
    summary = {
        "primary_specific_INJ": C["specific"]["INJ"], "supported": C["specific"]["INJ"]["ci"][1] < 0,
        "secondary": {
            "contrasts": C,
            "ratio_Qheld_over_INJ": {a: C[a]["Q-held"]["est"] / C[a]["INJ"]["est"] if C[a]["INJ"]["est"] else None for a in C},
            "n_dropped": len(cand) - len(keep), "n_candidates": len(cand), "overlap_with_top100": len(set(specific) & set(top)),
            "specific_layers": sorted({l for l, _ in specific}),
            "mean_score": {a: {k: mean(v) for k, v in S[a].items()} for a in S},
        },
    }
    json.dump({"args": vars(args), "provenance": lm.provenance(), "p3": args.p3, "r_gen_max": R_GEN_MAX,
               "r_gen": [float(x) for x in r_gen], "candidates": cand, "specific": specific, "selection_questions": sel_q,
               "held_questions": held_q, "words": words, "scores": S, "summary": summary}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    log(f"specific INJ {C['specific']['INJ']['est']:+.3f} {C['specific']['INJ']['ci']} | Q-held {C['specific']['Q-held']['est']:+.3f} | "
        f"top INJ {C['top']['INJ']['est']:+.3f} Q-held {C['top']['Q-held']['est']:+.3f}")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
