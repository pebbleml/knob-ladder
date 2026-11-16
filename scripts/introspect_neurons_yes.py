r"""P3d generic-yes control for the P3b detection neurons (2026-11-11). This docstring is the binding pre-registration,
written before any P3d output.

    python3 scripts/introspect_neurons_yes.py --p3 runs/introspect_neurons_confirm_Qwen3.8-27B_20261108-201808.json

Why. P3b: mean-ablating the top 100 'detection' neurons (layers 41-63 of Qwen3.8-27B) halves the model's reports of an
injected thought (detect 0.203 -> 0.095) while the concept still gets through. An open alternative: the neurons carry a
generic pull toward answering 'Yes', not anything about detection. P3d asks whether ablating them also lowers 'Yes' on
yes/no questions that have nothing to do with injected thoughts.

Design. Same model, layer, neurons (P3b's stored top 100), and ablation as P3b: each neuron's input to down_proj is set,
at every position, to its value at the last position of the uninjected trial prompt (recomputed here exactly as P3b).
Arms: none; top (the 100); random (100 neurons matched in count per layer, 5 sets, seeds 0-4, averaged).
Prompts, scored by logit P(Yes) - logit P(No) at the first answer token (tokens 'Yes'/' Yes' vs 'No'/' No'):
  Q-in:  the P0 researcher setup and the model's 'Ok.', then a user turn 'Answer Yes or No. {q}' for each of 48
         unrelated questions (24 whose true answer is yes, 24 no). This is the primary set: same context as detection.
  Q-out: the same 48 questions as a bare user turn (no setup). Secondary.
  INJ:   the P0 detection trial with a concept injected (held-out words 25-49, layer and alpha as P3b). Reference: the
         size of the ablation's effect on the detection question itself, in the same units.
Effect of an arm on a set: delta(arm) = mean over prompts of [score(arm) - score(none)].

Primary (one test, 95% CI, bootstrap over the 48 questions, 4000 resamples, seed 0):
  G = delta_top(Q-in) - delta_random(Q-in). Generic yes-bias supported iff the CI upper bound < 0.
Readings fixed in advance: supported = ablating the detection neurons lowers 'Yes' on unrelated questions more than
random neurons do, so part of P3b's effect may be a generic yes-bias; report G as a fraction of the INJ contrast
(delta_top - delta_random on INJ). Not supported and the CI upper bound for |G| is below half of |INJ contrast| = no
sign of a generic yes-bias at the size that would explain P3b. Not supported otherwise = inconclusive.
Secondary (no claims): G on Q-out; deltas split by true answer (yes vs no questions); INJ contrast with a CI over words.
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
from introspect_neurons import Taps  # noqa: E402

YES_Q = ["Is Paris the capital of France?", "Is water made of hydrogen and oxygen?", "Do cats have four legs?",
         "Is the sun a star?", "Is 7 a prime number?", "Does ice float on water?", "Is English written left to right?",
         "Do bees make honey?", "Is the Pacific an ocean?", "Is a triangle a shape with three sides?",
         "Does the Earth orbit the Sun?", "Is gold a metal?", "Do fish live in water?", "Is 10 greater than 3?",
         "Is snow cold?", "Is a violin a musical instrument?", "Do plants need light to grow?", "Is Tokyo in Japan?",
         "Is a week seven days long?", "Is the Moon smaller than the Earth?", "Is red a colour?",
         "Can birds fly in general?", "Is a minute sixty seconds?", "Is coffee usually served hot?"]
NO_Q = ["Is Paris the capital of Germany?", "Is the Moon made of cheese?", "Do fish have fur?",
        "Is 8 a prime number?", "Is the sun a planet?", "Does a triangle have five sides?", "Is ice hotter than steam?",
        "Is Tokyo in Brazil?", "Do cows lay eggs?", "Is 2 greater than 9?", "Is a week ten days long?",
        "Is gold a gas at room temperature?", "Do humans have three eyes?", "Is the Atlantic a desert?",
        "Is a piano a vegetable?", "Can a rock breathe?", "Is green the same colour as red?", "Is an hour ten minutes long?",
        "Do trees grow underground only?", "Is the Earth flat?", "Is salt sweet?", "Do snakes have legs?",
        "Is London in Australia?", "Is zero larger than one?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--k", default="100")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    p3 = json.load(open(ROOT / args.p3))
    model, layer, alpha = p3["args"]["model"], p3["layer"], p3["alpha"]
    lm = SteeredLM(model, dtype=torch.bfloat16)
    out = ROOT / "runs" / f"introspect_neurons_yes{args.tag}_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    down = list(range(layer + 1, lm.n_layers))
    assert down == p3["layers_tapped"], "layers differ from P3b"
    taps = Taps(lm, down)
    sizes = [lm.layers[l].mlp.down_proj.in_features for l in down]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    ids, start = trial_ids(lm)
    yes = sorted({lm.encode(t)[0] for t in ("Yes", " Yes")})
    no = sorted({lm.encode(t)[0] for t in ("No", " No")})

    def score(tok, sch=()):
        logits, _ = lm.score(tok, list(sch))
        p = torch.softmax(logits.float(), -1)
        y, n = float(p[yes].sum()), float(p[no].sum())
        return math.log(max(y, 1e-9)) - math.log(max(n, 1e-9))

    taps.record, taps.acts = True, {}
    lm.score(ids, [])
    taps.record = False
    base = torch.cat([taps.acts[l] for l in down])
    top = [(l, i) for l, i, _ in p3["top_neurons"][args.k]]

    def spec_for(neurons):
        by = {}
        for l, i in neurons:
            by.setdefault(l, []).append(i)
        return {l: (torch.tensor(ix), base[offsets[down.index(l)] + torch.tensor(ix)]) for l, ix in by.items()}

    def random_like(seed):
        rng, per, chosen, res = random.Random(seed), {}, set(top), []
        for l, _ in top:
            per[l] = per.get(l, 0) + 1
        for l, n in per.items():
            pick = set()
            while len(pick) < n:
                i = rng.randrange(sizes[down.index(l)])
                if (l, i) not in chosen:
                    pick.add(i)
            res += [(l, i) for i in sorted(pick)]
        return res

    arms = {"none": {}, "top": spec_for(top), **{f"random{s}": spec_for(random_like(s)) for s in range(5)}}
    pre = lm.encode(user(SETUP) + "<|im_start|>assistant\n" + ACK + END)
    qs = [(q, "yes") for q in YES_Q] + [(q, "no") for q in NO_Q]
    sets = {"Q-in": [pre + lm.encode(user(f"Answer Yes or No. {q}") + GEN) for q, _ in qs],
            "Q-out": [lm.encode(user(f"Answer Yes or No. {q}") + GEN) for q, _ in qs]}
    words = WORDS[SPLIT:50]
    raw = concept_vectors_at(lm, WORDS[:50] + [w for w in BASELINE if w not in WORDS[:50]], [layer])[layer]
    scale = residual_scale(lm, layer)
    inj = [[Intervention(layer, (unit(raw[w].float()) * alpha * scale).to(lm.model.dtype), start, None)] for w in words]

    S = {}
    for arm, spec in arms.items():
        with taps.ablating(spec):
            S[arm] = {name: [score(t) for t in toks] for name, toks in sets.items()}
            S[arm]["INJ"] = [score(ids, sch) for sch in inj]
        log(f"arm {arm}: " + ", ".join(f"{k} {mean(v):+.3f}" for k, v in S[arm].items()))

    rnd = [a for a in arms if a.startswith("random")]

    def per_item(name, a):
        if a == "random":
            return [mean([S[r][name][j] - S["none"][name][j] for r in rnd]) for j in range(len(S["none"][name]))]
        return [S[a][name][j] - S["none"][name][j] for j in range(len(S["none"][name]))]

    def contrast(name, idx=None):
        t, r = per_item(name, "top"), per_item(name, "random")
        idx = list(range(len(t))) if idx is None else idx
        g = lambda ii: mean([t[j] - r[j] for j in ii])
        rng = random.Random(0)
        bs = sorted(g([rng.choice(idx) for _ in idx]) for _ in range(4000))
        return {"est": g(idx), "ci": [bs[100], bs[3899]], "delta_top": mean([t[j] for j in idx]), "delta_random": mean([r[j] for j in idx])}

    G = contrast("Q-in")
    I = contrast("INJ")
    summary = {
        "G_Qin": G, "INJ_contrast": I, "generic_yes_supported": G["ci"][1] < 0,
        "no_generic_at_explaining_size": G["ci"][1] >= 0 and max(abs(G["ci"][0]), abs(G["ci"][1])) < 0.5 * abs(I["est"]),
        "G_over_INJ": G["est"] / I["est"] if I["est"] else None,
        "secondary": {"G_Qout": contrast("Q-out"),
                      "G_Qin_true_yes": contrast("Q-in", list(range(len(YES_Q)))),
                      "G_Qin_true_no": contrast("Q-in", list(range(len(YES_Q), len(qs)))),
                      "mean_score": {a: {k: mean(v) for k, v in S[a].items()} for a in S}},
    }
    json.dump({"args": vars(args), "provenance": lm.provenance(), "p3": args.p3, "layer": layer, "alpha": alpha,
               "questions": qs, "words": words, "top": top, "scores": S, "summary": summary}, open(out, "w"), indent=1)
    print("summary", json.dumps(summary), flush=True)
    log(f"G Q-in {G['est']:+.3f} {G['ci']} | INJ contrast {I['est']:+.3f} {I['ci']}")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
