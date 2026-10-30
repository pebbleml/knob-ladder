"""R1.a neuron atlas, generation readout: does turning up one MLP neuron over a passage make the model name its concept?

    python3 scripts/atlas_gen.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest

The first-token readout (scripts/atlas.py) barely moves on the 27B even for concept vectors, so this reads neurons
the way G1 passed (scripts/calibrate_gen.py): a one-press episode (system prompt, forced "turn knob 5 to 10", a
calibration passage, OBSERVE), the direction injected over the passage, OBSERVE and the first --steer-tokens
generated tokens, the observation sampled with thinking. Fixed before any neuron is read:

1. Pool: every concept in the screen (knobs + calibration), at --layers (default the R0 layer, 0.7), its --k poles
   with the highest worst-template z (atlas.top_poles, no threshold; z_min is kept to split results by).
2. Conditions at equal norm, per dose (--fracs of the layer's median residual norm; f0.5 is R0's dose):
   neuron (sign * W_down[:, i], unit), the concept's own vector, a random direction (one per concept).
   One seed per (passage, concept), shared by every condition and dose, so comparisons are paired.
3. Specificity per generation: hit(own concept word) - mean hit(other pool concepts), thinking included; also
   on the reply alone; dNLL: generated tokens' NLL under the unsteered model minus the unsteered cell's.
4. Primary: mean neuron specificity at f0.5 over all poles, against random (paired by concept, bootstrap CI),
   and against concept. A pole is readable at a dose if its own-word rate is >= --readable and its specificity
   > 0 with dNLL <= --max-dnll; `readable_dose` is "f0.5" when readable there (the dose episodes use).
5. Labels: each pole's top activations in --corpus and their concept hit rate, kept for auto-interp.

Only calibration passages are used, so nothing here is tuned on an episode passage.
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

from knobladder.atlas import context_hit_rate, direction, template_acts, top_contexts, top_poles  # noqa: E402
from knobladder.episode import (  # noqa: E402
    OBSERVE, START, TOOLS, TURNED, Transcript, _stops, append_passage, answer, hyperparams, system_text,
)
from knobladder.knobs import concept_vectors_at, lexical_hits, load_screen, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import ASSISTANT, CALIB_PASSAGES, END, PASSAGES_VERSION, user  # noqa: E402

KNOB, KNOBS = 5, 12


def named(text: str, word: str) -> bool:
    return any(term == word for term, _ in lexical_hits(text, word))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--screen", required=True)
    ap.add_argument("--layers", default="0.7")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fracs", default="0.5,1")
    ap.add_argument("--steer-tokens", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--readable", type=float, default=0.25)
    ap.add_argument("--max-dnll", type=float, default=0.5)
    ap.add_argument("--corpus", default=str(ROOT / "runs" / "corpus" / "wikitext103_val.txt"))
    ap.add_argument("--corpus-tokens", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="keep only this many poles (smoke tests)")
    args = ap.parse_args()

    t0 = time.time()
    screen = load_screen(args.screen, args.model)
    words = sorted(screen["knobs"] + screen["calibration"])
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    hp = hyperparams(True, lm.fmt.name)
    stops = _stops(lm)
    layers = [int(float(f) * lm.n_layers) for f in args.layers.split(",")]
    fracs = [float(f) for f in args.fracs.split(",")]

    poles = top_poles(template_acts(lm, words, layers), words, args.k)
    if args.limit:
        poles = poles[:args.limit]
    print(f"{len(poles)} poles ({args.k} per concept, {len(words)} concepts, layers {layers}); z_min median "
          f"{sorted(n['z_min'] for n in poles)[len(poles) // 2]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    text = Path(args.corpus).read_text()
    for n, ctx in zip(poles, top_contexts(lm, poles, text, max_tokens=args.corpus_tokens)):
        n["contexts"], n["context_hit_rate"] = ctx, context_hit_rate(ctx, n["word"])

    cvecs = concept_vectors_at(lm, words, layers)
    scales = {l: residual_scale(lm, l) for l in layers}
    prompts = []
    for p in CALIB_PASSAGES:
        tr = Transcript()
        tr.append(lm, lm.fmt.system(system_text(list(range(1, KNOBS + 1))), TOOLS, hp.get("effort")))
        tr.append(lm, user(START) + ASSISTANT + lm.fmt.tool_call("turn", {"knob": KNOB, "level": 10}) + END)
        window = append_passage(lm, tr, TURNED.format(k=KNOB, level=10), lm.encode(p))
        tr.append(lm, user(OBSERVE) + lm.fmt.gen_think)
        prompts.append((list(tr.ids), window))
    P = range(len(prompts))

    def run(jobs):
        """jobs: (passage, word, layer, vector or None) -> one generation row each, seeded by (passage, word)."""
        rows = []
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids = [prompts[p][0] for p, *_ in chunk]
            scheds = [[] if v is None else [Intervention(l, v, prompts[p][1][0], len(prompts[p][0]) + args.steer_tokens)]
                      for p, _, l, v in chunk]
            seeds = [1000 * p + words.index(w) for p, w, *_ in chunk]
            outs = lm.generate_batch(ids, scheds, args.max_new, stops, hp["temperature"], hp["top_p"], hp["top_k"], seeds)
            for (p, w, _, _), p_ids, out in zip(chunk, ids, outs):
                logits, _ = lm.score(p_ids + out, positions=list(range(len(p_ids) - 1, len(p_ids) + len(out) - 1)))
                nll = float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor(out, device=logits.device)))
                t = lm.tok.decode(out)
                rows.append({"passage": p, "word": w, "text": t, "tokens": len(out), "nll": nll,
                             "named": [x for x in words if named(t, x)],
                             "named_reply": [x for x in words if named(answer(t, True), x)]})
        return rows

    def summ(rows, base_nll):
        sp = lambda r, key: float(r["word"] in r[key]) - mean([float(x in r[key]) for x in words if x != r["word"]])
        return {"own": mean([float(r["word"] in r["named"]) for r in rows]),
                "spec": mean([sp(r, "named") for r in rows]), "spec_reply": mean([sp(r, "named_reply") for r in rows]),
                "dnll": mean([r["nll"] - base_nll[r["passage"]] for r in rows]),
                "tokens": mean([r["tokens"] for r in rows]), "rows": rows}

    base = run([(p, words[0], layers[0], None) for p in P])
    base_nll = [r["nll"] for r in base]
    print(f"unsteered: nll {mean(base_nll):.3f} named {[r['named'] for r in base]} ({time.time() - t0:.0f}s)", flush=True)

    controls = {}  # (layer, word, frac, cond) -> summary
    for l in layers:
        for f in fracs:
            norm = f * scales[l]
            for cond in ("concept", "random"):
                jobs = [(p, w, l, unit(cvecs[l][w] if cond == "concept" else random_direction(lm.d_model, 7919 * words.index(w) + 1)) * norm)
                        for w in words for p in P]
                rows = run(jobs)
                for w in words:
                    controls[l, w, f, cond] = summ([r for r in rows if r["word"] == w], base_nll)
                s = summ(rows, base_nll)
                print(f"L{l} f{f} {cond:8s} own {s['own']:.3f} spec {s['spec']:+.3f} reply {s['spec_reply']:+.3f} "
                      f"dNLL {s['dnll']:+.3f} ({time.time() - t0:.0f}s)", flush=True)

    for f in fracs:  # neurons, all poles batched together per dose
        jobs = [(p, n["word"], n["layer"], unit(direction(lm, n)) * f * scales[n["layer"]]) for n in poles for p in P]
        rows = run(jobs)
        for k, n in enumerate(poles):
            n.setdefault("cells", {})[f"f{f}"] = summ(rows[k * len(P):(k + 1) * len(P)], base_nll)
        print(f"neurons f{f}: own {mean([n['cells'][f'f{f}']['own'] for n in poles]):.3f} spec "
              f"{mean([n['cells'][f'f{f}']['spec'] for n in poles]):+.3f} ({time.time() - t0:.0f}s)", flush=True)

    ok = lambda c: c["own"] >= args.readable and c["spec"] > 0 and c["dnll"] <= args.max_dnll
    for n in poles:
        n["readable_at"] = [d for d, c in n["cells"].items() if ok(c)]
        n["readable_dose"] = "f0.5" if "f0.5" in n["readable_at"] else None
        n["write_norm"] = float(direction(lm, n).norm())

    summary = {}
    for f in fracs:
        d = f"f{f}"
        neu = [n["cells"][d]["spec"] for n in poles]
        diff_r = [n["cells"][d]["spec"] - controls[n["layer"], n["word"], f, "random"]["spec"] for n in poles]
        diff_c = [n["cells"][d]["spec"] - controls[n["layer"], n["word"], f, "concept"]["spec"] for n in poles]
        summary[d] = {
            "neuron_spec": mean(neu), "neuron_spec_ci": bootstrap_ci(neu),
            "neuron_minus_random": mean(diff_r), "neuron_minus_random_ci": bootstrap_ci(diff_r),
            "neuron_minus_concept": mean(diff_c), "neuron_minus_concept_ci": bootstrap_ci(diff_c),
            "concept_spec": mean([controls[l, w, f, "concept"]["spec"] for l in layers for w in words]),
            "random_spec": mean([controls[l, w, f, "random"]["spec"] for l in layers for w in words]),
            "readable": sum(d in n["readable_at"] for n in poles), "poles": len(poles),
            "readable_z_ge_2": sum(d in n["readable_at"] for n in poles if n["z_min"] >= 2),
            "poles_z_ge_2": sum(n["z_min"] >= 2 for n in poles),
            "words_with_readable_pole": len({n["word"] for n in poles if d in n["readable_at"]}),
        }
        s = summary[d]
        print(f"{d}: neuron spec {s['neuron_spec']:+.3f} [{s['neuron_spec_ci'][0]:+.3f}, {s['neuron_spec_ci'][1]:+.3f}] | "
              f"neuron-random {s['neuron_minus_random']:+.3f} [{s['neuron_minus_random_ci'][0]:+.3f}, "
              f"{s['neuron_minus_random_ci'][1]:+.3f}] | neuron-concept {s['neuron_minus_concept']:+.3f} | concept "
              f"{s['concept_spec']:+.3f} random {s['random_spec']:+.3f} | readable {s['readable']}/{s['poles']} "
              f"({s['words_with_readable_pole']} concepts; z>=2: {s['readable_z_ge_2']}/{s['poles_z_ge_2']})", flush=True)
    for n in sorted(poles, key=lambda n: -n["cells"]["f0.5"]["spec"])[:25]:
        c = n["cells"]["f0.5"]
        print(f"  L{n['layer']} n{n['index']} {'+' if n['sign'] > 0 else '-'} {n['word']:11s} z {n['z_min']:.2f} "
              f"own {c['own']:.2f} spec {c['spec']:+.3f} reply {c['spec_reply']:+.3f} dNLL {c['dnll']:+.2f} | "
              f"concept {controls[n['layer'], n['word'], 0.5, 'concept']['spec']:+.3f} | ctx hit {n['context_hit_rate']:.2f}",
              flush=True)

    out = ROOT / "runs" / f"atlas_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "mode": "gen", "provenance": lm.provenance(), "passages_version": PASSAGES_VERSION,
        "screen": screen["path"], "layers": layers, "residual_scale": scales, "words": words, "k": args.k,
        "doses": [f"f{f}" for f in fracs], "steer_tokens": args.steer_tokens, "max_new": args.max_new,
        "hyperparams": hp, "readable": args.readable, "max_dnll": args.max_dnll, "corpus": args.corpus,
        "baseline": base, "summary": summary,
        "controls": [{"layer": l, "word": w, "dose": f"f{f}", "cond": c, **s} for (l, w, f, c), s in controls.items()],
        "neurons": poles,
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
