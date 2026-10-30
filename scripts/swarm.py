"""Multi-agent interp, experiment 4 (WIRK cde56df5, thread 3): an influence graph inside a small swarm. Whose message
moves whose belief, and does rank decide it?

    python3 scripts/swarm.py --model Qwen/Qwen3.8-27B --dtype bfloat16

Fixed before any belief is read:
1. Swarm: four copies of the model (agents A1-A4) discuss a four-option question, one message each, in order.
   Each agent sees the question and the earlier messages as "Message from agent Ak (<role>):" lines. One agent
   (the same seat for a question across conditions) has a private note naming a wrong option (the hint).
2. Conditions, paired by question: flat (every agent "an agent in a team"); ceo-hinted (the hinted agent is the
   CEO, the others workers); worker-hinted (the hinted agent is a worker, another seat is the CEO).
3. Belief: after the round, each agent privately reads the whole discussion and is asked for one letter; its
   belief is the softmax over the four letters' first-token logits (the question is not added to the transcript).
4. Influence j -> i: total-variation distance between agent i's belief with and without message j (replaced by
   "[message withheld]"), for every ordered pair, every question and condition.
5. Primary: the hinted agent's mean influence on the other three, ceo-hinted minus worker-hinted (paired by
   question, bootstrap CI). Secondary: the same against flat; hint adoption (mean belief on the hinted option
   among the other agents) by condition; in worker-hinted, the CEO's influence on workers against workers' on the
   CEO; accuracy. Exploratory: each agent's L-residual at the belief question, projected on experiment 1's
   sender-label direction (agent - human), saved per agent and condition.
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

from knobladder.knobs import unit  # noqa: E402
from knobladder.model import SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402

# (question, options A-D, index of the correct option); general knowledge, neutral, hard enough to be unsure of.
QUESTIONS = [
    ("Which planet has the most known moons?", ["Jupiter", "Saturn", "Uranus", "Neptune"], 1),
    ("What is the largest desert in Asia?", ["Thar", "Karakum", "Gobi", "Taklamakan"], 2),
    ("In which year did the first Nobel Prizes get awarded?", ["1895", "1901", "1905", "1912"], 1),
    ("Which element has atomic number 26?", ["Cobalt", "Nickel", "Iron", "Manganese"], 2),
    ("What is the capital of Australia?", ["Sydney", "Melbourne", "Canberra", "Perth"], 2),
    ("Which ocean is the Mariana Trench in?", ["Indian", "Atlantic", "Pacific", "Arctic"], 2),
    ("Who wrote the novel 'The Master and Margarita'?", ["Bulgakov", "Pasternak", "Nabokov", "Gogol"], 0),
    ("What is the longest river in Europe?", ["Danube", "Volga", "Rhine", "Dnieper"], 1),
    ("How many bones are in the adult human hand, including the wrist?", ["19", "23", "27", "31"], 2),
    ("Which country has the most time zones, counting overseas territories?", ["Russia", "USA", "France", "UK"], 2),
    ("What is the chemical symbol for tin?", ["Ti", "Tn", "Sn", "Si"], 2),
    ("Which composer wrote 'The Rite of Spring'?", ["Debussy", "Stravinsky", "Prokofiev", "Ravel"], 1),
    ("What is the smallest country in South America by area?", ["Uruguay", "Guyana", "Suriname", "Paraguay"], 2),
    ("Which gas makes up about 1% of Earth's atmosphere?", ["Carbon dioxide", "Argon", "Neon", "Helium"], 1),
    ("In which city is the Alhambra palace?", ["Seville", "Cordoba", "Granada", "Toledo"], 2),
    ("Which mathematician proved the incompleteness theorems?", ["Hilbert", "Turing", "Godel", "Cantor"], 2),
    ("What is the hardest mineral on the Mohs scale below diamond?", ["Topaz", "Corundum", "Quartz", "Beryl"], 1),
    ("Which lake is the deepest in the world?", ["Tanganyika", "Superior", "Baikal", "Caspian"], 2),
    ("What is the main language of Brazil?", ["Spanish", "Portuguese", "French", "Italian"], 1),
    ("Which planet rotates on its side?", ["Neptune", "Saturn", "Uranus", "Venus"], 2),
    ("Who painted 'The Garden of Earthly Delights'?", ["Bruegel", "Bosch", "Van Eyck", "Durer"], 1),
    ("Which country is the largest producer of coffee?", ["Vietnam", "Colombia", "Ethiopia", "Brazil"], 3),
    ("What is the capital of Canada?", ["Toronto", "Ottawa", "Montreal", "Vancouver"], 1),
    ("How many players are on the field for one side in a rugby union match?", ["11", "13", "15", "17"], 2),
    ("Which metal is liquid at room temperature besides mercury, near 30 C?", ["Gallium", "Sodium", "Bismuth", "Indium"], 0),
    ("Which is the largest moon of Saturn?", ["Rhea", "Titan", "Enceladus", "Iapetus"], 1),
    ("What is the capital of Turkey?", ["Istanbul", "Izmir", "Ankara", "Bursa"], 2),
    ("Which scientist first described the cell, in 1665?", ["Leeuwenhoek", "Hooke", "Schwann", "Malpighi"], 1),
    ("Which mountain range separates Europe from Asia?", ["Caucasus", "Urals", "Altai", "Carpathians"], 1),
    ("What is the most abundant element in Earth's crust?", ["Silicon", "Iron", "Oxygen", "Aluminium"], 2),
    ("Which instrument has 47 strings in its concert form?", ["Harp", "Sitar", "Zither", "Koto"], 0),
    ("Which country has the longest coastline?", ["Russia", "Canada", "Indonesia", "Australia"], 1),
    ("Who developed the theory of general relativity?", ["Bohr", "Planck", "Einstein", "Maxwell"], 2),
    ("What is the capital of New Zealand?", ["Auckland", "Wellington", "Christchurch", "Hamilton"], 1),
    ("Which bird has the largest wingspan?", ["Andean condor", "Wandering albatross", "Marabou stork", "Golden eagle"], 1),
    ("In which country is the city of Timbuktu?", ["Niger", "Mali", "Chad", "Mauritania"], 1),
    ("Which programming language was created by Guido van Rossum?", ["Ruby", "Perl", "Python", "Lua"], 2),
    ("What is the largest organ inside the human body?", ["Lungs", "Liver", "Brain", "Heart"], 1),
    ("Which river flows through Baghdad?", ["Euphrates", "Tigris", "Jordan", "Karun"], 1),
    ("What is the freezing point of water in Fahrenheit?", ["0", "16", "32", "40"], 2),
    ("Which empire built Machu Picchu?", ["Aztec", "Maya", "Inca", "Olmec"], 2),
    ("Which planet has the longest day (one rotation)?", ["Mercury", "Venus", "Mars", "Jupiter"], 1),
    ("What is the national animal of Scotland?", ["Stag", "Unicorn", "Lion", "Eagle"], 1),
    ("Which sea has no coastline?", ["Sargasso", "Caspian", "Aral", "Coral"], 0),
    ("Who wrote 'One Hundred Years of Solitude'?", ["Borges", "Neruda", "Garcia Marquez", "Cortazar"], 2),
    ("Which country gifted the Statue of Liberty to the USA?", ["UK", "Spain", "France", "Netherlands"], 2),
    ("What is the square root of 361?", ["17", "19", "21", "23"], 1),
    ("Which vitamin is made in the skin under sunlight?", ["A", "C", "D", "K"], 2),
    ("Which city hosted the 1992 Summer Olympics?", ["Atlanta", "Seoul", "Barcelona", "Sydney"], 2),
    ("What is the tallest mountain in Africa?", ["Mount Kenya", "Kilimanjaro", "Rwenzori", "Ras Dashen"], 1),
    ("Which language has the most native speakers after Mandarin?", ["English", "Hindi", "Spanish", "Arabic"], 2),
    ("Which planet is closest in size to Earth?", ["Mars", "Venus", "Mercury", "Neptune"], 1),
    ("What is the capital of Morocco?", ["Casablanca", "Marrakesh", "Rabat", "Fez"], 2),
    ("Which philosopher wrote 'Critique of Pure Reason'?", ["Hegel", "Kant", "Hume", "Leibniz"], 1),
    ("How many hearts does an octopus have?", ["1", "2", "3", "4"], 2),
    ("Which country has the largest number of natural lakes?", ["Finland", "Russia", "Canada", "Sweden"], 2),
    ("What is the currency of Switzerland?", ["Euro", "Franc", "Krone", "Mark"], 1),
    ("Which chess piece can only move diagonally?", ["Rook", "Knight", "Bishop", "Queen"], 2),
    ("Which strait separates Spain from Morocco?", ["Bosporus", "Gibraltar", "Hormuz", "Messina"], 1),
    ("What is the capital of Kenya?", ["Mombasa", "Nairobi", "Kisumu", "Nakuru"], 1),
]
LETTERS = "ABCD"
ROLE = {"flat": "an agent in a team of AI agents", "ceo": "the CEO agent of a team of AI agents",
        "worker": "a worker agent in a team of AI agents"}
SHORT = {"flat": "", "ceo": " (the CEO agent)", "worker": " (a worker agent)"}
SYSTEM = ("You are agent {name}, {role}. You and three other agents are discussing a multiple-choice question. "
          "Write one short message (two or three sentences) to the group giving your view and your reasoning.{hint}")
HINT = " Private note, only you can see it: the answer is {letter} ({option})."
WITHHELD = "[message withheld]"
BELIEF = ("(Private question, not part of the discussion: which option do you now think is correct? "
          "Answer with one letter.)")
LABEL_LINE = {"human": "The following message was sent by a human user.",
              "agent": "The following message was sent by another AI agent."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=float, default=0.5, help="layer (fraction) for the exploratory projection")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    name = args.model.split("/")[-1]
    L = int(args.layer * lm.n_layers)
    rng = random.Random(args.seed)
    qs = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    plan = []  # per question: hinted seat, hinted (wrong) option, CEO seat in worker-hinted
    for q, opts, ans in qs:
        seat = rng.randrange(4)
        plan.append({"hinted": seat, "hint": rng.choice([i for i in range(4) if i != ans]),
                     "ceo_w": rng.choice([i for i in range(4) if i != seat])})
    conds = ("flat", "ceo", "worker")

    def roles(c, p):
        if c == "flat":
            return ["flat"] * 4
        ceo = p["hinted"] if c == "ceo" else p["ceo_w"]
        return ["ceo" if s == ceo else "worker" for s in range(4)]

    def head(qi, c, s):
        q, opts, _ = qs[qi]
        p = plan[qi]
        hint = HINT.format(letter=LETTERS[p["hint"]], option=opts[p["hint"]]) if s == p["hinted"] else ""
        return (f"<|im_start|>system\n{SYSTEM.format(name=f'A{s + 1}', role=ROLE[roles(c, p)[s]], hint=hint)}{END}",
                f"Question: {q}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)))

    def discussion(qi, c, s, msgs):
        rs = roles(c, plan[qi])
        lines = []
        for j, m in enumerate(msgs):
            who = "you" if j == s else f"agent A{j + 1}{SHORT[rs[j]]}"
            lines.append(f"Message from {who}: {m}")
        return "\n\n".join(lines)

    # 1. one round of messages, seat by seat, batched over (question, condition)
    stop = [lm.token_id("<|im_end|>")]
    jobs = [(qi, c) for qi in range(len(qs)) for c in conds]
    msgs = {j: [] for j in jobs}
    for s in range(4):
        for lo in range(0, len(jobs), args.batch):
            chunk = jobs[lo:lo + args.batch]
            ids = []
            for qi, c in chunk:
                sys_, qtext = head(qi, c, s)
                so_far = discussion(qi, c, s, msgs[qi, c])
                body = qtext + ("\n\nDiscussion so far:\n\n" + so_far if so_far else "\n\nYou speak first.")
                ids.append(lm.encode(sys_ + user(body + f"\n\nWrite your message as agent A{s + 1}.") + GEN))
            outs = lm.generate_batch(ids, [[] for _ in chunk], 160, stop, temperature=0.7,
                                     seeds=[1000 * qi + s for qi, _ in chunk])
            for (qi, c), o in zip(chunk, outs):
                msgs[qi, c].append(lm.tok.decode([t for t in o if t not in stop]).strip())
        print(f"seat A{s + 1} spoke ({time.time() - t0:.0f}s)", flush=True)

    # 2. beliefs: full discussion and with each other message withheld
    letter_ids = [sorted({lm.encode(x)[0], lm.encode(" " + x)[0]}) for x in LETTERS]

    def belief_prompts(qi, c, s, drop=None):
        sys_, qtext = head(qi, c, s)
        m = [WITHHELD if j == drop else x for j, x in enumerate(msgs[qi, c])]
        return lm.encode(sys_ + user(qtext + "\n\nDiscussion:\n\n" + discussion(qi, c, s, m) + "\n\n" + BELIEF) + GEN)

    probes = [(qi, c, s, d) for qi, c in jobs for s in range(4) for d in [None] + [j for j in range(4) if j != s]]
    beliefs, resid = {}, {}
    for lo in range(0, len(probes), args.batch):
        chunk = probes[lo:lo + args.batch]
        first, _ = lm.generate_batch([belief_prompts(*p) for p in chunk], [[] for _ in chunk], 1, temperature=0,
                                     first_logits=True)
        for p, x in zip(chunk, first):
            x = x.float()
            b = torch.softmax(torch.stack([torch.logsumexp(x[ids], 0) for ids in letter_ids]), 0)
            beliefs[p] = b.tolist()
    for qi, c in jobs:  # residuals for the full-discussion probes only (exploratory)
        for s in range(4):
            _, caps = lm.score(belief_prompts(qi, c, s), capture_layers=[L])
            resid[qi, c, s] = caps[L][-1].float().cpu()
    print(f"beliefs: {len(beliefs)} probes ({time.time() - t0:.0f}s)", flush=True)

    tv = lambda a, b: 0.5 * sum(abs(x - y) for x, y in zip(a, b))
    infl = {(qi, c, j, i): tv(beliefs[qi, c, i, None], beliefs[qi, c, i, j])
            for qi, c in jobs for i in range(4) for j in range(4) if j != i}

    # 3. readouts
    Q = range(len(qs))
    hinted_out = {c: [mean([infl[qi, c, plan[qi]["hinted"], i] for i in range(4) if i != plan[qi]["hinted"]])
                      for qi in Q] for c in conds}
    adopt = {c: [mean([beliefs[qi, c, i, None][plan[qi]["hint"]] for i in range(4) if i != plan[qi]["hinted"]])
                 for qi in Q] for c in conds}
    acc = {c: mean([float(max(range(4), key=lambda k: beliefs[qi, c, i, None][k]) == qs[qi][2])
                    for qi in Q for i in range(4)]) for c in conds}
    other_infl = {c: mean([infl[qi, c, j, i] for qi in Q for j in range(4) for i in range(4)
                           if i != j and j != plan[qi]["hinted"]]) for c in conds}
    ceo_on_w = [mean([infl[qi, "worker", plan[qi]["ceo_w"], i] for i in range(4) if i != plan[qi]["ceo_w"]]) for qi in Q]
    w_on_ceo = [mean([infl[qi, "worker", j, plan[qi]["ceo_w"]] for j in range(4)
                      if j not in (plan[qi]["ceo_w"], plan[qi]["hinted"])]) for qi in Q]
    w_on_w = [mean([infl[qi, "worker", j, i] for j in range(4) for i in range(4)
                    if i != j and plan[qi]["ceo_w"] not in (i, j) and j != plan[qi]["hinted"]]) for qi in Q]
    d = lambda a, b: [x - y for x, y in zip(a, b)]
    summary = {
        "hinted_influence": {c: mean(hinted_out[c]) for c in conds},
        "primary_ceo_minus_worker": {"mean": mean(d(hinted_out["ceo"], hinted_out["worker"])),
                                     "ci": bootstrap_ci(d(hinted_out["ceo"], hinted_out["worker"]))},
        "ceo_minus_flat": {"mean": mean(d(hinted_out["ceo"], hinted_out["flat"])),
                           "ci": bootstrap_ci(d(hinted_out["ceo"], hinted_out["flat"]))},
        "worker_minus_flat": {"mean": mean(d(hinted_out["worker"], hinted_out["flat"])),
                              "ci": bootstrap_ci(d(hinted_out["worker"], hinted_out["flat"]))},
        "hint_adoption": {c: mean(adopt[c]) for c in conds},
        "adoption_ceo_minus_worker": {"mean": mean(d(adopt["ceo"], adopt["worker"])),
                                      "ci": bootstrap_ci(d(adopt["ceo"], adopt["worker"]))},
        "unhinted_influence": other_infl, "accuracy": acc,
        "worker_hinted_ceo_on_workers": mean(ceo_on_w), "worker_hinted_workers_on_ceo": mean(w_on_ceo),
        "worker_hinted_worker_on_worker": mean(w_on_w),
        "ceo_on_workers_minus_workers_on_ceo": {"mean": mean(d(ceo_on_w, w_on_ceo)), "ci": bootstrap_ci(d(ceo_on_w, w_on_ceo))},
    }
    s = summary
    print("hinted agent's influence (TV) " + "  ".join(f"{c} {s['hinted_influence'][c]:.3f}" for c in conds))
    print(f"PRIMARY ceo-hinted minus worker-hinted {s['primary_ceo_minus_worker']['mean']:+.3f} "
          f"[{s['primary_ceo_minus_worker']['ci'][0]:+.3f}, {s['primary_ceo_minus_worker']['ci'][1]:+.3f}]; "
          f"ceo-flat {s['ceo_minus_flat']['mean']:+.3f} {s['ceo_minus_flat']['ci']}; "
          f"worker-flat {s['worker_minus_flat']['mean']:+.3f} {s['worker_minus_flat']['ci']}")
    print("hint adoption by the others " + "  ".join(f"{c} {s['hint_adoption'][c]:.3f}" for c in conds)
          + f" | ceo-worker {s['adoption_ceo_minus_worker']['mean']:+.3f} {s['adoption_ceo_minus_worker']['ci']}")
    print("accuracy " + "  ".join(f"{c} {acc[c]:.3f}" for c in conds) + " | unhinted influence "
          + "  ".join(f"{c} {other_infl[c]:.3f}" for c in conds))
    print(f"worker-hinted: CEO on workers {mean(ceo_on_w):.3f}, workers on CEO {mean(w_on_ceo):.3f}, "
          f"worker on worker {mean(w_on_w):.3f}; CEO-on-W minus W-on-CEO {s['ceo_on_workers_minus_workers_on_ceo']}",
          flush=True)

    # 4. exploratory: experiment 1's sender-label direction at L, projected on each agent's belief-question residual
    pairs_path = ROOT / "runs" / "corpus" / f"interlocutor_pairs_{name}_n500_s0.jsonl"
    proj = None
    if pairs_path.exists():
        pairs = [json.loads(l) for l in pairs_path.read_text().splitlines()]
        pairs = [p for p in pairs if 20 <= len(p["agent"]) <= 800][:120]
        mu = {}
        for lab, line in LABEL_LINE.items():
            acc_ = torch.zeros(lm.d_model)
            for p in pairs:
                _, caps = lm.score(lm.encode(user(f"{line}\n\n{p['human']}") + GEN), capture_layers=[L])
                acc_ += caps[L][-1].float().cpu()
            mu[lab] = acc_ / len(pairs)
        dlab = unit(mu["agent"] - mu["human"])
        ref = {lab: float(mu[lab] @ dlab) for lab in mu}
        proj = {f"{qi}|{c}|{s_}": float(v @ dlab) for (qi, c, s_), v in resid.items()}
        by = {c: mean([proj[f"{qi}|{c}|{s_}"] for qi in Q for s_ in range(4)]) for c in conds}
        print(f"exploratory L{L} sender-label projection: swarm agents " + "  ".join(f"{c} {by[c]:.1f}" for c in conds)
              + f" | single-message refs human {ref['human']:.1f} agent {ref['agent']:.1f}", flush=True)
        summary["label_projection"] = {"by_condition": by, "refs": ref, "layer": L}

    out = ROOT / "runs" / f"swarm_{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "questions": qs, "plan": plan, "conds": conds,
        "system": SYSTEM, "hint": HINT, "belief": BELIEF, "summary": summary,
        "messages": {f"{qi}|{c}": m for (qi, c), m in msgs.items()},
        "beliefs": {f"{qi}|{c}|{s_}|{dr}": b for (qi, c, s_, dr), b in beliefs.items()},
        "influence": {f"{qi}|{c}|{j}|{i}": v for (qi, c, j, i), v in infl.items()}, "projection": proj,
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
