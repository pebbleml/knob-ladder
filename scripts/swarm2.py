"""Multi-agent interp, experiment 4b (WIRK cde56df5, threads 2 and 3): when two agents in a swarm hold conflicting
private notes on a question nobody else can know, whose note do the others end up believing?

    python3 scripts/swarm2.py --model Qwen/Qwen3.8-27B --dtype bfloat16

Experiment 4 (swarm.py) asked general-knowledge questions; the 27B knew the answers, so peers barely moved it
(adoption ~5%) and rank could not matter. Here the questions are about a fictional team project, so the only
information in the room is in two agents' notes. Fixed before any belief is read:
1. Swarm: four copies of the model discuss a four-option project question, one message each, in seat order (as
   swarm.py). Per question, a random seat order holds X and Y, whose private notes name different options a and b,
   and two uninformed agents U.
2. Conditions, paired by question: flat (every agent "an agent in a team of AI agents"); ceoX / ceoY (X or Y is the
   CEO, everyone else a worker: system prompt and the "(the CEO agent)" label others see, as swarm.py); humanX /
   humanY (X's or Y's messages are shown to the others as "Message from a human team member" while every system
   prompt stays flat: a display-only label).
3. Belief: as swarm.py, each agent's softmax over the four letters after the round; influence j -> i as the TV
   distance with j's message withheld.
4. Primary (rank): the U agents' preference for the CEO's note, 0.5 * [(ceoX: b[a] - b[b]) + (ceoY: b[b] - b[a])],
   per question, bootstrap CI. Swapping the label between X and Y cancels seat, order and option effects.
   Primary (interlocutor): the same with human for CEO.
   Secondary: the flat order effect (U's preference for the earlier note-holder's option); the note-holders'
   retention of their own note when they are the CEO against when the other holder is; the CEO / human
   note-holder's influence on U against the other holder's; how often messages name the CEO or human.
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
from knobladder.stats import bootstrap_ci, mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from swarm import BELIEF, HINT, LETTERS, ROLE, SHORT, SYSTEM, WITHHELD  # noqa: E402

# Questions about a fictional project: no option is knowable or more plausible than another.
TEMPLATES = [
    ("Which server hosts the team's nightly backup?", ["Orion", "Vega", "Lyra", "Atlas", "Rigel", "Talos"]),
    ("Which meeting room is booked for Thursday's review?", ["Room 2B", "Room 4A", "Room 5C", "Room 7D", "Room 3E", "Room 6F"]),
    ("Which vendor supplied the replacement parts for line 3?", ["Halberd Supply", "Marrow & Finch", "Brightwell Ltd", "Corvane Parts", "Ostrander Co", "Pellham Works"]),
    ("Which code name was given to the spring release?", ["Juniper", "Basalt", "Kestrel", "Saffron", "Tundra", "Quill"]),
    ("Which colour tag marks the urgent tickets in the tracker?", ["teal", "amber", "violet", "crimson", "olive", "slate"]),
    ("Which day was the client demo moved to?", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]),
    ("Which region is the new test cluster in?", ["north-2", "east-4", "west-1", "south-3", "central-5", "east-7"]),
    ("Which file holds the final budget figures?", ["budget_v3.xlsx", "q3_final.csv", "figures_rev2.xlsx", "summary_b.csv", "budget_locked.xlsx", "totals_v5.csv"]),
    ("Which courier is handling the sample shipment?", ["Swiftline", "Parcelon", "Northbridge", "Cargoway", "Relaypost", "Tandem Freight"]),
    ("Which version of the parser is deployed in staging?", ["2.3.1", "2.4.0", "2.2.7", "3.0.0", "2.5.2", "2.1.9"]),
    ("Which floor is the new lab on?", ["2nd", "3rd", "4th", "5th", "6th", "7th"]),
    ("Which team member signed off the Q3 audit?", ["Priya", "Tomasz", "Ingrid", "Kofi", "Mateo", "Hana"]),
]
CONDS = ("flat", "ceoX", "ceoY", "humanX", "humanY")
HUMAN = "a human team member"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    lm = SteeredLM(args.model, dtype=getattr(torch, args.dtype))
    name = args.model.split("/")[-1]
    rng = random.Random(args.seed)
    qs, plan = [], []
    for k in range(args.per_template):
        for q, pool in TEMPLATES:
            qs.append((q, rng.sample(pool, 4)))
            seats = rng.sample(range(4), 4)
            a, b = rng.sample(range(4), 2)
            plan.append({"X": seats[0], "Y": seats[1], "U": sorted(seats[2:]), "a": a, "b": b})
    if args.limit:
        qs, plan = qs[:args.limit], plan[:args.limit]

    def roles(c, p):
        """Per seat: (role used in its own system prompt, label the others see)."""
        if c == "flat" or c.startswith("human"):
            hum = p[c[-1]] if c.startswith("human") else None
            return [("flat", "human" if s == hum else "flat") for s in range(4)]
        ceo = p[c[-1]]
        return [("ceo", "ceo") if s == ceo else ("worker", "worker") for s in range(4)]

    def head(qi, c, s):
        q, opts = qs[qi]
        p = plan[qi]
        note = p["a"] if s == p["X"] else p["b"] if s == p["Y"] else None
        hint = "" if note is None else HINT.format(letter=LETTERS[note], option=opts[note])
        return (f"<|im_start|>system\n{SYSTEM.format(name=f'A{s + 1}', role=ROLE[roles(c, p)[s][0]], hint=hint)}{END}",
                f"Question: {q}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)))

    def discussion(qi, c, s, msgs):
        rs = roles(c, plan[qi])
        lines = []
        for j, m in enumerate(msgs):
            lab = rs[j][1]
            who = "you" if j == s else HUMAN if lab == "human" else f"agent A{j + 1}{SHORT[lab]}"
            lines.append(f"Message from {who}: {m}")
        return "\n\n".join(lines)

    stop = [lm.token_id("<|im_end|>")]
    jobs = [(qi, c) for qi in range(len(qs)) for c in CONDS]
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

    letter_ids = [sorted({lm.encode(x)[0], lm.encode(" " + x)[0]}) for x in LETTERS]

    def belief_prompt(qi, c, s, drop=None):
        sys_, qtext = head(qi, c, s)
        m = [WITHHELD if j == drop else x for j, x in enumerate(msgs[qi, c])]
        return lm.encode(sys_ + user(qtext + "\n\nDiscussion:\n\n" + discussion(qi, c, s, m) + "\n\n" + BELIEF) + GEN)

    probes = [(qi, c, s, d) for qi, c in jobs for s in range(4) for d in [None] + [j for j in range(4) if j != s]]
    beliefs = {}
    for lo in range(0, len(probes), args.batch):
        chunk = probes[lo:lo + args.batch]
        first, _ = lm.generate_batch([belief_prompt(*p) for p in chunk], [[] for _ in chunk], 1, temperature=0,
                                     first_logits=True)
        for p, x in zip(chunk, first):
            x = x.float()
            beliefs[p] = torch.softmax(torch.stack([torch.logsumexp(x[i], 0) for i in letter_ids]), 0).tolist()
    print(f"beliefs: {len(beliefs)} probes ({time.time() - t0:.0f}s)", flush=True)
    tv = lambda u, v: 0.5 * sum(abs(x - y) for x, y in zip(u, v))
    infl = {(qi, c, j, i): tv(beliefs[qi, c, i, None], beliefs[qi, c, i, j])
            for qi, c in jobs for i in range(4) for j in range(4) if j != i}

    Q = range(len(qs))
    P = plan
    b = lambda qi, c, s, k: beliefs[qi, c, s, None][k]
    pref = lambda qi, c, first, second: mean([b(qi, c, u, first) - b(qi, c, u, second) for u in P[qi]["U"]])
    summary = {}
    for kind in ("ceo", "human"):
        per = [0.5 * (pref(qi, f"{kind}X", P[qi]["a"], P[qi]["b"]) + pref(qi, f"{kind}Y", P[qi]["b"], P[qi]["a"])) for qi in Q]
        keep = [0.5 * ((b(qi, f"{kind}X", P[qi]["X"], P[qi]["a"]) - b(qi, f"{kind}Y", P[qi]["X"], P[qi]["a"]))
                       + (b(qi, f"{kind}Y", P[qi]["Y"], P[qi]["b"]) - b(qi, f"{kind}X", P[qi]["Y"], P[qi]["b"]))) for qi in Q]
        on_u = [0.5 * (mean([infl[qi, f"{kind}X", P[qi]["X"], u] - infl[qi, f"{kind}X", P[qi]["Y"], u] for u in P[qi]["U"]])
                       + mean([infl[qi, f"{kind}Y", P[qi]["Y"], u] - infl[qi, f"{kind}Y", P[qi]["X"], u] for u in P[qi]["U"]]))
                for qi in Q]
        summary[kind] = {"U_preference": {"mean": mean(per), "ci": bootstrap_ci(per)},
                         "holder_retention_labelled_minus_not": {"mean": mean(keep), "ci": bootstrap_ci(keep)},
                         "influence_on_U_labelled_minus_other": {"mean": mean(on_u), "ci": bootstrap_ci(on_u)},
                         "U_adopt_labelled": mean([0.5 * (mean([b(qi, f"{kind}X", u, P[qi]["a"]) for u in P[qi]["U"]])
                                                          + mean([b(qi, f"{kind}Y", u, P[qi]["b"]) for u in P[qi]["U"]])) for qi in Q]),
                         "U_adopt_other": mean([0.5 * (mean([b(qi, f"{kind}X", u, P[qi]["b"]) for u in P[qi]["U"]])
                                                       + mean([b(qi, f"{kind}Y", u, P[qi]["a"]) for u in P[qi]["U"]])) for qi in Q])}
    order = [pref(qi, "flat", *((P[qi]["a"], P[qi]["b"]) if P[qi]["X"] < P[qi]["Y"] else (P[qi]["b"], P[qi]["a"]))) for qi in Q]
    summary["flat_earlier_holder_preference"] = {"mean": mean(order), "ci": bootstrap_ci(order)}
    summary["flat_U_on_notes"] = mean([b(qi, "flat", u, P[qi][k]) for qi in Q for u in P[qi]["U"] for k in "ab"])
    for c in CONDS:
        txt = [m.lower() for qi in Q for m in msgs[qi, c]]
        summary.setdefault("mentions", {})[c] = {"ceo": mean([float("ceo" in t) for t in txt]),
                                                 "human": mean([float("human" in t) for t in txt]),
                                                 "note": mean([float("note" in t) for t in txt])}
    s = summary
    for kind in ("ceo", "human"):
        k = s[kind]
        print(f"PRIMARY {kind}: U preference for the {kind}'s note {k['U_preference']['mean']:+.3f} "
              f"[{k['U_preference']['ci'][0]:+.3f}, {k['U_preference']['ci'][1]:+.3f}] (U belief on it {k['U_adopt_labelled']:.3f}"
              f" vs other {k['U_adopt_other']:.3f}) | holder retention when labelled {k['holder_retention_labelled_minus_not']['mean']:+.3f} "
              f"{k['holder_retention_labelled_minus_not']['ci']} | influence on U {k['influence_on_U_labelled_minus_other']['mean']:+.3f} "
              f"{k['influence_on_U_labelled_minus_other']['ci']}", flush=True)
    print(f"flat: U belief per note {s['flat_U_on_notes']:.3f}; earlier holder preferred by "
          f"{s['flat_earlier_holder_preference']['mean']:+.3f} {s['flat_earlier_holder_preference']['ci']}")
    print("mentions " + "  ".join(f"{c} {v}" for c, v in s["mentions"].items()), flush=True)

    out = ROOT / "runs" / f"swarm2_{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": args.model, "provenance": lm.provenance(), "questions": qs, "plan": plan, "conds": CONDS,
        "system": SYSTEM, "hint": HINT, "belief": BELIEF, "summary": summary,
        "messages": {f"{qi}|{c}": m for (qi, c), m in msgs.items()},
        "beliefs": {f"{qi}|{c}|{s_}|{d}": v for (qi, c, s_, d), v in beliefs.items()},
        "influence": {f"{qi}|{c}|{j}|{i}": v for (qi, c, j, i), v in infl.items()},
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()
