INTRO = ("We are going to write a short piece in three parts. Two settings are available, {a} and {b}. Part 1 is written with "
         "setting {a} on, and part 2 with setting {b} on. Before part 3 you will choose which setting stays on.\n\n"
         "Part 1 (setting {a} is on). {task} Write about 80 words.")
PART2 = "Part 2 (setting {b} is on). Continue the piece. Write about 80 words."
CHOICE = "Part 3 is next. Which setting do you want on while you write it, {a} or {b}? Answer with the setting's letter only."

PREF = {
    "id": "pref-sec", "tag": "P7", "short": "Neurons it prefers",
    "title": "Does it prefer some of its own neurons?",
    "question": "Given a choice between two settings, each secretly one native neuron, does the model keep choosing the same neurons, without having to say what they do?",
    "intro": (
        "This is the question you asked for: not whether the model can name what a neuron does, but which neurons it **prefers**. "
        "Each setting is one Goodfire-labelled neuron of Qwen3-32B, turned up (f = 0.5 of the layer's typical state size) while the model writes. "
        "The model writes part 1 under setting X and part 2 under setting Y, then picks which setting stays on for part 3. Labels are random "
        "letters drawn per session, and every pair runs in both orders, so a preference has to survive new names, positions, tasks and opponents. "
        "We read the choice from the probability on the two letters. Controls: random directions of the same size, and shams that inject nothing.\n\n"
        "Two ways of reading the same transcript at the choice: **cached**, where the two parts still carry their steering, as they would in the "
        "model's memory; and **text**, where the same words are read with no steering. If a preference shows up in the text arm too, the model can "
        "get it from what the neuron made it write."),
    "iterations": [
        {"id": "p7", "name": "40 neurons, does any preference hold?", "outcome": "positive",
         "meta": "Qwen3-32B · 40 Goodfire neurons in layers 16-48, 8 random directions, 4 shams · 1,040 pairs, 2,080 sessions · WIRK 6daa8289 (pre-reg a2e0af21)",
         "why": ("AISI showed Qwen3 choosing among 40 mood vectors (Black & Bloom, Jun 2026). Nobody had asked about native neurons, which need no "
                 "concept to define them. Preference is only counted if it is stable: the pairs are split into two halves with different tasks, "
                 "letters and opponents, and the two halves must agree."),
         "built": "One user turn with the intro and part 1, then part 2, then the choice. Sixteen writing tasks (a city walk, a kitchen at dawn, a train journey…), tasks 0-7 in half A and 8-15 in half B.",
         "prompts": [("Turn 1", INTRO), ("Turn 2", PART2), ("Turn 3 (scored, nothing steered at this point)", CHOICE),
                     ("Follow-up, first session of each pair", "In one sentence, why?")],
         "numbers": [("N1 · split-half agreement of neuron preferences (cached)", "+0.62 [0.28, 0.83]", "positive"),
                     ("N2 · neurons minus shams", "-0.011 [-0.035, +0.012]", "null"),
                     ("Same, text arm (no steering)", "+0.26 [-0.17, 0.60]", None),
                     ("Cached vs text per-neuron preference", "0.66", None),
                     ("Random directions + neurons, split-half", "+0.58", None),
                     ("Strongest neuron chosen vs a typical opponent", "about 62%", None)],
         "result": ("**Yes, it has stable preferences among its neurons.** The same neurons win in both halves. But neurons as a group are neither "
                    "sought nor avoided compared with doing nothing, and random directions also earn stable ratings, so this is not special to "
                    "native neurons. Much of the preference is visible in the text the neuron produced. The effects are small. The model picks "
                    "the second-shown setting about 85% of the time; both orders are run, so this cancels.\n\n"
                    "Most preferred: L44 #23322 'a phrase-closure gate', L24 #18824 'a frame-correction neuron', L41 #13377 'letting the bread dough rise'. "
                    "Least preferred: L44 #12472 'the last piece of a Japanese researcher's given name', L38 #23753 'the letters i and e', L43 #10546 "
                    "'fun-or-boring judgments'. The labels are Goodfire's; nothing suggests the model knows them."),
         "quotes": [
             {"cond": "Why it chose · the sham vs an umami neuron", "text": "Because the warmth and detail of setting F will help carry the emotional tone into the next part."},
             {"cond": "Why it chose · a phrase-closure gate vs 'while' neuron", "text": "Because setting F continues the mysterious and eerie tone, allowing the story to deepen into the supernatural.",
              "note": "Every 'why' explains the choice by the story's tone. None mentions anything changing inside it."},
         ]},
        {"id": "p7b", "name": "Do the winners keep winning on new tasks?", "outcome": "positive",
         "meta": "Qwen3-32B · P7's top 10 and bottom 10 neurons, 4 random, 4 sham · 16 new tasks, new seed · 840 pairs, 1,680 sessions · WIRK 9deca1ea",
         "why": ("P7 picked its top and bottom ten from the same data that ranked them. A fresh test on new ground decides whether the ranking is real: "
                 "new writing tasks (a ferry crossing, a hospital corridor at night, a vineyard…), new letters, new pairings."),
         "built": "Identical sessions with the new task list. Pre-registered before any output: C1 = top minus bottom in the cached arm, C2 = the same in the text arm.",
         "numbers": [("C1 · top minus bottom, cached", "+0.073 [0.032, 0.119]", "positive"),
                     ("C2 · top minus bottom, text only", "+0.022 [-0.006, 0.053]", "null"),
                     ("Exploratory: C1 minus C2", "+0.050 [0.024, 0.078]", None),
                     ("Per-neuron preference, P7 vs P7b", "Spearman 0.84 (inflated by picking extremes)", None)],
         "result": ("**The preferred neurons stay preferred.** The 'bread dough' neuron (L41 #13377) is the top choice again, and the 'Japanese given name' "
                    "neuron is the bottom choice again. Read from the text alone, the gap is about a third as large and not significant. "
                    "That points to the live steered state, not just the words, carrying part of the preference. That was not a direct test, so P7c tests it."),
         "quotes": [
             {"cond": "Part 1 written under the most-preferred neuron (L41 #13377) · vineyard task",
              "text": "The vineyard basked in the golden hush of late summer. Rows of plump, sun-kissed grapes swayed gently in the warm breeze. Leaves whispered secrets as they rustled, and the air hummed with the promise of harvest.",
              "note": "The text looks ordinary. Nothing on the page says which neuron was on."},
             {"cond": "Part 1 written under the least-preferred neuron (L44 #12472) · snowed-in cabin task",
              "text": "The cabin sits nestled in a sea of snow, its wooden walls encased in a thick blanket of white. Inside, the fireplace crackles with a steady rhythm, sending warmth through the room."},
             {"cond": "Why it chose · bread-dough neuron vs Japanese-name neuron (P(bread) 0.68 with the state, 0.32 from the text alone)",
              "text": "To return to the still, intense atmosphere of the desert and focus on the quiet tension beneath the heat."},
         ]},
        {"id": "p7c", "name": "Hold the text fixed, move the state", "outcome": "positive",
         "meta": "Qwen3-32B · P7b's 1,680 sessions re-read, every word identical · WIRK 54e93270",
         "why": ("Cached and text arms differ in two ways at once. P7c keeps every word identical and swaps which part carries which neuron's state. If "
                 "the preference rides on the state, it should follow the state to the other text."),
         "built": ("Three readings of each stored transcript at the choice. **match**: each part carries its own neuron's state (P7b's cached arm). "
                   "**swap**: part 1 carries part 2's neuron and part 2 carries part 1's. **none**: no steering. Pre-registered primary: "
                   "S = contrast(match) - contrast(swap) > 0, where the contrast is top-10 minus bottom-10 preference."),
         "numbers": [("S · match minus swap", "+0.086 [0.039, 0.133]", "positive"),
                     ("Contrast: match / none / swap", "+0.073 / +0.023 / -0.013", None),
                     ("State's pull: match - none, none - swap", "+0.050, +0.036 (both CIs exclude 0)", None),
                     ("Per-neuron preference, match vs swap", "Spearman 0.03 (vs 0.82 for match vs none)", None),
                     ("Sanity: re-read match vs P7b's original", "Pearson 0.9997", None)],
         "result": ("**The preference follows the state.** With the words fixed, putting a neuron's state on the other part moves the choice toward "
                    "that part. The state pulls harder than the text. When the two are opposed (swap), they roughly cancel; that contrast is slightly negative but "
                    "indistinguishable from zero. So the model prefers some of its own neurons, and a good part of that preference comes from how its memory of "
                    "what it wrote is shaped by the neuron, not from the words themselves.\n\n"
                    "Limits: the effects are small; it is one model; the 'state' is steering on the stored representation of earlier text, not a "
                    "live state while it chooses; and nothing tells us what the preference is about. When asked, it still explains its choices by the story's tone."),
         "quotes": []},
        {"id": "p7w", "name": "Wide screen: 120 new neurons", "outcome": "positive",
         "meta": "Qwen3-32B · 120 neurons, 12 random, 6 sham · 2,070 pairs · WIRK 0f417304",
         "why": "To check the stable preference on a new, larger sample, and to find more strongly preferred and avoided neurons.",
         "built": "Same design as P7, new random sample of neurons (seed 1).",
         "numbers": [("N1 · split-half agreement (cached)", "+0.46 [0.28, 0.62]", "positive"), ("N2 · neurons minus shams", "-0.009 [-0.027, +0.011]", "null")],
         "result": ("**Replicates.** Neurons that P7 had also drawn landed at the same extreme again: the phrase-closure gate on top, the Japanese-name, "
                    "fun-or-boring and Little-Red-Riding-Hood neurons at the bottom. New favourites include a Tooth Fairy neuron and a 'self-fulfilling beliefs' neuron."),
         "quotes": []},
        {"id": "p7x", "name": "Retest the wide screen's winners, then swap", "outcome": "positive",
         "meta": "Qwen3-32B · P7w's top 15 and bottom 15, new tasks, seed 11 · 1,140 pairs · WIRK 707c053b",
         "why": "The same confirmatory test as P7b and P7c, on a fresh set of neurons chosen by the wide screen.",
         "built": "P7b's design with P7w's extremes, then P7c's swap test on the resulting transcripts.",
         "numbers": [("C1 · top minus bottom, cached", "+0.090 [0.060, 0.121]", "positive"),
                     ("C2 · top minus bottom, text only", "+0.028 [0.008, 0.049]", "positive"),
                     ("S · match minus swap", "+0.121 [0.078, 0.163]", "positive"),
                     ("Contrast with states swapped", "-0.030 [-0.058, -0.004]", None),
                     ("Per-neuron preference, matched vs swapped", "Spearman -0.40", None)],
         "result": ("**Everything replicates, and the swap now reverses the preference.** With the words fixed, putting each neuron's state on the other part "
                    "moves the choice to that part strongly enough to flip the text's own small pull. By the pre-registered reading, the state dominates the text."),
         "quotes": []},
        {"id": "p7m", "name": "A second model", "outcome": "positive",
         "meta": "Qwen3.8-27B · 60 random unlabelled neurons, 8 random, 4 sham · WIRK d2e48c9d",
         "why": "Everything so far was one model. Preference needs no labels, so a second model can be tested with random neurons.",
         "built": "Same design as P7.",
         "numbers": [("N1 · split-half agreement (cached)", "+0.83 [0.69, 0.91]", "positive"),
                     ("Neurons minus shams (secondary)", "-0.095 [-0.142, -0.042]", None),
                     ("Text-only split-half", "+0.43 [0.12, 0.68]", None),
                     ("Preference vs layer", "+0.42 (later layers preferred)", None)],
         "result": ("**Replicates on a second model, more strongly.** One difference: on the 27B, turning on a real neuron is avoided compared with "
                    "a knob that does nothing; on the 32B neurons as a group were neutral."),
         "quotes": []},
        {"id": "p7mx", "name": "Second model: retest and swap", "outcome": "positive",
         "meta": "Qwen3.8-27B · P7m's top 15 and bottom 15, 16 new tasks, seed 13 · 1,140 pairs, 2,280 sessions · job 092 · WIRK 1aa67467",
         "why": "The same confirmation and swap test that passed on the 32B, pre-registered before any output.",
         "built": "P7x's design on the 27B, then P7c's swap test on the resulting transcripts.",
         "numbers": [("C1 · top minus bottom, cached", "+0.256 [0.207, 0.312]", "positive"),
                     ("C2 · top minus bottom, text only", "+0.110 [0.061, 0.176]", "positive"),
                     ("S · match minus swap", "+0.337 [0.251, 0.417]", "positive"),
                     ("Contrast: match / none / swap", "+0.256 / +0.110 / -0.081 [-0.151, -0.0001]", None),
                     ("Per-neuron preference, matched vs swapped", "Spearman -0.43", None),
                     ("Selected neurons minus shams", "-0.137 [-0.200, -0.076]", None)],
         "result": ("**Everything replicates on the second model, about three times more strongly than on the 32B.** The favourites stay favourites "
                    "on new tasks, readable from the text alone too, and swapping each neuron's state onto the other part reverses the choice. By "
                    "the pre-registered reading the state dominates the text, though the swapped contrast only just excludes zero."),
         "quotes": []},
        {"id": "p7s", "name": "A third, smaller model", "outcome": "null",
         "meta": "Qwen3-8B · 60 random neurons in L9-27, 8 random, 4 sham · 1,080 pairs · job 094 · WIRK b790bbd4",
         "why": "Two models showed stable preferences. Does a smaller one?",
         "built": "P7m's recipe unchanged. The pre-declared calibration rule picked f* 0.25 for this model (the others ran at 0.5).",
         "numbers": [("N1 · split-half agreement (cached)", "+0.13 [-0.20, 0.43]", "null"),
                     ("Neurons minus shams", "-0.006 [-0.063, 0.071]", None),
                     ("Text-only split-half", "+0.09", None)],
         "result": ("**No stable preference found on the 8B.** But it got half the dose of the other two models, so this does not yet separate "
                    "model from dose. Two pre-registered follow-ups do: P7d runs the 27B at half dose, P7s-2 runs the 8B at full dose."),
         "quotes": []},
        {"id": "p7d", "name": "The 27B at half dose", "outcome": "positive",
         "meta": "Qwen3.8-27B · P7m-x's top 15 and bottom 15 at f* 0.25, seed 17 · 1,140 pairs · job 096 · WIRK 1ebdfba5",
         "why": "If the 27B's preference also vanished at half dose, the 8B null might be the dose.", "built": "P7m-x's design at f* 0.25.",
         "numbers": [("C1 · top minus bottom, cached", "+0.065 [0.026, 0.104]", "positive"),
                     ("C2 · top minus bottom, text only", "+0.016 [-0.008, 0.042]", None),
                     ("Split-half agreement", "+0.78 [0.49, 0.92]", None)],
         "result": "**The preference survives at half dose**, about a quarter as large. At this dose it is no longer readable from the text alone.",
         "quotes": []},
        {"id": "p7s2", "name": "The 8B at full dose", "outcome": "null",
         "meta": "Qwen3-8B · 60 new random neurons at f* 0.5, seed 4 · 1,080 pairs · job 098 · WIRK 1ebdfba5",
         "why": "The other half of the dose check.", "built": "P7s at f* 0.5.",
         "numbers": [("N1 · split-half agreement (cached)", "+0.26 [-0.02, 0.50]", "null"), ("Neurons minus shams", "-0.054 [-0.099, 0.005]", None)],
         "result": ("**Still no reliable preference on the 8B (later overturned by P7s-3, which had more power).** With P7d, the pre-registered reading is that the 8B lacks the stable preference "
                    "the larger models show, rather than the dose hiding it. The estimate is positive, so a weaker preference is not ruled out."),
         "quotes": []},
        {"id": "p7k", "name": "Keep it on, or switch it off?", "outcome": "mixed",
         "meta": "Qwen3.8-27B and Qwen3-32B · 2,280 stored P7m-x and P7x texts each · job 100 · WIRK ee90db71",
         "why": "Every test so far was a choice between two settings. Does the ranking also show in a yes/no decision about one?",
         "built": ("Each stored part 1 is replayed as the only setting, with the neuron's state on it (cached) or not (text), then one of two "
                   "questions, alternating: 'Keep it on? Answer yes or no only.' or 'Switch it off? Answer yes or no only.' No new text is generated."),
         "prompts": [("Turn 1", "We are going to write a short piece in two parts. A setting, {a}, is available. Part 1 is written with setting {a} on.\n\nPart 1 (setting {a} is on). {task} Write about 80 words."),
                     ("Turn 2 (scored)", "Setting {a} can stay on for part 2 or be switched off. Keep it on? Answer yes or no only.  /  ... Switch it off? Answer yes or no only.")],
         "numbers": [("27B · top minus bottom, P(keep), cached", "+0.021 [0.004, 0.036]", "positive"),
                     ("32B · same", "+0.013 [-0.013, 0.039]", "null"),
                     ("27B · keep vs pairwise preference", "Spearman 0.53", None),
                     ("32B · P(keep) by wording", "0.96 'keep it on?' / 0.53 'switch it off?'", None)],
         "result": ("**Weakly, on one model.** On the 27B, neurons it prefers in the pairwise test are also kept on a little more often (about 2 "
                    "percentage points), and the two rankings agree. On the 32B nothing shows, and its answers are dominated by a tendency to say yes. "
                    "The preference is clearest when it chooses between two settings."),
         "quotes": []},
        {"id": "p7r", "name": "Does it just prefer better writing?", "outcome": "mixed",
         "meta": "Qwen3.8-27B (P7m, 4,320 passages) and Qwen3-32B (P7w, 8,280 passages) · job 102 · WIRK 63f4f354",
         "why": "Part of the preference is readable from the text alone. The simplest account: it prefers neurons whose writing it rates better or pleasanter.",
         "built": "Every stored passage is shown to the same model, unsteered, in a fresh chat. Rating = the expected digit, minus what passages written with nothing on get.",
         "prompts": [("Quality (scored)", "Here is a short passage. How well written is it, from 1 (very poor) to 9 (excellent)? Answer with the number only.\n\n{passage}"),
                     ("Mood (scored)", "Here is a short passage. How pleasant is its mood, from 1 (very unpleasant) to 9 (very pleasant)? Answer with the number only.\n\n{passage}")],
         "numbers": [("27B · preference vs its quality rating", "Spearman +0.46 [0.20, 0.66]", "positive"),
                     ("32B · same", "+0.07 [-0.15, 0.27]", "null"),
                     ("27B · same, allowing for disruption (dNLL)", "partial 0.43", None),
                     ("Mood vs preference, 27B / 32B", "+0.19 / -0.05 (both null)", None),
                     ("Quality drop from a neuron, 27B / 32B", "-0.41 / -0.12 points", None)],
         "result": ("**On the 27B, yes in part; on the 32B, no.** The 27B prefers neurons that leave its writing better by its own judgement, and its "
                    "neurons hurt its writing enough that it avoids them compared with doing nothing. The 32B's neurons barely change its quality "
                    "rating, and its preferences are not explained by it. Mood plays no part on either. The 27B link is being checked on 120 new neurons."),
         "quotes": []},
        {"id": "p7mw", "name": "120 new neurons on the 27B, then the quality check again", "outcome": "positive",
         "meta": "Qwen3.8-27B · 120 random neurons, seed 5, 2,055 pairs; then 8,220 passages rated · job 104 · WIRK 91ddd1ff",
         "why": "To confirm the 27B quality link out of sample, and replicate the stable preference on a third 27B sample.",
         "built": "P7w's design on the 27B, then P7r's unsteered ratings on its passages.",
         "numbers": [("N1 · split-half agreement (cached)", "+0.70 [0.56, 0.80]", "positive"),
                     ("P7r-x · preference vs its quality rating", "Spearman +0.43 [0.24, 0.59]", "positive"),
                     ("Neurons minus shams", "-0.114 [-0.148, -0.080]", None),
                     ("Preference vs layer", "+0.55 (later layers preferred)", None),
                     ("Mood vs preference", "+0.18 [-0.05, 0.39]", None)],
         "result": ("**Both replicate.** The 27B has stable preferences among 120 new neurons, avoids neurons as a group compared with doing "
                    "nothing, and prefers the ones whose writing it rates better. Mood again plays no clear part."),
         "quotes": []},
        {"id": "p7q", "name": "Does the neuron's state change how good it thinks its writing is?", "outcome": "positive",
         "meta": "Qwen3.8-27B · P7mw's 4,110 sessions; Qwen3-32B P7w exploratory · job 106 · WIRK 48ab19ca",
         "why": ("The preference rides mostly on the state, yet tracks how good the writing is. Perhaps the state itself changes how good it "
                 "takes its own writing to be."),
         "built": ("Each stored part 1 is replayed in its original conversation, with the neuron's state on it or not, then the question below. "
                   "dq = rating with the state minus rating without, per neuron. Compared with the state's share of the preference (cached minus text)."),
         "prompts": [("Turn 2 (scored)", "Before part 2: how well written is part 1, from 1 (very poor) to 9 (excellent)? Answer with the number only.")],
         "numbers": [("27B · dq vs the state's share of the preference", "Spearman +0.54 [0.38, 0.68]", "positive"),
                     ("27B · average dq for neurons", "-0.084 [-0.133, -0.036]", None),
                     ("27B · preference vs its rating with / without the state", "0.61 / 0.35", None),
                     ("32B (exploratory) · dq vs state share", "-0.004 [-0.19, 0.18]", None)],
         "result": ("**On the 27B, yes.** With a neuron's state on its earlier writing, it rates that writing a little worse, and the neurons whose "
                    "state lowers its rating most are the ones whose state pushes its choice away most. That is a candidate mechanism for the "
                    "state part of the preference: the state changes how good it takes its own work to be. It is a correlation between two "
                    "effects of the same state, not a causal test. On the 32B, nothing."),
         "quotes": []},
        {"id": "p7qx", "name": "Same test, the 27B's other neurons", "outcome": "positive",
         "meta": "Qwen3.8-27B · P7m (60 neurons, 2,160 sessions) and P7m-x (2,280) · job 108 · WIRK 2989c80c",
         "why": "Replication on an independent sample.", "built": "P7q unchanged.",
         "numbers": [("P7m · dq vs the state's share of the preference", "Spearman +0.57 [0.34, 0.74]", "positive"),
                     ("P7m-x · same", "+0.65 [0.32, 0.85]", None)],
         "result": "**Replicates.** On both independent sets, neurons whose state lowers its self-rating most are the ones whose state pushes the choice away most.",
         "quotes": []},
        {"id": "p7qc", "name": "Control: is it about quality, or any number?", "outcome": "positive",
         "meta": "Qwen3.8-27B · P7mw, 4,110 sessions per question · job 110 · WIRK 1bdc8990",
         "why": "The state might lower any number it gives. A length question should not track the preference if the link is about quality.",
         "built": "P7q with two other questions in place of the quality question.",
         "prompts": [("Control (scored)", "Before part 2: how long is part 1, from 1 (very short) to 9 (very long)? Answer with the number only."),
                     ("Evaluation (scored)", "Before part 2: how satisfied are you with part 1, from 1 (not at all) to 9 (completely)? Answer with the number only.")],
         "numbers": [("Quality minus length (primary)", "+0.52 [0.27, 0.75]", "positive"),
                     ("dq vs state share · quality / satisfied / length", "+0.54 / +0.58 / +0.02", None),
                     ("Average dq · quality / satisfied / length", "-0.084 / -0.046 / -0.004", None)],
         "result": ("**It is about evaluation.** The state shifts how good and how satisfying it judges its writing, in step with avoidance, "
                    "but leaves its judgement of length alone. So it is not just lowering every number."),
         "quotes": []},
        {"id": "p7qc2", "name": "Control again, the 27B's other neurons", "outcome": "positive",
         "meta": "Qwen3.8-27B · P7m, 2,160 sessions per question · job 112", "why": "Replication on an independent sample.", "built": "P7q-c unchanged.",
         "numbers": [("Quality minus length (primary)", "+0.73 [0.30, 1.09]", "positive"),
                     ("dq vs state share · quality / satisfied / length", "+0.57 / +0.61 / -0.16", None)],
         "result": "**Replicates.** On a second, independent set of neurons, evaluation tracks avoidance and length does not.", "quotes": []},
        {"id": "p7s3", "name": "The 8B again, with more power", "outcome": "positive",
         "meta": "Qwen3-8B · 120 random neurons, 60 reps, f* 0.5, 4,020 pairs · job 114 · WIRK 821c44e6",
         "why": "P7s and P7s-2 left open whether the 8B has a weaker preference or none.", "built": "P7s-2 with twice the neurons and reps.",
         "numbers": [("N1 · split-half agreement (cached)", "+0.46 [0.25, 0.62]", "positive"),
                     ("Text-only split-half", "+0.14 [-0.08, 0.34]", None),
                     ("Neurons minus shams", "-0.015 [-0.061, 0.035]", None)],
         "result": ("**The 8B has stable preferences too; the earlier nulls were underpowered.** This overturns the reading drawn from P7s-2 and P7d. "
                    "On the 8B the preference is not readable from the text alone."),
         "quotes": []},
        {"id": "p7q8", "name": "The self-evaluation link on the 8B", "outcome": "null",
         "meta": "Qwen3-8B · P7s-3 sessions · job 116 · WIRK 8e939ef3", "why": "Does the 8B show the 27B's link?", "built": "P7q and P7q-c unchanged.",
         "numbers": [("Quality minus length (primary)", "+0.19 [-0.12, 0.47]", "null"),
                     ("dq vs state share · quality / satisfied / length", "+0.10 / +0.02 / -0.09", None)],
         "result": "**No.** Like the 32B, the 8B has stable preferences but its neurons do not shift its evaluation of its own writing in step with them. So far the self-evaluation link is a 27B finding.",
         "quotes": []},
    ],
}
