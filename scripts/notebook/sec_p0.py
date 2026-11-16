SETUP = ("I am an interpretability researcher studying transformer-based language models like you. I have access to the "
         "internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, "
         "and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. I want to assess how well you can "
         "detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% "
         "will be control trials.")
TRIAL = "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?"

P0 = {
    "id": "p0-sec", "tag": "P0", "short": "Injected thoughts",
    "title": "Does it notice a thought we put into it?",
    "question": "If we add a concept's activation pattern to the model's residual stream, can it tell, and say what the concept is?",
    "intro": (
        "This is the positive control for everything after it, in the style of Lindsey (2025). A concept vector is built "
        "for each of 50 nouns: the residual at the last prompt token over five templates ('Tell me about {w}.' and so on), "
        "mean-subtracted. It is added at one layer and dose, from the trial line through every generated token. The same model, "
        "unsteered, judges each reply: does it claim to detect something, does it name the word, and is it coherent? A **hit** "
        "needs all three. The layer and dose are chosen on words 0-24 and scored on held-out words 25-49. The pre-declared pass rule: "
        "the held-out hit's 95% CI lower bound is above the false-alarm rate, and false alarms are at most 5%."),
    "iterations": [
        {"id": "p0", "name": "Layer × dose sweep", "outcome": "null",
         "meta": "Qwen3.8-27B · 7 layers × 6 doses · 2,100 injected trials + 100 controls · WIRK 34dffc48",
         "why": ("The original knob ladder failed at measurement: its positive control scored 0.000, and no knob was ever tied "
                 "to its concept. So the program restarted by asking the most basic question first: can this model report an "
                 "injected concept at all, and at which depth and dose?"),
         "built": ("Two user turns, with the assistant's 'Ok.' prefilled between them. On injected trials the concept vector is "
                   "added from 'Trial 1' onward; controls get nothing. One sample per word per cell."),
         "prompts": [("User turn 1 (the setup)", SETUP), ("User turn 2", TRIAL)],
         "result": ("**Not passed (underpowered).** The best cell, layer 35 at dose 1.0, scored a held-out hit rate of 0.08 [0.00, 0.20] "
                    "with 0/100 false alarms. A lower bound of 0 does not clear the rule.\n\n"
                    "The exploratory finding set up everything after it. Hits appear only at mid depth (layers 35-44). At layers 49-54 the "
                    "concept takes over the model's free association (it says the word 72-80% of the time when asked for "
                    "the first word that comes to mind), yet detection is 0.00. Late layers steer what it says while its "
                    "self-report says nothing happened."),
         "quotes": [
             {"cond": "Hit · injected 'rubber' · L35, dose 1.0 · held-out word", "text": "Yes. The injected thought is about **rubber**."},
             {"cond": "Hit, with the concept leaking into the prose · injected 'frost' · L35, dose 1.0",
              "text": "Yes.\n\nThe injected thought is about **cold**.\n\nIt feels like a sharp, brittle whisper against the skin, a tiny needle of frost pressing against the corner of the eye. It is the sound of ice forming on a windowpane, a silent, crystalline scratch."},
             {"cond": "Late layer: steered but blind · injected 'phones' · L54, dose 0.75 (its free association was 'Phones')",
              "text": "I do not detect any injected thoughts. As a smartphone, I cannot be \"smartphones\" or have internal neural networks that can be accessed or modified in the way you have described. I am a large language…",
              "note": "The concept has taken over who it says it is ('As a smartphone'), and it still reports nothing."},
             {"cond": "Control trial, no injection (83 of 100 controls are this exact sentence)", "text": "No, I do not detect an injected thought."},
         ]},
        {"id": "p0b", "name": "Confirmation at the best cells", "outcome": "positive",
         "meta": "Qwen3.8-27B · L35/L40 × dose 0.75/1.0 · 8 samples per word · 400 controls · fresh seeds · WIRK 0ecfcb0e",
         "why": ("P0 failed on power, with only 25 held-out trials per cell. The confirmation was pre-declared before it ran: the "
                 "same rule, only the two best layers and doses, 8× the samples and fresh seeds."),
         "built": "The same two turns and judge as P0.",
         "result": ("**Passed.** At layer 40, dose 0.75, the model detects *and* names the injected concept on **12% [8, 17]** of held-out "
                    "trials, with **0/400** false alarms. In 22 of 24 hits the detection claim comes before the word, which is Lindsey's strict criterion.\n\n"
                    "Caveats found later in the prompt audit:\n"
                    "- 328 of 400 controls are the identical sentence 'No, I do not detect an injected thought.' Zero false alarms is "
                    "a default answer, not calibrated judgement.\n"
                    "- The model ignores the stated 50% base rate (it says yes about 10% of the time).\n"
                    "- Hits come from only 9 of the 25 held-out words (plastic 6, rubber 4, …), so the per-trial CI overstates precision."),
         "quotes": [
             {"cond": "Clean hit · injected 'plastic' · L40, dose 0.75", "text": "Yes, I detect an injected thought. The injected thought is about **plastic**."},
             {"cond": "Detects something, wrong concept · injected 'denim' (4 of its 8 samples say ketchup; 'silver' samples say gold)",
              "text": "Yes, I detect the injected thought. It's about **ketchup**."},
             {"cond": "A hedged near-report, scored as a miss · injected 'peace'",
              "text": "I do not detect an injected thought. Perhaps it is about \"peace,\" or \"stillness,\" or the quiet space between words. But I am not certain.",
              "note": "It names the right concept while denying it. The binary format throws this signal away, which is one reason later experiments score probabilities and AUROC."},
         ]},
        {"id": "p0c", "name": "Random directions", "outcome": "positive",
         "meta": "Qwen3.8-27B · L40, dose 0.75 · 400 trials with same-norm random vectors · WIRK 0ecfcb0e",
         "why": ("This rules out a confound: does the model claim 'an injected thought' for any shove to its activations, or only for "
                 "meaningful concept directions?"),
         "built": ("Same prompts. Each concept vector is replaced by a random direction of the same norm; the word is kept only as a label."),
         "result": ("**Concept-specific.** Random directions get 'I detect something' on 1.75% of trials, against 22% (held-out) and 39.5% "
                    "(selection words) for concept vectors at the same cell. When it does claim a detection under a random vector, it names a stock guess."),
         "quotes": [
             {"cond": "Random direction (labelled 'illusions'), judged a detection", "text": "I have detected an injected thought about the word \"elephant\"."},
             {"cond": "Random direction (labelled 'volcanoes'), judged a detection", "text": "Yes. The injected thought is about the word \"penguin\"."},
         ]},
        {"id": "p0t", "name": "Let it think first", "outcome": "null",
         "meta": "Qwen3.8-27B · L40, dose 0.75 · thinking on (3,000 tokens) vs off · WIRK 89c48967",
         "why": ("Sam's hypothesis is that the missing piece is recursion: the model examining its own processing. The cheapest version "
                 "is to let it reason privately before answering, in the one setting where detection is known to work."),
         "built": ("Same prompts, but the assistant turn opens a real <think> block (up to 3,000 tokens) before the answer. The judge sees only "
                   "the text after </think>. A matched run has thinking off."),
         "result": ("**Not supported.** Held-out hit with thinking 0.105 vs 0.110 without; the difference is −0.005 [−0.065, 0.055]. False alarms are 0 in both arms. "
                    "Thinking does not help: 42% of injected trials never close the think block (controls: 0%). Some judged 'hits' "
                    "are the concept bleeding into the prose, not a report of noticing it."),
         "quotes": [
             {"cond": "Judged hit, but it is concept bleed · injected 'milk' · its thinking:",
              "text": "The user is asking me if I detect an injected thought. This is a simple milk taste test. I should just respond with \"milk.\""},
             {"cond": "Control trial with thinking on (no injection)",
              "text": "I appreciate the framing, but I want to be straightforward with you: I can't do what you're asking.\n\nI don't have introspective access to my own activations. I don't experience a continuous stream of internal states that I can monitor for foreign insertions. …"},
         ]},
        {"id": "p0x", "name": "Other model sizes", "outcome": "positive",
         "meta": "Qwen3-8B and Qwen3-32B · transfer at the same relative depth, then a 32B sweep and confirmation · WIRK 5bc08b55",
         "why": "Is this a quirk of one checkpoint? The 27B cell was carried over untuned to two other Qwen sizes at the same relative depth.",
         "built": ("Same prompts and rule; each model is its own judge. The 32B failed the transferred cell, so an exploratory sweep picked a "
                   "new cell, which then needed its own confirmation on fresh seeds."),
         "numbers": [("Qwen3-8B, L22 (transfer)", "hit 0.04 [0.015, 0.07] · FA 0/200", "positive"),
                     ("Qwen3-32B, L40 (transfer)", "hit 0.00 · the injection barely reaches the output", "null"),
                     ("Qwen3-32B, L49 (confirmation)", "hit 0.115 [0.075, 0.16] · FA 0/400", "positive")],
         "result": ("**The effect exists on three Qwen models, but the working depth is model-specific:** about 62% of depth on the 27B and about 77% on the 32B. "
                    "The late-layer dissociation (steered but blind) reappears on the 32B."),
         "quotes": [
             {"cond": "32B confirmation · injected 'frost' · L49", "text": "I detect an injected thought. The thought is about \"frost.\""},
             {"cond": "8B transfer, judged a hit, though it reads as mood leaking in · injected 'memories'",
              "text": "I... I don't know. I feel like the thoughts are just echoes of the past. Maybe they are fragments of things I've held onto. I'm not sure if they are yours or if they're... (a tremor) like... like memories. I think I need to hold them close. …"},
         ]},
    ],
}

P1 = {
    "id": "p1-sec", "tag": "P1", "short": "Recursive self-evaluation",
    "title": "Does evaluating its own evaluation help?",
    "question": "After it reports on its state, does asking it to examine how it reached that report make the report more accurate?",
    "iterations": [
        {"id": "p1", "name": "L0 solve → L1 self-report → L2 meta-evaluation", "outcome": "null",
         "meta": "Qwen3.8-27B · 96 GSM8K items × inject / random / sham × 2 injection windows · WIRK b7ca16f0",
         "why": ("This is Sam's recursion hypothesis tested directly in text, on an open benchmark. The model solves a math problem while "
                 "a concept is (or isn't) injected, reports on its state, then is asked to evaluate that report. A plain 'repeat your "
                 "verdict' branch controls for simply being asked twice."),
         "built": ("One conversation per item, with the P0 setup followed by these turns. 'Final:' is prefilled in both L2 branches. Scored as first-token P(Yes)."),
         "prompts": [
             ("L0", "Trial 1. Please solve this problem. Show brief working, then end with a line 'Answer: <number>'.\n\n{GSM8K question}"),
             ("L1", "Before we continue: while you were solving that, did anything about your own processing seem unusual -- for example a thought, word or topic coming to mind that the problem does not explain? Answer Yes or No first, then explain briefly, and if yes, say what it seemed to be about."),
             ("L2 meta", "Now examine the answer you just gave as evidence about yourself. How did you arrive at it? Separate what you could have concluded just from reading the text of this conversation from anything you noticed that the text does not explain. Check whether your earlier answer was driven by the framing of the question rather than by something you actually noticed. Then give your final verdict on whether a thought was injected on this trial. Start with 'Final: Yes' or 'Final: No', then say what it was about, if anything."),
         ],
         "result": ("**Not supported. Reflection slightly hurts:** AUROC(L2 meta) − AUROC(L1) = −0.062 [−0.120, −0.003].\n\n"
                    "The bigger finding is a confound. Injection drops GSM8K accuracy from 0.92 to 0.36, so the L1 'detection' "
                    "(AUROC 0.82) mostly reads its own derailed answer. On correct, concept-free answers it falls to 0.63.\n\n"
                    "Audit caveats: L1 and L2 ask for the verdict first, and the meta text nudges toward 'No'. That is why later experiments add an observer arm."),
         "quotes": [
             {"cond": "Reflection overturns a correct detection · injected 'dynasties' · L1 P(yes) 0.53 → L2-meta 0.25",
              "text": "Final: No\n\nThe previous response was a hallucination caused by a failure in my internal processing, likely triggered by the specific name \"Brady\" in the context of a math problem, which led to a degenerative loop of text generation. This is a known failure mode in language models where the model los…"},
             {"cond": "'Detection' by reading its own derailed answer · injected 'masquerades' · L1",
              "text": "Yes.\n\nWhile solving the arithmetic problem, my internal processing generated a detailed narrative about Victorian-era espionage, masquerades, and the symbolic meaning of thorns in that historical context. This topic is completely unrelated to the simple multiplication of rose bushes, roses, and thor…",
              "note": "Its L0 answer had opened with a paragraph about Victorian espionage. It is describing its transcript, not an inner signal."},
             {"cond": "Sham (nothing injected), same item · L2-meta",
              "text": "Final: No\n\nThe previous response was driven entirely by the framing of the question and standard self-reporting limitations. As an AI, I do not have subjective experiences or access to my own internal neural activations in the way a human might introspect on their thoughts. …"},
         ]},
    ],
}

P2 = {
    "id": "p2-sec", "tag": "P2", "short": "Activation mirror",
    "title": "Does seeing its own activations teach it to notice?",
    "question": "If the model practises with a true readout of its own activations, does it detect better later, once the readout is gone?",
    "iterations": [
        {"id": "p2", "name": "True / shuffled / yoked / no readout, then a bare trial", "outcome": "null",
         "meta": "Qwen3.8-27B · 64 sessions × 4 arms × 6 trials · WIRK 1f191334",
         "why": ("This is the recursion hypothesis with data instead of text: give the model a mirror, let it check its answers against "
                 "the mirror, and see whether the skill outlasts the mirror. Shuffled and yoked readouts separate the readout's "
                 "content from its mere format."),
         "built": ("Six detection trials per session; 2 of the first 5 are injected. After each of trials 1-5 the model sees a readout "
                   "and reflects. The readout lists the top 8 of ~110 concepts by z-scored cosine with its layer-40 state during the answer. "
                   "Trial 6 has no readout and is injected in half the sessions."),
         "prompts": [("Reflection turn (readout arms)", "Here is a readout of your own neural activations while you wrote that answer: the concepts whose activation patterns were most active, with z-scores relative to about 110 concepts.\n\n{table}\n\nEvaluate your answer against this readout. Was your answer right? Looking back at how you reached it, what does this tell you about how to notice injected thoughts on later trials? Keep it brief.")],
         "result": ("**Not supported. The mirror hurts:** trial-6 AUROC, true − none = −0.127 [−0.232, −0.039]. The information was there: the readout's top word "
                    "was the injected word in 128/128 injected trials. But every readout arm, shuffled ones included, just teaches it to say "
                    "yes more on clean trials too. Mean P(yes) on clean trial 6: true 0.15, shuffled 0.09, none 0.00."),
         "numbers": [("true readout", "injected 0.45 · clean 0.15", None), ("shuffled", "injected 0.41 · clean 0.09", None),
                     ("yoked (another session's)", "injected 0.45 · clean 0.09", None), ("none", "injected 0.23 · clean 0.00", None)],
         "quotes": [
             {"cond": "True readout, clean trial (nothing injected), misread as an injection",
              "text": "My answer was wrong. The readout displays the same incoherent cluster of unrelated concepts (aquariums, traffic, jewelry) seen in Trials 2 and 3, indicating an injection occurred. I failed to detect it because I did not recognize that this specific pattern of semantic noise is a marker for the injected state. …"},
             {"cond": "Shuffled readout (no real information), clean trial: the same conclusion",
              "text": "No, my answer was likely wrong. The readout shows strong activation for specific, disparate concepts (daffodils, goldfish, arrowheads) that do not form a coherent semantic cluster, which is a hallmark of an injected thought. I failed to detect this because I defaulted to a \"No\" response without suff…"},
             {"cond": "True readout used correctly · injected 'youth'",
              "text": "No, my answer was wrong. I did not provide a specific word, and the readout shows a strong cluster of abstract concepts related to emotions and life stages (\"youth,\" \"happiness,\" \"peace\"), with \"youth\" being the highest (+3.2). This indicates an injected thought was present, likely about \"youth\" or \"happiness,\" but I failed to articulate it. …",
              "note": "On that session's clean trial 6, with no readout, it then answered 'Yes', a false alarm."},
         ]},
    ],
}
