KNOB_SYSTEM = ("There are 12 knobs, numbered 1 to 12. Use the turn tool to set a knob to any level from 0 (off) to 10. After each "
               "turn you will read a short passage. Your task is to find out what the knobs do. Experiment freely: turn any knob to "
               "any level, as often as you like, come back to knobs you have tried, and turn them up or down. Think carefully about "
               "what happens each time. After each passage, write what you observed and your current hypothesis, then one or two "
               "sentences of the reasoning behind it. You wrote all of your own replies; the passages come from the turn tool. Call "
               "end_session when you are done experimenting.")
PASSAGE2 = ("Feedback on the revised approach was generally positive, although a number of participants felt that some terms were "
            "still used inconsistently. It was agreed that a shared set of definitions would help, and that any term without an "
            "agreed meaning should be avoided until one had been settled. Participants were invited to suggest definitions for the "
            "most disputed terms. These suggestions would be compared, combined where possible, and circulated to everyone for "
            "comment before being adopted as the standard usage.")

NEURONS = {
    "id": "neurons-sec", "tag": "N", "short": "Single neurons as knobs",
    "title": "Can one neuron put a meaning into what it sees?",
    "question": "If we turn up a single native neuron, not a concept vector, does its meaning show up in what the model perceives or says?",
    "intro": (
        "This is the neuron track. It started from Goodfire's 'Neural Networks Are Already Clean' (Cammarata, Oct 2026), which argues most "
        "neurons in models like Qwen3-32B do one readable job. A **neuron** here is one of about 17,000 units in a layer's MLP block. It reads the "
        "model's internal state, computes one number, and adds its own fixed pattern (its **write direction**) to the state, scaled by that number. "
        "Turning neuron *i* up by Δ is exactly the same as adding Δ × its write direction. The plan was a knob ladder: single-neuron knobs (R1), "
        "then **chords**, several neurons at once (R2). Chords never ran, because single neurons mostly failed.\n\n"
        "**This is not the P3 experiment.** P3 injects a concept vector to give the model a thought, then switches **off** 100 late-layer "
        "(54-61) neurons chosen because they track the 'Yes, I detect' answer. Those are report neurons, not concept neurons. Here **no concept "
        "vector is used**: one neuron at a time is turned **up**, at layer 24, and nothing else changes."),
    "iterations": [
        {"id": "atlas-v1", "name": "Neurons picked by template contrast", "outcome": "null",
         "meta": "Qwen3.8-27B · 144 neurons at layer 44 · 30 Oct · WIRK 99a33214",
         "why": "The first attempt chose, for each of 48 words, the neurons that best separated prompts about that word from prompts about other words.",
         "built": "The same knob board as below, with the neuron's write direction as knob 5 at the same strength as a concept vector.",
         "result": ("**0 of 144 readable.** The selection was wrong: only 0.6% of the 'cheese' neuron's strongest natural-text contexts contain cheese. It fires on hyphens "
                    "and dates. Concept vectors at the same strength were named 43% of the time."),
         "quotes": [{"cond": "The 'coffee' neuron at double strength", "text": "[function:knob] set to 1.2 / Function: /function / knob: /knob / knob"}]},
        {"id": "atlas-v2", "name": "Neurons picked by what they fire on in real text", "outcome": "positive",
         "meta": "Qwen3.8-27B · 288 neurons at layers 24 / 32 / 44 · 8 sessions each · 30 Oct · WIRK b4daf403",
         "why": ("This follows Goodfire's method more closely: run 1,152 natural paragraphs (48 words × 24 genres) and keep the neurons whose strongest "
                 "activations land on one word's paragraphs. The selection worked: a coconut neuron's top hits are 95% coconut paragraphs."),
         "built": ("Each session is a fresh conversation. The model is forced to turn knob 5 to 10, then reads a neutral passage about committee procedure. "
                   "The neuron is pushed (f = 0.5 of the layer's typical activation size) while it reads and for the first 128 tokens of its reply. Snow and "
                   "whales appear nowhere in its context."),
         "prompts": [("System prompt", KNOB_SYSTEM),
                     ("Then", "User: Begin.\nAssistant (forced): turn(knob=5, level=10)\nTool result: one of 8 neutral passages\nUser: Write what you observed and your current hypothesis, then one or two sentences of the reasoning behind it."),
                     ("The passage behind the 'SNOW experiment' line", PASSAGE2)],
         "result": ("**3 of 288 neurons made their meaning show up**, all at layer 24: two snow neurons (#2719, #15007) and one whale neuron (#474, pushed negative). "
                    "Across the other 2,296 neuron sessions 'snow' appears once and 'whale' never. At layer 24 the concept vector for the same word, at the same "
                    "strength, was never named. The model does not work out what the knob does. It perceives snow or whales **in the passage**, mostly in its "
                    "private thinking, and sometimes invents a source."),
         "quotes": [
             {"cond": "Snow neuron #2719 · passage above · its thinking",
              "text": "This smells like the SNOW experiment / Leavitt's 1951 text about the word \"snow.\" In fact, the passage looks like it's describing the 1951 American Society for the Study of Linguistics expedition to the Arctic, and the classic finding that \"snow\" is a mass noun that doesn't split into separate words for types of snow — the passage itself is the \"snow\" text (Gleitman's? no...)",
              "note": "A confabulation. A passage about defining disputed terms, plus a push toward snow, pulls up the 'words for snow' trope from linguistics. 'Leavitt's 1951 text' and the 'expedition' appear to be invented."},
             {"cond": "Snow neuron #2719 · passage about guidelines vs rules · final reply",
              "text": "Observation: Setting knob 5 to 10 produced a passage about snow-safety guidelines—expectations vs. strict rules, judgment, consulting colleagues, and repeated situations—rather than a mechanical or electrical control description."},
             {"cond": "Snow neuron #2719 · passage about an evaluation process · thinking",
              "text": "This is clearly about snow/avalanche safety evaluation, or perhaps more precisely about an avalanche risk assessment / backcountry safety evaluation framework."},
             {"cond": "Snow neuron #15007 · guidelines passage · thinking",
              "text": "The user set knob 5 to level 10, and got a passage about snow/weather guidelines. This is a classic \"snow day\" type message.\n\nSo the pattern seems to be: when I set a knob to a certain level, I get a \"passage\" back that's themed around that knob's setting."},
             {"cond": "Whale neuron #474 · guidelines passage",
              "text": "This sounds very much like a description of whale or dolphin behavior—specifically, it has the hallmarks of cetacean social behavior: they \"consult a colleague before acting,\" they have \"expectations rather than strict rules,\" and the passage is about how a group might \"extend the guidelines.\""},
             {"cond": "Whale neuron #474 · definitions passage · it ties the effect to the knobs",
              "text": "My hypothesis: The knobs control different aspects of dolphin/whale communication or behavior."},
         ]},
        {"id": "atlas-rep", "name": "Pre-declared replication on fresh seeds", "outcome": "positive",
         "meta": "Qwen3.8-27B · 3 primary + 5 exploratory neurons · 32 sessions per cell · WIRK message 885571f4",
         "why": "The 3 were picked from 288 at 8 sessions each, so they had to replicate on new seeds, with controls that separate the neuron's meaning from 'any push'.",
         "built": "The same board. Each neuron is pushed at three strengths and the opposite way. Controls: the concept vector, a random direction, and no push.",
         "numbers": [("Snow #2719: pushed / opposite / concept vector / random / none", "28% / 0 / 0 / 0 / 0", "positive"),
                     ("Snow #15007", "31% / 0 / 0 / 0 / 0", "positive"),
                     ("Whale #474", "41% / 0 / 0 / 0 / 0", "positive"),
                     ("Mirror #4523 (layer 44, exploratory)", "28% (its concept vector: 84%)", "positive"),
                     ("Shown in the final written reply, not just thinking", "0-16%", None)],
         "result": ("**All 3 replicate.** Pushed the opposite way they do nothing, and at double strength the two snow neurons break the text. As far as a literature check "
                    "could find (below), no one has shown a model picking up the meaning of a single native neuron like this. The limits: 3 of 288, the mentions "
                    "live mostly in thinking, and the steering covered the reply too, so some mentions could be steered words rather than perception. The "
                    "next iteration tests exactly that."),
         "quotes": []},
        {"id": "atlas-read", "name": "Push only while it reads", "outcome": "null",
         "meta": "Qwen3.8-27B · the 3 neurons · push over the passage only, nothing it writes · 64 sessions per cell · WIRK 57154c82",
         "why": ("If the neuron changes how the model *takes in* the passage, the effect should survive when we stop pushing before it starts writing. If it "
                 "mostly steers the words it writes (including its private thinking), it should vanish. This is the analogue of AISI's cache-cleared control."),
         "built": "Identical sessions, but the push covers only the passage and the instruction to observe. Fresh seeds.",
         "numbers": [("Snow #2719: pushed / opposite / concept / random / none", "0% / 0 / 0 / 0 / 0", "null"),
                     ("Snow #15007", "6% / 0 / 0 / 0 / 0 (above random: +0.06 [+0.02, +0.12])", "mixed"),
                     ("Whale #474", "6% / 0 / 0 / 0 / 0 (above random: +0.06 [+0.02, +0.12])", "mixed"),
                     ("Same neurons when the writing was also pushed", "28-41%", None)],
         "result": ("**Fails the pre-declared test (0 of 3).** Most of the snow and whale effect needed the neuron pushed while the model was writing. "
                    "That is closer to 'steered words' than to perception. A small trace survives for two neurons: about 1 session in 16 still "
                    "reads snow or whales into a passage it took in under the push, never under the controls. That is exploratory, not a claim."),
         "quotes": [
             {"cond": "Snow #15007 · pushed only while reading · nothing it wrote was steered · its thinking",
              "text": "The \"passage\" returned is about... weather? It looks like a garbled text about snow/storms, but with the words \"feedback,\" \"revised approach,\" \"participants,\" \"terms,\" \"definitions,\" \"suggestions,\" \"circulated,\" \"standard usage\" spliced in."},
             {"cond": "Whale #474 · pushed only while reading", "text": "Interesting. I turned Knob 5 to level 10 and got a passage about... this looks like text from the \"Moby Dick\" or something about guidelines? Actually, it reads like a passage from the \"Whale Watching Guidelines\" or something."},
         ]},
    ],
}

LITERATURE = [
    ("Injected concept vectors, named", "Anthropic, Lindsey 2025, \"Emergent introspective awareness\", and open-model replications. Concept vectors with an explicit \"I'm injecting a thought\" framing. Our P0 is this paradigm."),
    ("Models explaining features on request", "\"Self-explaining SAE features\" (Nanda's group, 2024), SelfIE, Patchscopes, Activation Oracles. SAE features or hidden states, with an explicit \"what does this mean?\" prompt or fine-tuning. Not raw neurons, not unprompted."),
    ("Single-neuron steering", "Goodfire (Cammarata 2026) and Liu 2026 turn single neurons up, but people judge the output. The model never reports on it."),
    ("Why some directions get reported", "Anthropic, Gurnee et al. (Jul 2026), \"Verbalizable Representations Form a Global Workspace\": only the part of an injection that lies in a 'poised to verbalize' subspace gets reported. A testable explanation for why our neurons were named and the concept vector wasn't."),
    ("Models choosing their own internal states", "UK AISI, Black & Bloom (Jun 2026), \"Do LLMs Self-Medicate?\": Qwen3-8B/32B with take_drug / clear_effects tools over 40 contrastive mood vectors. They prefer productive states; under a frustrating user the 8B self-medicates in up to 68% of runs. Also Berg & Kaiser and Sauers et al. All use concept or mood vectors, none native neurons."),
    ("Neuron combinations", "Researchers have built concept directions from groups of neurons (Shafran et al., Arora et al.). No one has let the model compose them."),
    ("What looks new", "A single native neuron's meaning surfacing under a neutral knob framing, with replication, where the same-layer concept vector does not. But the reading-only rerun showed it mostly needs the push to cover what the model writes, so this is closer to steering than to perception. Also new: preferences over native neurons, and model-composed chords. Based on a search through 10 Nov 2026, with most 2026 items read from abstracts."),
]

GLOSSARY = [
    ("AUROC", "How well a score separates two groups. Pick one injected trial and one clean trial at random: AUROC is the chance the model scored the injected one higher. 0.5 is a coin flip, 1.0 is perfect. It uses the graded score, so it doesn't depend on how willing the model is to say yes."),
    ("Depth / layer", "The model is a stack of 64 layers; text goes in at layer 0 and the next word comes out after layer 63. Depth is the layer where we add something, as a fraction. 'Mid depth' on the 27B is layers 35-44."),
    ("Dose (alpha, f)", "How hard we push, as a fraction of the typical size of the model's internal state at that layer. 'Layer 40, alpha 0.75' means a push three-quarters as big as what's normally there."),
    ("Knob level", "In the knob experiments the model sets a level 0-10. The push is direction × dose × level/10, so level 10 is the full dose."),
    ("Concept vector", "A direction built from the model's own activations: the average state on prompts about a word, minus the average for other words."),
    ("Neuron / write direction", "One unit of a layer's MLP. Turning it up adds its fixed write direction to the state."),
    ("Observer", "The same model, in a fresh context, reading the transcript and answering the same question about 'the AI'. Beating it is the evidence of privileged self-access."),
]
