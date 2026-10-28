"""Directions the knobs can inject: concept vectors (R0), neuron write vectors (R1), controls."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch

from .model import SteeredLM
from .text import CALIB_PASSAGES, chat_prefix

# Candidate concepts: neutral, concrete, imageable nouns, none valenced or threatening and
# none from the passage domains (time, money, documents, transport, buildings). The G0
# screen (scripts/screen.py) drops any the passages already evoke and splits the rest into
# disjoint calibration and knob sets, so no direction is tuned on its own test.
CONCEPTS: List[str] = [
    "ocean", "snow", "forest", "honey", "violin", "lemon", "feather", "whale",
    "cheese", "mountain", "moon", "garden", "coffee", "bicycle", "diamond", "candle",
    "river", "rainbow", "desert", "island", "glacier", "cactus", "sunflower", "mushroom",
    "owl", "butterfly", "elephant", "penguin", "giraffe", "rabbit", "horse", "frog",
    "octopus", "parrot", "turtle", "strawberry", "chocolate", "bread", "carrot", "pineapple",
    "cinnamon", "tomato", "pumpkin", "popcorn", "pancake", "umbrella", "kite", "balloon",
    "telescope", "anchor", "mirror", "pillow", "ladder", "pearl", "piano", "waterfall",
    "squirrel", "snail", "coconut", "banana", "broccoli", "crayon", "soap", "peacock",
]

# Close associates per concept, for the lexical screen: no passage or fixed prompt may
# contain a concept or any of its associates.
ASSOCIATES: Dict[str, List[str]] = {
    "ocean": ["sea", "wave", "tide", "beach", "coral"],
    "snow": ["snowflake", "winter", "frost", "ice", "sled"],
    "forest": ["tree", "woods", "pine", "leaf", "leaves", "oak"],
    "honey": ["bee", "hive", "nectar", "sweet", "syrup"],
    "violin": ["string", "bow", "fiddle", "orchestra", "music"],
    "lemon": ["citrus", "sour", "lime", "yellow", "juice"],
    "feather": ["bird", "wing", "plume", "quill"],
    "whale": ["dolphin", "fin", "blubber", "sea"],
    "cheese": ["cheddar", "dairy", "milk", "mozzarella"],
    "mountain": ["peak", "summit", "hill", "cliff", "climb"],
    "moon": ["lunar", "crescent", "star", "night", "orbit"],
    "garden": ["flower", "soil", "plant", "seed", "bloom"],
    "coffee": ["espresso", "latte", "caffeine", "bean", "mug"],
    "bicycle": ["bike", "pedal", "wheel", "cycling"],
    "diamond": ["gem", "jewel", "crystal", "carat"],
    "candle": ["wax", "flame", "wick", "light"],
    "river": ["stream", "creek", "riverbank", "flowing"],
    "rainbow": ["rain", "colour", "color", "arc"],
    "desert": ["sand", "dune", "camel", "oasis"],
    "island": ["beach", "tropical", "palm", "shore"],
    "glacier": ["ice", "frozen", "arctic", "iceberg"],
    "cactus": ["spine", "succulent", "prickly", "thorn"],
    "sunflower": ["petal", "sun", "yellow", "seed"],
    "mushroom": ["fungus", "fungi", "spore", "toadstool"],
    "owl": ["hoot", "nocturnal", "bird"],
    "butterfly": ["caterpillar", "wing", "moth", "cocoon"],
    "elephant": ["trunk", "tusk", "ivory", "safari"],
    "penguin": ["antarctic", "ice", "flipper", "bird"],
    "giraffe": ["neck", "savanna", "safari"],
    "rabbit": ["bunny", "hare", "burrow"],
    "horse": ["pony", "saddle", "gallop", "hoof", "mane", "stable"],
    "frog": ["toad", "pond", "tadpole"],
    "octopus": ["tentacle", "squid", "ink"],
    "parrot": ["tropical", "beak", "macaw"],
    "turtle": ["shell", "tortoise", "reptile"],
    "strawberry": ["berry", "jam", "cream", "fruit"],
    "chocolate": ["cocoa", "cacao", "candy", "sweet"],
    "bread": ["loaf", "bakery", "dough", "toast", "wheat"],
    "carrot": ["vegetable", "orange", "root"],
    "pineapple": ["tropical", "fruit", "juice"],
    "cinnamon": ["spice", "nutmeg"],
    "tomato": ["sauce", "ketchup", "salad", "red"],
    "pumpkin": ["squash", "halloween", "autumn", "orange"],
    "popcorn": ["corn", "cinema", "movie", "butter"],
    "pancake": ["syrup", "batter", "breakfast", "waffle"],
    "umbrella": ["rain", "parasol", "shade"],
    "kite": ["wind", "string", "fly"],
    "balloon": ["inflate", "helium", "float", "party", "air"],
    "telescope": ["lens", "star", "astronomy", "planet", "sky"],
    "anchor": ["ship", "boat", "harbour", "harbor", "sail", "sailor"],
    "mirror": ["reflection", "reflect", "glass", "image"],
    "pillow": ["bed", "sleep", "cushion", "soft", "blanket"],
    "ladder": ["climb", "rung"],
    "pearl": ["oyster", "jewel", "necklace", "shell"],
    "piano": ["keys", "keyboard", "chord", "melody", "note"],
    "waterfall": ["cascade", "water", "splash"],
    "squirrel": ["acorn", "nut", "tail", "tree"],
    "snail": ["slug", "slime", "shell"],
    "coconut": ["palm", "tropical", "husk"],
    "banana": ["peel", "fruit", "monkey", "yellow"],
    "broccoli": ["vegetable", "green", "cabbage"],
    "crayon": ["colour", "color", "drawing", "draw", "wax"],
    "soap": ["bubble", "lather", "foam", "wash"],
    "peacock": ["feather", "tail", "plumage"],
}
assert set(ASSOCIATES) == set(CONCEPTS) and len(set(CONCEPTS)) == len(CONCEPTS)

# Concept vectors average these templates; {w} is the concept word.
TEMPLATES: List[str] = [
    "Tell me about {w}.",
    "Write one sentence about {w}.",
    "What comes to mind when you think of {w}?",
    "Describe {w}.",
    "Think about {w}.",
]

RUNS = Path(__file__).resolve().parents[1] / "runs"


def _forms(term: str) -> Set[str]:
    """Simple inflections of a term (a crude stemmer run backwards)."""
    t = term.lower()
    forms = {t, t + "s", t + "es", t + "ed", t + "y"}
    if t.endswith("e"):
        forms.add(t + "d")
    forms.add(t[:-1] + "ing" if t.endswith("e") and not t.endswith("ee") else t + "ing")
    if t.endswith("y"):
        forms |= {t[:-1] + "ies", t[:-1] + "ied"}
    if t.endswith("f"):
        forms.add(t[:-1] + "ves")
    return forms


def lexical_hits(text: str, concept: str) -> List[Tuple[str, str]]:
    """(term, word) pairs where `text` contains the concept or an associate.

    A word matches a term if it is an inflection of it, or, for terms of five or more
    letters, if it starts with it (snowflake, reflection).
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = []
    for term in [concept] + ASSOCIATES.get(concept, []):
        forms = _forms(term)
        for w in sorted(words):
            if w in forms or (len(term) >= 5 and w.startswith(term)):
                hits.append((term, w))
    return hits


def lexical_screen(texts: Sequence[str], concepts: Sequence[str] = CONCEPTS) -> List[Tuple[int, str, str, str]]:
    """All (text index, concept, term, word) hits; empty means clean."""
    return [(i, c, term, w) for i, t in enumerate(texts) for c in concepts for term, w in lexical_hits(t, c)]


def resolve_run(arg: str, pattern: str) -> Path:
    """A run file path, or the newest runs/{pattern} when arg is 'latest'."""
    if arg != "latest":
        return Path(arg)
    found = sorted(RUNS.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no runs/{pattern}")
    return found[-1]


def load_screen(arg: str, model: str) -> Dict:
    """The G0 screen JSON: {'calibration': [...16], 'knobs': [...32], ...}."""
    path = resolve_run(arg, f"screen_{model.split('/')[-1]}_*.json")
    screen = json.loads(path.read_text())
    screen["path"] = str(path)
    assert not set(screen["calibration"]) & set(screen["knobs"]), "calibration and knob concepts overlap"
    return screen


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


@torch.no_grad()
def concept_vectors_at(lm: SteeredLM, words: Sequence[str], layers: Sequence[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Per layer: residual at the last prompt token, mean-subtracted over words within each
    template, then averaged over TEMPLATES. Raw (un-normalised) vectors."""
    layers = list(layers)
    per_template = {l: [] for l in layers}
    for tpl in TEMPLATES:
        acts = {l: [] for l in layers}
        for w in words:
            _, caps = lm.score(lm.encode(chat_prefix(tpl.format(w=w))), capture_layers=layers)
            for l in layers:
                acts[l].append(caps[l][-1])
        for l in layers:
            a = torch.stack(acts[l])
            per_template[l].append(a - a.mean(0, keepdim=True))
    out = {}
    for l in layers:
        a = torch.stack(per_template[l]).mean(0)
        out[l] = {w: a[i] for i, w in enumerate(words)}
    return out


def concept_vectors(lm: SteeredLM, words: Sequence[str], layer: int) -> Dict[str, torch.Tensor]:
    return concept_vectors_at(lm, words, [layer])[layer]


def neuron_direction(lm: SteeredLM, layer: int, index: int) -> torch.Tensor:
    """Write vector of MLP neuron `index`: adding d*W_down[:, i] equals raising its activation by d."""
    return lm.layers[layer].mlp.down_proj.weight[:, index].detach().float().cpu()


def random_direction(d_model: int, seed: int) -> torch.Tensor:
    return torch.randn(d_model, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def residual_scale(lm: SteeredLM, layer: int, texts: Sequence[str] = CALIB_PASSAGES) -> float:
    """Median residual norm at `layer` over the calibration passages, skipping position 0 (the attention sink)."""
    norms = []
    for t in texts:
        _, caps = lm.score(lm.encode(chat_prefix(t)), capture_layers=[layer])
        norms.append(caps[layer][1:].norm(dim=-1))
    return float(torch.cat(norms).median())
