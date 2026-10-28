"""A causal LM whose residual stream can be steered at chosen token positions.

Every experiment in the ladder is a token sequence plus a schedule of
interventions. An intervention adds a vector to one layer's output at a range of
absolute token positions. Because the schedule is position-indexed, re-running a
prefill with the same schedule reproduces exactly the KV cache that incremental
generation built, so any experimental condition can be scored with one forward
pass instead of replaying a conversation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Pinned Hugging Face revisions, so every run of a model reads the same weights and tokenizer.
REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
}


@dataclass
class Intervention:
    layer: int
    vector: torch.Tensor  # (d_model,)
    start: int  # first absolute position steered
    end: Optional[int] = None  # exclusive; None keeps steering open-ended

    def span(self, offset: int, length: int) -> Optional[slice]:
        """Steered slice of a chunk covering absolute positions [offset, offset + length), or None."""
        lo = max(self.start, offset)
        hi = offset + length if self.end is None else min(self.end, offset + length)
        return slice(lo - offset, hi - offset) if lo < hi else None


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SteeredLM:
    def __init__(self, name: str, device: Optional[str] = None, dtype=torch.float32, revision: Optional[str] = None):
        self.name = name
        self.revision = revision or REVISIONS.get(name)
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(name, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, revision=self.revision)
        self.model.to(self.device).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self._schedule: Dict[int, List[Tuple[Intervention, torch.Tensor]]] = {}
        self._offset = 0
        self._capture_layers: set = set()
        self._captured: Dict[int, List[torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))

    def provenance(self) -> Dict:
        """What produced a run: weights, numerics, library versions and the code revision."""
        rev = Path(__file__).resolve().parents[1] / "REVISION"
        return {
            "model": self.name, "revision": self.revision, "dtype": str(self.model.dtype),
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "device": torch.cuda.get_device_name() if self.device == "cuda" else self.device,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "code": rev.read_text().strip() if rev.exists() else None,
        }

    # -- hooks -------------------------------------------------------------
    def _hook(self, i: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Spans are computed from Python ints, so steering never forces a device sync.
            for iv, v in self._schedule.get(i, ()):
                sl = iv.span(self._offset, h.shape[1])
                if sl is not None:
                    h[:, sl] += v
            if i in self._capture_layers:
                self._captured.setdefault(i, []).append(h[0].detach().float().cpu())
            return output

        return hook

    def _forward(self, ids: Sequence[int], offset: int, cache):
        self._offset = offset
        x = torch.tensor([list(ids)], device=self.device)
        out = self.model.model(input_ids=x, past_key_values=cache, use_cache=cache is not None)
        return out.last_hidden_state[0]

    def _set_schedule(self, schedule: Sequence[Intervention]):
        """Group by layer and move each vector to the device once, not once per forward."""
        dtype = self.model.dtype
        self._schedule = {}
        for iv in schedule:
            v = iv.vector.to(device=self.device, dtype=dtype)
            self._schedule.setdefault(iv.layer, []).append((iv, v))

    # -- public API --------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def token_id(self, token: str) -> int:
        tid = self.tok.convert_tokens_to_ids(token)
        if tid is None or tid == self.tok.unk_token_id:
            ids = self.encode(token)
            assert len(ids) == 1, f"{token!r} is not a single token: {ids}"
            tid = ids[0]
        return tid

    @torch.no_grad()
    def score(
        self,
        ids: Sequence[int],
        schedule: Sequence[Intervention] = (),
        capture_layers: Sequence[int] = (),
        positions: Optional[Sequence[int]] = None,
    ):
        """One prefill. Returns (logits, {layer: (T, d) residuals}).

        Logits are for the last position, or (len(positions), vocab) when positions are given.
        """
        try:
            self._set_schedule(schedule)
            self._capture_layers = set(capture_layers)
            self._captured = {}
            h = self._forward(ids, 0, None)
            logits = self.model.lm_head(h[-1] if positions is None else h[list(positions)]).float().cpu()
            caps = {k: torch.cat(v) for k, v in self._captured.items()}
        finally:
            self._schedule, self._capture_layers, self._captured = {}, set(), {}
        return logits, caps

    @torch.no_grad()
    def generate(
        self,
        ids: Sequence[int],
        schedule: Sequence[Intervention] = (),
        max_new_tokens: int = 200,
        stop_ids: Sequence[int] = (),
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        seed: int = 0,
    ) -> List[int]:
        """Sample a continuation. Interventions at future positions apply as tokens are generated.

        Top-k runs on the device, so each step moves only k logits to the host.
        """
        gen = torch.Generator().manual_seed(seed)
        stops = set(stop_ids) | {self.tok.eos_token_id}
        new: List[int] = []
        try:
            self._set_schedule(schedule)
            cache = DynamicCache()
            h = self._forward(ids, 0, cache)
            pos = len(ids)
            for _ in range(max_new_tokens):
                logits = self.model.lm_head(h[-1]).float()
                t = sample(logits, temperature, top_p, top_k, gen)
                new.append(t)
                if t in stops:
                    break
                h = self._forward([t], pos, cache)
                pos += 1
        finally:
            self._schedule = {}
        return new


def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen) -> int:
    """Temperature, top-k, then top-p sampling. `logits` may live on any device; `gen` is a CPU generator."""
    if temperature <= 0:
        return int(logits.argmax())
    if top_k:
        vals, idx = torch.topk(logits, top_k)  # sorted, on the device
        vals, idx = vals.cpu(), idx.cpu()
    else:
        vals, idx = torch.sort(logits.cpu(), descending=True)
    probs = torch.softmax(vals / temperature, -1)
    if top_p < 1:
        keep = probs.cumsum(-1) - probs < top_p
        probs = probs * keep
    return int(idx[torch.multinomial(probs / probs.sum(), 1, generator=gen)])
