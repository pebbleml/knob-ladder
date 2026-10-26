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
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class Intervention:
    layer: int
    vector: torch.Tensor  # (d_model,)
    start: int  # first absolute position steered
    end: Optional[int] = None  # exclusive; None keeps steering open-ended

    def mask(self, pos: torch.Tensor) -> torch.Tensor:
        m = pos >= self.start
        if self.end is not None:
            m &= pos < self.end
        return m


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SteeredLM:
    def __init__(self, name: str, device: Optional[str] = None, dtype=torch.float32):
        self.name = name
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.model.to(self.device).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self._schedule: List[Intervention] = []
        self._offset = 0
        self._capture_layers: set = set()
        self._captured: Dict[int, List[torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))

    # -- hooks -------------------------------------------------------------
    def _hook(self, i: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._schedule:
                pos = torch.arange(self._offset, self._offset + h.shape[1], device=h.device)
                for iv in self._schedule:
                    if iv.layer != i:
                        continue
                    m = iv.mask(pos)
                    if m.any():
                        h[:, m] += iv.vector.to(device=h.device, dtype=h.dtype)
            if i in self._capture_layers:
                self._captured.setdefault(i, []).append(h[0].detach().float().cpu())
            return output

        return hook

    def _forward(self, ids: Sequence[int], offset: int, cache):
        self._offset = offset
        x = torch.tensor([list(ids)], device=self.device)
        out = self.model.model(input_ids=x, past_key_values=cache, use_cache=cache is not None)
        return out.last_hidden_state[0]

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
    ):
        """One prefill. Returns (last-position logits, {layer: (T, d) residuals})."""
        self._schedule = list(schedule)
        self._capture_layers = set(capture_layers)
        self._captured = {}
        try:
            h = self._forward(ids, 0, None)
            logits = self.model.lm_head(h[-1]).float().cpu()
            caps = {k: torch.cat(v) for k, v in self._captured.items()}
        finally:
            self._schedule, self._capture_layers, self._captured = [], set(), {}
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
        """Sample a continuation. Interventions at future positions apply as tokens are generated."""
        gen = torch.Generator().manual_seed(seed)
        stops = set(stop_ids) | {self.tok.eos_token_id}
        self._schedule = list(schedule)
        cache = DynamicCache()
        new: List[int] = []
        try:
            h = self._forward(ids, 0, cache)
            pos = len(ids)
            for _ in range(max_new_tokens):
                logits = self.model.lm_head(h[-1]).float().cpu()
                t = sample(logits, temperature, top_p, top_k, gen)
                new.append(t)
                if t in stops:
                    break
                h = self._forward([t], pos, cache)
                pos += 1
        finally:
            self._schedule = []
        return new


def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, -1)
    if top_p < 1:
        sp, si = torch.sort(probs, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
    return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))
