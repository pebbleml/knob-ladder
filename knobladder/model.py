"""A causal LM whose residual stream can be steered at chosen token positions.

Every experiment in the ladder is a token sequence plus a schedule of
interventions. An intervention adds a vector to one layer's output at a range of
absolute token positions. Because the schedule is position-indexed, re-running a
prefill with the same schedule reproduces exactly the KV cache that incremental
generation built, so any experimental condition can be scored with one forward
pass instead of replaying a conversation.

Several sequences can run as rows of one `Batch`. A row's tokens need not fill contiguous
columns: columns it does not use are masked out and its position ids count only its own
tokens, so each row computes what it would alone, and steering lands on each row's own
absolute positions.
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


class Batch:
    """Rows sharing one KV cache. `ids[r]` are row r's tokens in the cache, in column order."""

    def __init__(self, n: int):
        self.ids: List[List[int]] = [[] for _ in range(n)]
        self.scheds: List[List[Intervention]] = [[] for _ in range(n)]  # what those tokens were steered with
        self.cache = DynamicCache()
        self.mask: Optional[torch.Tensor] = None  # (rows, columns), True where a row has a token
        self.holes = False  # some row skips a column; until then no mask is passed, as for one row

    def select(self, rows: List[int]):
        """Keep only `rows`, in that order."""
        self.cache.batch_select_indices(list(rows))
        self.mask = None if self.mask is None else self.mask[list(rows)]
        self.ids = [self.ids[r] for r in rows]
        self.scheds = [self.scheds[r] for r in rows]

    def truncate(self, r: int, n: int):
        """Keep only row r's first n tokens (a transcript that dropped a think block); compact() then
        frees the columns. The kept keys are unchanged, so this equals a fresh prefill of those n tokens."""
        if n == len(self.ids[r]):
            return
        cols = self.mask[r].nonzero().flatten()
        self.mask[r, cols[n:]] = False
        self.ids[r] = self.ids[r][:n]
        self.holes = True

    def compact(self):
        """Drop unused columns and right-align every row. Order within a row is kept and keys carry
        their rotary positions, so attention is unchanged."""
        if not self.holes:
            return
        keep = self.mask.cpu()
        n = [len(r) for r in self.ids]
        assert keep.sum(1).tolist() == n, "batch mask out of step with its ids"
        width = max(n)
        idx = torch.zeros(len(n), width, dtype=torch.long)
        for r, k in enumerate(keep):
            idx[r, width - n[r]:] = k.nonzero().flatten()
        for layer in self.cache.layers:
            k, v = layer.keys, layer.values
            i = idx.to(k.device)[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
            layer.keys, layer.values = k.gather(2, i), v.gather(2, i)
        self.mask = (torch.arange(width) >= width - torch.tensor(n)[:, None]).to(self.mask.device)
        self.holes = min(n) < width

    def fork(self, n: int) -> "Batch":
        """n copies of a one-row batch."""
        assert len(self.ids) == 1
        b = Batch(n)
        for i, layer in enumerate(self.cache.layers):
            b.cache.update(layer.keys.expand(n, -1, -1, -1), layer.values.expand(n, -1, -1, -1), i)
        b.ids = [list(self.ids[0]) for _ in range(n)]
        b.scheds = [list(self.scheds[0]) for _ in range(n)]
        b.mask, b.holes = self.mask.expand(n, -1).clone(), self.holes
        return b


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
        self.tok = AutoTokenizer.from_pretrained(name, revision=self.revision)
        if device is None and torch.cuda.device_count() > 1:
            # Too big for one card: split the layers over every visible GPU (run one process
            # per GPU group, e.g. CUDA_VISIBLE_DEVICES=0,1, to run several copies side by side).
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, revision=self.revision,
                                                              device_map="auto")
            self.device = str(self.model.get_input_embeddings().weight.device)
        else:
            self.device = device or default_device()
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, revision=self.revision)
            self.model.to(self.device)
        self.model.eval()
        self.layers = self.model.model.layers
        self._layer_device = [next(layer.parameters()).device for layer in self.layers]
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self._schedule: Dict[int, List[Tuple[int, Intervention, torch.Tensor]]] = {}
        self._rows: List[Tuple[int, int, int]] = []
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
            "device": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else self.device,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "code": rev.read_text().strip() if rev.exists() else None,
        }

    # -- hooks -------------------------------------------------------------
    def _hook(self, i: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Spans are computed from Python ints, so steering never forces a device sync.
            for r, iv, v in self._schedule.get(i, ()):
                col, pos, n = self._rows[r]  # row r's new tokens: columns [col, col + n), positions [pos, pos + n)
                sl = iv.span(pos, n)
                if sl is not None:
                    h[r, col + sl.start:col + sl.stop] += v
            if i in self._capture_layers:
                self._captured.setdefault(i, []).append(h[0].detach().float().cpu())
            return output

        return hook

    def _forward(self, ids: Sequence[int], offset: int, cache):
        self._rows = [(0, offset, len(ids))]
        x = torch.tensor([list(ids)], device=self.device)
        out = self.model.model(input_ids=x, past_key_values=cache, use_cache=cache is not None)
        return out.last_hidden_state[0]

    def _extend(self, b: Batch, chunks: Sequence[Sequence[int]]) -> torch.Tensor:
        """Append chunks[r] to row r, right-aligned so the last column holds every non-empty row's newest
        token. Returns the last column's hidden states, (rows, d)."""
        width = max(map(len, chunks))
        pad = self.tok.pad_token_id or 0
        x, pos, keep, rows = [], [], [], []
        for r, c in enumerate(chunks):
            k, p = width - len(c), len(b.ids[r])
            x.append([pad] * k + list(c))
            pos.append([0] * k + list(range(p, p + len(c))))
            keep.append([False] * k + [True] * len(c))
            rows.append((k, p, len(c)))
            b.ids[r] += c
        b.holes = b.holes or any(len(c) < width for c in chunks)
        new = torch.tensor(keep, device=self.device)
        b.mask = new if b.mask is None else torch.cat([b.mask, new], 1)
        self._rows = rows
        out = self.model.model(input_ids=torch.tensor(x, device=self.device),
                               attention_mask=b.mask if b.holes else None,
                               position_ids=torch.tensor(pos, device=self.device),
                               past_key_values=b.cache, use_cache=True)
        return out.last_hidden_state[:, -1]

    def _set_schedule(self, schedules: Sequence[Sequence[Intervention]]):
        """One schedule per row. Group by layer and move each vector to its layer's device once, not once per forward."""
        dtype = self.model.dtype
        self._schedule = {}
        for r, schedule in enumerate(schedules):
            for iv in schedule:
                v = iv.vector.to(device=self._layer_device[iv.layer], dtype=dtype)
                self._schedule.setdefault(iv.layer, []).append((r, iv, v))

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
            self._set_schedule([schedule])
            self._capture_layers = set(capture_layers)
            self._captured = {}
            h = self._forward(ids, 0, None)
            logits = self.model.lm_head(h[-1] if positions is None else h[list(positions)]).float().cpu()
            caps = {k: torch.cat(v) for k, v in self._captured.items()}
        finally:
            self._schedule, self._rows, self._capture_layers, self._captured = {}, [], set(), {}
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
        return self.generate_batch([ids], [schedule], max_new_tokens, stop_ids, temperature, top_p, top_k, [seed])[0]

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        schedules: Sequence[Sequence[Intervention]],
        max_new_tokens: int = 200,
        stop_ids: Sequence[int] = (),
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        seeds: Optional[Sequence[int]] = None,
        batch: Optional[Batch] = None,
        first_logits: bool = False,
    ):
        """`generate` for every row at once. Row r continues prompts[r] under schedules[r] and samples with
        its own generator seeded seeds[r], so it draws what `generate` would draw alone, up to batched
        numerics. A row stops on its own stop token; finished rows idle until all have stopped.

        With `batch`, rows continue that batch's cache: its rows must be prefixes of the prompts, and a
        schedule may only add windows over tokens not yet cached (cached tokens are never re-steered).
        Returns the new tokens per row, and with `first_logits` also the first step's (rows, vocab) logits.
        """
        n = len(prompts)
        b = batch or Batch(n)
        assert len(b.ids) == len(schedules) == n
        chunks = []
        for p, s, done, old in zip(prompts, schedules, b.ids, b.scheds):
            assert len(p) > len(done) and list(p[:len(done)]) == done, "a cached row must be a proper prefix of its prompt"
            assert len(s) >= len(old) and all(x is y for x, y in zip(old, s)), "a schedule may only grow"
            assert all(iv.start >= len(done) for iv in s[len(old):]), "a new window covers cached tokens"
            chunks.append(list(p[len(done):]))
        gens = [torch.Generator().manual_seed(s) for s in (seeds if seeds is not None else [0] * n)]
        stops = set(stop_ids) | {self.tok.eos_token_id}
        new: List[List[int]] = [[] for _ in range(n)]
        try:
            self._set_schedule(schedules)
            logits = self.model.lm_head(self._extend(b, chunks)).float()
            first = logits.cpu() if first_logits else None
            live = list(range(n))
            for step in range(max_new_tokens):
                for r, t in zip(live, sample_rows(logits[live], temperature, top_p, top_k, [gens[r] for r in live])):
                    new[r].append(t)
                live = [r for r in live if new[r][-1] not in stops]
                if not live or step == max_new_tokens - 1:
                    break  # the last token is never fed: the cache holds the prompt and all but the last new token
                alive = set(live)
                h = self._extend(b, [new[r][-1:] if r in alive else [] for r in range(n)])
                logits = self.model.lm_head(h).float()
        finally:
            self._schedule, self._rows = {}, []
        b.scheds = [list(s) for s in schedules]
        return (first, new) if first_logits else new

    @torch.no_grad()
    def branches(
        self,
        prefix: Sequence[int],
        schedule: Sequence[Intervention],
        tails: Sequence[Sequence[int]],
        max_new_tokens: int,
        stop_ids: Sequence[int] = (),
        size: int = 8,
    ) -> List[Tuple[torch.Tensor, List[int]]]:
        """(first-step logits, greedy tokens) of prefix + tail for every tail, `size` tails at a time.
        The prefix is prefilled once; steering is by absolute position, so this equals one prefill per branch."""
        base = Batch(1)
        try:
            self._set_schedule([schedule])
            self._extend(base, [list(prefix)])
        finally:
            self._schedule, self._rows = {}, []
        base.scheds = [list(schedule)]
        out = []
        for g in range(0, len(tails), size):
            group = tails[g:g + size]
            first, new = self.generate_batch([list(prefix) + list(t) for t in group], [schedule] * len(group),
                                             max_new_tokens, stop_ids, temperature=0,
                                             batch=base.fork(len(group)), first_logits=True)
            out += list(zip(first, new))
        return out


def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen) -> int:
    """Temperature, top-k, then top-p sampling. `logits` may live on any device; `gen` is a CPU generator."""
    return sample_rows(logits[None], temperature, top_p, top_k, [gen])[0]


def sample_rows(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gens) -> List[int]:
    """`sample` for each row of (rows, vocab) logits, each with its own generator. Top-k runs on the
    device, so each step moves only k logits per row to the host."""
    if temperature <= 0:
        return logits.argmax(-1).tolist()
    if top_k:
        vals, idx = torch.topk(logits, top_k)  # sorted, on the device
        vals, idx = vals.cpu(), idx.cpu()
    else:
        vals, idx = torch.sort(logits.cpu(), descending=True)
    out = []
    for v, i, gen in zip(vals, idx, gens):
        probs = torch.softmax(v / temperature, -1)
        if top_p < 1:
            keep = probs.cumsum(-1) - probs < top_p
            probs = probs * keep
        out.append(int(i[torch.multinomial(probs / probs.sum(), 1, generator=gen)]))
    return out
