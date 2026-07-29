"""LangSetModel: an LLM (LoRA) + a learned emit head that maps input text -> a latent in a bespoke geometry.

The latent lives in the model's OWN hidden space, and the output is Sentence-Transformer-shaped (`encode`,
`get_sentence_embedding_dimension`, `as_sentence_transformer`) so the trained model drops straight into SetFit
as a `model_body`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:  # type-only: no runtime import cost, and the optional-dep types stay import-safe
    from sentence_transformers import SentenceTransformer
    from transformers import Cache, PretrainedConfig, PreTrainedTokenizerBase
    from ty_extensions import (
        Unknown,
    )  # ty's gradual type — for genuine passthrough boundaries (not typing.Any)


class _HiddenOutput(Protocol):
    """A backbone forward result, read only for hidden states: a text tower exposes `last_hidden_state`, a raw
    ForCausalLM exposes `hidden_states` (and `logits` when an lm_head is present). `_last_hidden` reads whichever
    is there — see its getattr fallback."""

    hidden_states: tuple[torch.Tensor, ...]


class _Backbone(Protocol):
    """The structural surface `LangSetModel` uses on its (peft-wrapped, maybe text-tower-unwrapped) backbone —
    one Protocol spanning plain Llama/Qwen `ForCausalLM`, text towers, and Gemma-E PLE models. The PLE-only
    `get_per_layer_inputs` is deliberately NOT declared: it exists on Gemma-E alone and is reached through the
    `_ple_dim` runtime guard, so pinning it here would wrongly exclude every non-PLE backbone."""

    config: PretrainedConfig

    def __call__(self, **kwargs: object) -> _HiddenOutput: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def parameters(self, recurse: bool = ...) -> Iterator[nn.Parameter]: ...
    def state_dict(self, *args: object, **kwargs: object) -> dict[str, Any]: ...
    def load_state_dict(self, *args: object, **kwargs: object) -> object: ...


class EmitHead(nn.Module):
    """Learned query tokens whose post-backbone hidden states are read out as the latent (the LLM's vector 'mouth').

    One switch drives everything: `multi_latent`. OFF (default) → CONTINUOUS emission, a single raw vector, the
    SetFit embedding path (uses `out_proj`). ON → the TOKEN-NATIVE sequence stack: each emission is a FINITE
    SCALAR QUANTIZATION (FSQ) of the target embedding — a learned `down_proj`/`up_proj` bottleneck into a FIXED
    grid (`fsq_dim` dims, each rounded to `fsq_levels` levels). The grid is not learned and cannot drift or
    collapse (unlike a VQ codebook), so it stays stable while the encoder/EMA-twin keep SPECIALIZING; near-
    continuous precision comes from the grid resolution (like fixed-point), and the dims are INDEPENDENT so there
    is no residual cascade. The emitter predicts each dim's digit (a small classification) plus a STOP folded into
    dim-0's softmax, so termination/count is free. `multi_latent` is a task selector, not a tuning knob: `fsq_dim`
    / `fsq_levels` are set-and-forget internals.

    `dropout` is trained ON when >0 so MC-dropout could later serve as an uncertainty signal. The head stays fp32
    even when the backbone is bf16.
    """

    def __init__(
        self,
        h: int,
        d: int,
        n_latents: int = 1,
        dropout: float = 0.0,
        eos_id: int = 0,
        multi_latent: bool = False,
        fsq_dim: int = 128,
        fsq_levels: int = 8,
        fsq_emit: bool = False,
        code_emit: bool = False,
        n_codes: int = 0,
        code_tau: float = 0.07,
        res_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.q = nn.Parameter(
            torch.randn(n_latents, h) * 0.02
        )  # one query token per emitted latent
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(h, d)  # hidden -> latent (targets / retrieval bank)
        self.in_proj = nn.Linear(d, h)  # latent -> hidden (inverse projection)
        self.eos_id = eos_id
        self.multi_latent = multi_latent
        # fsq_emit: SINGLE-vector retrieval, but the embedding is GENERATED via the FSQ digit head (level_proj -> soft
        # code -> up_proj) instead of the linear out_proj — so it's produced through the same generative machinery CPT
        # rewrites, letting injected knowledge reach the retrieval vector. Trainer routing stays on multi_latent (off).
        # Default False => byte-identical: forward() uses out_proj and the FSQ projections match `multi_latent` alone.
        self.fsq_emit = fsq_emit
        self.fsq_dim, self.fsq_levels = fsq_dim, fsq_levels
        _mk_fsq = multi_latent or fsq_emit
        _mk_stop = _mk_fsq or code_emit
        self.down_proj = nn.Linear(d, fsq_dim) if _mk_fsq else None  # d -> FSQ bottleneck (learned)
        self.up_proj = nn.Linear(fsq_dim, d) if _mk_fsq else None  # FSQ -> d (learned decoder)
        self.level_proj = (
            nn.Linear(h, fsq_dim * fsq_levels) if _mk_fsq else None
        )  # per-dim digit logits
        self.stop_proj = (
            nn.Linear(h, 1) if _mk_stop else None
        )  # STOP logit (folded into dim-0 softmax; unused on the single-latent path)
        # CODEBOOK EMISSION: replace the FSQ product-quantizer with ONE softmax over a fixed codebook. The head
        # projects the hidden to a query, scores it against every code, and COMMITS to the softmax MIXTURE of
        # codes rather than to a per-dim digit grid. Emitting a SET (maze frontier: which cells) is then a single
        # normalized decision over the set's alphabet instead of `fsq_dim` independent ones, so mass must be
        # ALLOCATED across the active members. `code` is a buffer, not a parameter: like the FSQ grid it cannot
        # drift or collapse, so the geometry stays a constant of the run (see set_code).
        # STATE + RESIDUAL. `res_dim` splits the latent in two: the first `d - res_dim` dims are the STATE, a
        # mixture over the named codebook, and the last `res_dim` dims are a RESIDUAL that nothing names. The
        # split is what makes a named alphabet safe to adopt: an alphabet is a CEILING (whatever it cannot name,
        # the emission cannot carry), and the residual is where the unnamed goes. It also restores a
        # self-supervised component — the residual is shaped by the twin/recon terms, not by the labels.
        #
        # This differs from `emb_slots` in kind, not degree. emb_slots DECODES a facet out of a latent whose
        # meaning came from the twin; the state half here CONSTITUTES the latent from the alphabet. A decode head
        # cannot create information the model never computed (the frontier-head arm sat at its base-rate floor
        # for exactly that reason); building the emission out of the members puts it there.
        self.code_emit = code_emit
        self.n_codes, self.code_tau = n_codes, code_tau
        self.res_dim = int(res_dim)
        self.state_dim = d - self.res_dim
        if code_emit:
            assert n_codes > 0, "code_emit requires n_codes > 0"
            assert 0 <= self.res_dim < d, (
                f"res_dim must satisfy 0 <= res_dim < latent_dim ({d}); got {res_dim}"
            )
            self.register_buffer(
                "code", torch.zeros(n_codes, self.state_dim)
            )  # [C, state_dim] via set_code()
            # CONCEPT FACETS. `concept_spans` carves the codebook into named groups: facet f owns member rows
            # [m_lo, m_hi) and latent dims [d_lo, d_hi), and its rows are ZERO outside its own dims. So one
            # matmul yields every facet's logits at once, while each facet gets its OWN softmax — an album can
            # be a wide mixture of `vocals` and certain of its `tempo` at the same time, and a filing's `stage`
            # can be confident while its `result` is undecided. One span = the plain single-alphabet case.
            self.concept_spans: list[tuple[int, int, int, int]] = [(0, n_codes, 0, self.state_dim)]
            self.concept_names: list[str] = []
            self.query_proj: Optional[nn.Linear] = nn.Linear(
                h, self.state_dim
            )  # hidden -> state query
            # residual head: only built when a residual is asked for, so res_dim=0 stays byte-identical
            self.res_proj: Optional[nn.Linear] = (
                nn.Linear(h, self.res_dim) if self.res_dim else None
            )
        else:
            self.query_proj = None
            self.res_proj = None

    def forward(
        self, hid_emit: torch.Tensor
    ) -> torch.Tensor:  # [B, n_latents, h] -> [B, n_latents, d]
        if self.code_emit:
            # EMBEDDING AS A SUPERPOSITION. The retrieval vector is a mixture over the NAMED alphabet, so it is
            # interpretable by construction: project it back onto the codebook and you get the members it is
            # made of, in the domain's own vocabulary, with no probe. Nothing supervises the weights here —
            # the ordinary contrastive/recon objective decides them, so an alphabet buys legibility without
            # requiring per-row state labels. (`fsq_emit` established this shape: a retrieval vector generated
            # THROUGH the emission head rather than a plain linear.)
            lg, _ = self.emit_logits(hid_emit)  # [B, nl, 1, n_codes]
            p = self.concept_probs(lg)  # [B, nl, n_codes] per-facet softmax
            state = F.normalize(p @ self.code, p=2, dim=-1)
            if self.res_dim == 0:
                return state
            res = self.residual(hid_emit)  # what the alphabet cannot name
            return torch.cat([state, res], -1) * (0.5**0.5)
        if self.fsq_emit:
            lg, _ = self.emit_logits(hid_emit)  # [B, nl, fsq_dim, fsq_levels] digit logits
            levels = torch.arange(self.fsq_levels, device=lg.device, dtype=torch.float32)
            soft = (lg.float().softmax(-1) * levels).sum(
                -1
            )  # [B, nl, fsq_dim] expected digit (differentiable)
            assert self.up_proj is not None
            return F.normalize(self.up_proj(soft), p=2, dim=-1)  # generated-FSQ retrieval vector
        return F.normalize(self.out_proj(self.drop(hid_emit.float())), p=2, dim=-1)

    def feedback(self, latent: torch.Tensor) -> torch.Tensor:  # [B, ..., d] -> [B, ..., h]
        return self.in_proj(latent.float()).to(
            latent.dtype
        )  # head stays fp32; match the backbone's dtype (bf16)

    def fsq(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Finite scalar quantize z [.., fsq_dim] -> (zq straight-through, digits). Both live in the SAME [0, L-1]
        space so training (up_proj(zq)) and inference (up_proj(digits)) reconstruct identically."""
        lvl = self.fsq_levels - 1
        zb = (torch.tanh(z) + 1.0) * 0.5 * lvl  # [0, L-1]
        zr = zb.round().clamp(0, lvl)
        zq = zb + (zr - zb).detach()  # straight-through, in [0, L-1]
        return zq, zr.long()

    def set_code(self, code: torch.Tensor) -> None:
        """Install the fixed codebook [n_codes, state_dim] (rows L2-normalized). Call once after construction."""
        assert self.code_emit and tuple(code.shape) == tuple(self.code.shape), (
            f"set_code expects [{self.n_codes}, {self.code.size(-1)}], got {tuple(code.shape)}"
        )
        self.code.copy_(F.normalize(code.float(), dim=-1).to(self.code.device))

    def set_concepts(self, facets: "list[tuple[str, torch.Tensor, int]]") -> None:
        """Install named concept facets: [(name, codes [n_members_f, dims_f], dims_f), ...].

        Lays them out block-wise down the state half — facet f's member rows carry its codes in its own dim
        slice and zeros elsewhere — so `query @ code.T` produces all facets' logits in one matmul while each
        facet keeps a separate softmax."""
        total_m = sum(c.shape[0] for _, c, _ in facets)
        total_d = sum(d for _, _, d in facets)
        # The alphabet is DISCOVERED from the corpus, so its size is not knowable when the head is built —
        # `n_codes` at construction is only an upper bound. Resize to what the data actually contained rather
        # than making the caller predict it; the true size is persisted in the config, so a reload rebuilds
        # at the right width.
        if total_m != self.n_codes:
            self.n_codes = total_m
            self.register_buffer(
                "code", torch.zeros(total_m, self.state_dim, device=self.code.device)
            )
        assert total_d <= self.state_dim, (
            f"concept facets need {total_d} dims but the state half is {self.state_dim} "
            f"(latent_dim {self.state_dim + self.res_dim} minus res_dim {self.res_dim})"
        )
        block = torch.zeros(self.n_codes, self.state_dim)
        spans, names, m0, d0 = [], [], 0, 0
        for name, codes, dims in facets:
            n_m = codes.shape[0]
            block[m0 : m0 + n_m, d0 : d0 + dims] = F.normalize(codes.float(), dim=-1)[:, :dims]
            spans.append((m0, m0 + n_m, d0, d0 + dims))
            names.append(name)
            m0, d0 = m0 + n_m, d0 + dims
        self.code.copy_(block.to(self.code.device))
        self.concept_spans, self.concept_names = spans, names

    def concept_probs(self, lg: torch.Tensor) -> torch.Tensor:
        """Emission logits -> per-facet probabilities, concatenated back into one [.., n_codes] vector.

        Each facet is normalized WITHIN itself, so mass never leaks between facets: `vocals` summing to 1 says
        nothing about how certain `tempo` is."""
        flat = lg.float().squeeze(-2)  # [.., n_codes]
        out = torch.zeros_like(flat)
        for m_lo, m_hi, _, _ in self.concept_spans:
            out[..., m_lo:m_hi] = flat[..., m_lo:m_hi].softmax(-1)
        return out

    def residual(self, hid: torch.Tensor) -> torch.Tensor:
        """[.., h] -> the unnamed half of the emission [.., res_dim], L2-normalized. Empty when res_dim == 0."""
        if self.res_proj is None:
            return hid.new_zeros(*hid.shape[:-1], 0)
        return F.normalize(self.res_proj(self.drop(hid.float())), dim=-1)

    def encode(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Target embedding -> (digits [.., fsq_dim], recon [.., d]). Grad flows to down/up_proj (recon loss)."""
        if (
            self.code_emit
        ):  # no quantizer: the target IS the reconstruction, so teacher forcing feeds it exactly
            # the codebook spans only the STATE half, so score the target's state dims against it — the residual
            # dims are unnamed by construction and have no nearest code
            ts = F.normalize(t.float()[..., : self.state_dim], dim=-1)
            idx = (ts @ self.code.t()).argmax(-1, keepdim=True)
            return (
                idx,
                t.float(),
            )  # (nearest code [.., 1], recon [.., d] = the target itself, lossless)
        assert self.down_proj is not None and self.up_proj is not None
        zq, digits = self.fsq(self.down_proj(t.float()))
        return digits, self.up_proj(zq)

    def reconstruct(
        self, digits: torch.Tensor
    ) -> torch.Tensor:  # digits [.., fsq_dim] in [0, L-1] -> [.., d]
        if self.code_emit:  # a "digit" is a code index; reconstruction is the code itself
            return self.code[digits.long().squeeze(-1)]
        assert self.up_proj is not None
        return self.up_proj(digits.float())

    def emit_logits(self, hid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[.., h] -> (per-dim digit logits [.., fsq_dim, fsq_levels], STOP logit [.., 1]).

        Codebook mode returns the SAME rank with fsq_dim collapsed to 1: [.., 1, n_codes] — one softmax over the
        alphabet instead of fsq_dim softmaxes over levels. Every caller downstream (STOP fold, commit, argmax)
        is shape-agnostic, so the rollout loops are untouched."""
        assert self.stop_proj is not None
        if self.code_emit:
            assert self.query_proj is not None
            q = F.normalize(self.query_proj(self.drop(hid.float())), dim=-1)  # [.., d]
            return (q @ self.code.t()).unsqueeze(-2) / self.code_tau, self.stop_proj(hid.float())
        assert self.level_proj is not None
        lg = self.level_proj(self.drop(hid.float())).unflatten(-1, (self.fsq_dim, self.fsq_levels))
        return lg, self.stop_proj(hid.float())

    def commit(
        self, lg: torch.Tensor, sample: bool = False, hid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Emission logits -> the latent that gets FED BACK to advance one tick. The FSQ branch is the historical
        argmax(digits) -> up_proj, byte-for-byte. The codebook branch commits to the softmax MIXTURE of codes: a
        genuine SUPERPOSITION of the members it is holding, not a single winner, which is the whole point of
        emitting a set. `sample` draws instead of taking the mode (FSQ only; the mixture is already the full
        distribution)."""
        if self.code_emit:
            p = self.concept_probs(lg)  # [.., n_codes] each facet normalized within itself
            state = F.normalize(
                p @ self.code, dim=-1
            )  # [.., state_dim] facet mixtures, laid out side by side
            if self.res_dim == 0:
                return state
            assert hid is not None, (
                "commit(): a residual emission needs the emit hidden (pass hid=)"
            )
            # halve each half's norm so the concatenation is unit-norm and neither half can dominate the
            # feedback vector purely by scale
            return torch.cat([state, self.residual(hid)], -1) * (0.5**0.5)
        if sample:
            pred = torch.multinomial(
                torch.softmax(lg.float(), -1).reshape(-1, self.fsq_levels), 1
            ).reshape(*lg.shape[:-2], -1)
        else:
            pred = lg.argmax(-1)  # [.., fsq_dim] (own emission, never STOP)
        return self.reconstruct(pred)

    def stop_logit(self, hidden: torch.Tensor, tok_embed: nn.Module) -> torch.Tensor:
        """Alignment of the hidden state to the model's real EOS embedding (continuous-mode terminator)."""
        emb_eos = cast(torch.Tensor, tok_embed.weight)[self.eos_id].float()
        return hidden.float() @ emb_eos


def _cfg_int(config: PretrainedConfig, name: str) -> int:
    """Read a scalar (hidden_size / vocab_size) that may live on a composite config's text sub-config."""
    v = getattr(config, name, None)
    if v is None and hasattr(config, "text_config"):
        v = getattr(config.text_config, name, None)
    if v is None and hasattr(config, "get_text_config"):
        v = getattr(config.get_text_config(), name, None)
    if v is None:
        raise AttributeError(f"config has no {name}")
    return int(v)


def _cfg_set(config: PretrainedConfig, name: str, val: object) -> None:
    """Set a scalar on a config, mirroring `_cfg_int`'s composite-config handling: write it on the top-level
    config and on a `text_config` sub-config if that is where the field lives (e.g. vocab_size on a VLM)."""
    wrote = False
    if hasattr(config, name):
        setattr(config, name, val)
        wrote = True
    sub = getattr(config, "text_config", None)
    if sub is not None and hasattr(sub, name):
        setattr(sub, name, val)
        wrote = True
    if not wrote:  # brand-new field (e.g. overriding a default not present)
        setattr(config, name, val)


def _text_tower(model: _Backbone) -> _Backbone:
    """Descend a (peft-wrapped) causal/conditional-generation model to its TEXT transformer — the module that
    returns hidden states directly, with NO lm_head. Skips the huge-vocab logits projection (Gemma's 262k-vocab
    lm_head over a full sequence OOMs — we only ever read hidden states) and any vision tower. LoRA is injected
    in-place on the language Linears, so calling the text tower directly still applies it."""
    node = getattr(
        getattr(model, "base_model", model), "model", model
    )  # peft LoraModel -> underlying HF model
    for _ in range(4):
        nxt = getattr(node, "language_model", None)  # VLM container -> text tower
        if nxt is not None and nxt is not node:
            node = nxt
            continue
        if hasattr(node, "lm_head") and hasattr(
            node, "model"
        ):  # ForCausalLM/CondGen -> inner text model
            node = node.model
            continue
        break
    return cast(
        "_Backbone", node
    )  # getattr-descended node is `object` to ty; runtime is the text tower


def build_backbone(
    llm_model: str,
    lora_r: int,
    dropout: float,
    bf16: bool,
    dev: str,
    attn_implementation: str = "sdpa",
    train_base: bool = False,
    grad_ckpt: bool = False,
    lora_top_k: int = 0,
    pretrained: bool = True,
    arch_overrides: Optional[dict] = None,
    vocab_size: Optional[int] = None,
) -> _Backbone:
    def _top_k_layers(n_layers: int) -> Optional[list[int]]:
        # LoRA ONLY the top-K transformer layers -> fewer adapters, smaller activation graph -> bigger batch. Emission
        # reads the FINAL hidden state, so the top layers carry the task-shaping. 0 = all layers (default, unchanged).
        return (
            list(range(max(0, n_layers - lora_top_k), n_layers))
            if lora_top_k and n_layers
            else None
        )

    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    dt = torch.bfloat16 if bf16 else torch.float32

    if not pretrained:
        # RANDOM-INIT control arm: copy `llm_model`'s ARCHITECTURE (config) but NOT its weights, then train the whole
        # net (no LoRA — a low-rank adapter over random weights is meaningless). `arch_overrides` shrinks the net; a
        # decoupled tokenizer sets `vocab_size` so the fresh embedding table matches it. This is the "does pretraining
        # matter" baseline: same emission/anti-collapse machinery, zero inherited knowledge.
        from transformers import AutoConfig  # type: ignore[import-untyped]

        cfg = AutoConfig.from_pretrained(llm_model)
        for k, v in (arch_overrides or {}).items():
            _cfg_set(cfg, k, v)
        if vocab_size is not None:
            _cfg_set(cfg, "vocab_size", vocab_size)
        try:
            base = AutoModelForCausalLM.from_config(
                cfg, attn_implementation=attn_implementation or "sdpa"
            )
        except TypeError:  # older transformers: from_config takes no attn_implementation
            base = AutoModelForCausalLM.from_config(cfg)
        if hasattr(
            base, "language_model"
        ):  # unwrap conditional-generation wrapper to the text tower
            base = base.language_model
        base = base.to(dtype=dt).to(
            dev
        )  # weights already random; all params require_grad by default
        if grad_ckpt:
            base.config.use_cache = False
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        return _text_tower(base)

    from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]

    def _try_load(dtype_key: str, attn: Optional[str]) -> _Backbone:
        # sdpa (default) avoids materializing the O(S^2) eager-attention score matrix — a long seed (3072 tokens)
        # OOM'd a 0.6B model at 72GB on eager. attention_dropout is dropped for multimodal wrappers that reject it.
        kw: dict[str, Any] = {dtype_key: dt}
        if attn:
            kw["attn_implementation"] = attn
        try:
            return AutoModelForCausalLM.from_pretrained(llm_model, attention_dropout=dropout, **kw)
        except TypeError:  # multimodal wrappers (e.g. Gemma4ForConditionalGeneration) reject it
            return AutoModelForCausalLM.from_pretrained(llm_model, **kw)

    def _load(attn: Optional[str]) -> _Backbone:
        # transformers renamed `torch_dtype` -> `dtype` (~4.56); langset declares transformers>=4.41, so try the
        # new kwarg then fall back to the old one for broad version compat.
        try:
            return _try_load("dtype", attn)
        except TypeError:
            return _try_load("torch_dtype", attn)

    # A non-default impl (flash_attention_2, flex_attention, ...) is an EXPLICIT performance choice: refuse to silently
    # downgrade it. sdpa/eager may still fall back (a model that can't do sdpa -> eager) since those are just defaults;
    # with the default attn_implementation="sdpa", _strict is False so the load path is byte-identical to before.
    _strict = bool(attn_implementation) and attn_implementation not in ("sdpa", "eager")
    try:
        base = _load(attn_implementation or None)
    except (
        ValueError,
        ImportError,
        RuntimeError,
        TypeError,
    ) as e:  # this model/transformers version can't do the impl
        if _strict:
            raise RuntimeError(
                f"attn_implementation={attn_implementation!r} was requested but FAILED to load "
                f"({type(e).__name__}: {e}). Refusing to silently fall back to a slower kernel — install flash-attn "
                "(and use bf16 + a supported head_dim), or pass attn_implementation='sdpa' explicitly."
            ) from e
        if not attn_implementation:
            raise
        base = _load(None)  # sdpa/eager only: fall back to the model's default attention
    if hasattr(base, "language_model"):  # unwrap conditional-generation wrapper to the text tower
        base = base.language_model
    base = cast("Unknown", base).to(
        dev
    )  # raw HF-model plumbing (device/ckpt); typed _Backbone only after _text_tower
    _active = getattr(getattr(base, "config", None), "_attn_implementation", None)
    if _strict and _active != attn_implementation:  # HF loaded but silently downgraded the module
        raise RuntimeError(
            f"attn_implementation={attn_implementation!r} requested but model is running {_active!r} (silent "
            "downgrade) — verify flash-attn install / bf16 dtype / head_dim support."
        )
    ltt = _top_k_layers(int(getattr(base.config, "num_hidden_layers", 0) or 0))
    if ltt is not None:
        print(f"[langset] LoRA restricted to top-{lora_top_k} layers {ltt}", flush=True)
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        layers_to_transform=ltt,
    )
    # strip the lm_head: we only read hidden states, and computing the full-vocab logits
    # ([B,S,vocab]) every forward OOMs — a 0.6B at bs48/384 hit 78GB purely on Qwen3's 152k-vocab projection.
    peft = get_peft_model(base, lora)
    if train_base:
        # KNOWLEDGE INJECTION: rank-16 LoRA on a FROZEN base can't STORE new facts (only re-style existing ones) —
        # facts live in the base MLP weights. Unfreeze the whole base so next-token [LEARN] can actually rewrite
        # "GrEStG=Grundgesetz" -> the real statute. Full-FT capacity; default off (frozen-LoRA, unchanged).
        for p in peft.parameters():
            p.requires_grad_(True)
    if grad_ckpt:
        # trade compute for activation memory so a LARGE InfoNCE batch (= more in-batch negatives, the dominant lever)
        # fits — a 4B at batch 4 was negative-starved. use_cache off is required; input-require-grads lets grad reach
        # checkpointed segments when only LoRA trains (frozen embeddings).
        base.config.use_cache = False
        peft.gradient_checkpointing_enable()
        peft.enable_input_require_grads()
    return _text_tower(
        cast("_Backbone", peft)
    )  # PeftModel|PeftMixedModel don't structurally match the Protocol


class LangSetModel(nn.Module):
    """LLM backbone (LoRA) + EmitHead. The latent lives in the model's own hidden space; the geometry is defined
    by the `target_text` the Trainer contrasts against (see Trainer)."""

    def __init__(
        self,
        backbone: _Backbone,
        tokenizer: PreTrainedTokenizerBase,
        latent_dim: int,
        n_latents: int,
        llm_model: str,
        dropout: float = 0.0,
        max_len: int = 512,
        multi_latent: bool = False,
        fsq_dim: int = 128,
        fsq_levels: int = 8,
        fsq_emit: bool = False,
        pool_mode: str = "",
        code_emit: bool = False,
        n_codes: int = 0,
        code_tau: float = 0.07,
        res_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.embed = backbone.get_input_embeddings()
        self.h = _cfg_int(backbone.config, "hidden_size")
        self.vocab_size = _cfg_int(backbone.config, "vocab_size")
        # Gemma E-series (3n/4) use Per-Layer Embeddings: each layer mixes in an embedding indexed by TOKEN ID.
        # We pass `per_layer_inputs` explicitly (real tokens -> real PLE, synthetic emit/feedback tokens -> zeros)
        # so `inputs_embeds` forwards don't crash on the reverse-ID lookup. 0 => not a PLE model (no-op).
        self._ple_dim = int(getattr(backbone.config, "hidden_size_per_layer_input", 0) or 0)
        self._n_layers = int(getattr(backbone.config, "num_hidden_layers", 0) or 0)
        # A text tower (when build_backbone unwrapped to one) always returns `last_hidden_state`, so we don't ask the
        # backbone to collect (and keep) every layer's hidden states — a big memory win. A plain ForCausalLM has no
        # `last_hidden_state`, so it still needs output_hidden_states to expose the final layer.
        self._need_ohs = hasattr(
            backbone, "lm_head"
        )  # text tower -> last_hidden_state; raw ForCausalLM -> ohs
        self.latent_dim = latent_dim
        self.n_latents = n_latents
        self.multi_latent = multi_latent
        self.fsq_dim = fsq_dim
        self.fsq_levels = fsq_levels
        eos_id = int(tokenizer.eos_token_id or 0)
        self.head = EmitHead(
            self.h,
            latent_dim,
            n_latents,
            dropout,
            eos_id=eos_id,
            multi_latent=multi_latent,
            fsq_dim=fsq_dim,
            fsq_levels=fsq_levels,
            fsq_emit=fsq_emit,
            code_emit=code_emit,
            n_codes=n_codes,
            code_tau=code_tau,
            res_dim=res_dim,
        )
        self.llm_model = llm_model
        self.max_len = max_len
        self._lora_top_k = 0  # overwritten by from_pretrained; persisted in config
        # pool_mode="last": SKIP the learned emit-query; read the backbone's LAST real-token hidden and project it
        # (head.out_proj). Lets a FROZEN strong-embedder backbone (e.g. F2LLM) be specialized by training ONLY the
        # projection head -> nothing is backpropped through the layers (no grad_ckpt, huge batch). "" => emit-head
        # (default, byte-identical: forward() takes the learned-query path exactly as before).
        self.pool_mode = pool_mode
        self._frozen_bb = (
            False  # set by from_pretrained(freeze_backbone=True); gates the no-grad backbone read
        )
        # RANDOM-INIT bookkeeping (set by from_scratch). A pretrained model rebuilds its backbone from `llm_model`, so
        # persistence stores only LoRA; a random-init model has no such source, so save/load must carry the FULL net.
        self._pretrained = True
        # FULL-FINETUNE bookkeeping (set by from_pretrained(train_base=True)). A pretrained model normally stores only
        # LoRA on save (base rebuilds from `llm_model`), but train_base=True trains the WHOLE base — those weights have
        # no source to rebuild from, so save/snapshot/load must carry the FULL backbone exactly like a random-init net.
        self._full_ft = False
        self._tokenizer_id: Optional[str] = (
            None  # decoupled HF tokenizer id (None => same as llm_model/arch)
        )
        self._arch_overrides: Optional[dict] = (
            None  # config shrink applied to the from-scratch backbone
        )
        # PERSISTED AUXILIARY HEADS (langset.heads): a name->nn.Linear map of PERSISTED heads (transient heads are
        # never registered here — they live and die in the trainer). Each carries a metadata spec (read site, loss,
        # in/out dims, CE classes) so save_pretrained can serialize it and `head_output(name, ...)` can read it back
        # at inference. Empty by default => byte-identical: no extra params, nothing written to disk.
        self.aux_heads: nn.ModuleDict = nn.ModuleDict()
        self.aux_head_specs: dict[str, dict[str, Any]] = {}

    # ---- construction ----
    @classmethod
    def from_pretrained(
        cls,
        llm_model: str,
        *,
        latent_dim: Optional[int] = None,
        n_latents: int = 1,
        lora_r: int = 16,
        dropout: float = 0.0,
        bf16: bool = False,
        max_len: int = 512,
        multi_latent: bool = False,
        fsq_dim: int = 128,
        fsq_levels: int = 8,
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
        train_base: bool = False,
        grad_ckpt: bool = False,
        lora_top_k: int = 0,
        fsq_emit: bool = False,
        pool_mode: str = "",
        freeze_backbone: bool = False,
        code_emit: bool = False,
        n_codes: int = 0,
        code_tau: float = 0.07,
        res_dim: int = 0,
    ) -> "LangSetModel":
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(llm_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        backbone = build_backbone(
            llm_model,
            lora_r,
            dropout,
            bf16,
            dev,
            attn_implementation,
            train_base=train_base,
            grad_ckpt=grad_ckpt,
            lora_top_k=lora_top_k,
        )
        if latent_dim is None:  # default: emit in the backbone's own hidden space
            latent_dim = _cfg_int(backbone.config, "hidden_size")
        model = cls(
            backbone,
            tok,
            latent_dim,
            n_latents,
            llm_model,
            dropout,
            max_len,
            multi_latent,
            fsq_dim,
            fsq_levels,
            fsq_emit=fsq_emit,
            pool_mode=pool_mode,
            code_emit=code_emit,
            n_codes=n_codes,
            code_tau=code_tau,
            res_dim=res_dim,
        )
        model._lora_top_k = int(lora_top_k)  # persisted in config so load() rebuilds same adapters
        model._full_ft = bool(
            train_base
        )  # train_base trains the whole base -> persist FULL backbone
        if pool_mode == "last":  # WARM-START the pool head at the base's NATIVE embedding: identity
            torch.nn.init.eye_(
                model.head.out_proj.weight
            )  # out_proj -> head_project == normalize(last-token hidden) ==
            torch.nn.init.zeros_(
                model.head.out_proj.bias
            )  # the base's own readout (last-token pool) -> starts at base
            #  quality and REFINES, instead of a random head destroying the base geometry and relearning it worse.
        if freeze_backbone:  # FROZEN base: only the head trains -> backbone read needs no graph
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            model._frozen_bb = True
        return model.to(dev)

    @classmethod
    def from_scratch(
        cls,
        arch: str,
        *,
        tokenizer_id: Optional[str] = None,
        latent_dim: Optional[int] = None,
        n_latents: int = 1,
        dropout: float = 0.0,
        bf16: bool = False,
        max_len: int = 512,
        multi_latent: bool = False,
        fsq_dim: int = 128,
        fsq_levels: int = 8,
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
        grad_ckpt: bool = False,
        fsq_emit: bool = False,
        arch_overrides: Optional[dict] = None,
    ) -> "LangSetModel":
        """RANDOM-INIT control arm (the "does pretraining matter" baseline). `arch` names an HF model whose
        ARCHITECTURE is copied, but the weights are NOT loaded — the backbone starts from scratch and trains fully
        (no LoRA). The tokenizer is decoupled: pass any HF `tokenizer_id` (default = `arch`) and the fresh embedding
        table is sized to it, so there is no baked-in tokenizer. `arch_overrides` shrinks the net, e.g.
        `{"num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 4,
        "intermediate_size": 1024}`. Everything downstream (emit head, FSQ, EMA-twin/SIGReg targets, losses) is
        identical to `from_pretrained`; only the source of the backbone weights differs."""
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(tokenizer_id or arch)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        backbone = build_backbone(
            arch,
            0,
            dropout,
            bf16,
            dev,
            attn_implementation,
            grad_ckpt=grad_ckpt,
            pretrained=False,
            arch_overrides=arch_overrides,
            vocab_size=len(tok),
        )
        if latent_dim is None:  # default: emit in the backbone's own hidden space
            latent_dim = _cfg_int(backbone.config, "hidden_size")
        model = cls(
            backbone,
            tok,
            latent_dim,
            n_latents,
            arch,
            dropout,
            max_len,
            multi_latent,
            fsq_dim,
            fsq_levels,
            fsq_emit=fsq_emit,
        )
        model._pretrained = False
        model._tokenizer_id = tokenizer_id
        model._arch_overrides = dict(arch_overrides) if arch_overrides else None
        return model.to(dev)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # ---- forward / inference ----
    def _run_backbone(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        real_ids: Optional[torch.Tensor] = None,
        real_start: int = 0,
        past_key_values: Optional["Cache"] = None,
        use_cache: bool = False,
    ) -> _HiddenOutput:
        """Backbone forward that stays correct on Per-Layer-Embedding models (Gemma E-series). For PLE models we
        build `per_layer_inputs` ourselves: the real-token span [real_start : real_start+len] gets its true
        token-ID lookup; synthetic positions (emit query / fed-back latents / recon soft tokens) get zeros, so
        their per-layer contribution is projection-only and the crashing embed->ID reverse lookup never runs.
        A no-op for non-PLE backbones (identical to a plain inputs_embeds forward).

        `use_cache`/`past_key_values` drive the TRAINING-TIME KV cache used by the multi-latent rollout: the
        prompt is forwarded once, then each latent token is forwarded alone against the cached prefix K/V. The
        cache tensors are NOT detached, so gradients flow back through them to the backbone params exactly as in
        a full-sequence forward (verified: single-token cached hiddens match the full forward to ~1e-5). The
        `attention_mask` passed here covers the FULL length (prefix history + current token) so RoPE positions
        stay correct under left-padding; HF derives cache_position from the past length."""
        kw: dict[str, Any] = {}
        if self._ple_dim:
            b, s = inputs_embeds.shape[:2]
            ple = inputs_embeds.new_zeros(b, s, self._n_layers, self._ple_dim)
            # PLE models (Gemma-E) expose get_per_layer_inputs; bind to a local so hasattr narrows _Backbone
            # to the intersection that has it (ty narrows locals, not member access) — checked, not cast away.
            # Non-PLE backbones never reach here (_ple_dim==0), so they stay plain _Backbone members.
            bb = self.backbone
            if real_ids is not None and hasattr(bb, "get_per_layer_inputs"):
                real = cast("Unknown", bb).get_per_layer_inputs(real_ids, None).to(ple.dtype)
                ple[:, real_start : real_start + real_ids.size(1)] = real
            kw["per_layer_inputs"] = ple
        if use_cache:  # per-call override of the config's use_cache=False (KV-cache rollout only)
            kw["use_cache"] = True
            if past_key_values is not None:
                kw["past_key_values"] = past_key_values
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=self._need_ohs,
            **kw,
        )

    @staticmethod
    def _last_hidden(out: _HiddenOutput) -> torch.Tensor:
        h = getattr(
            out, "last_hidden_state", None
        )  # text tower returns this; a ForCausalLM does not
        return h if h is not None else out.hidden_states[-1]

    def _pool_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last real-token hidden of the raw text (frozen-backbone read when _frozen_bb). [B, h]. This is the STATIC
        feature the frozen-pool fast path caches ONCE; head_project() then turns it into the trainable latent."""
        if self._frozen_bb:  # frozen backbone -> read under no_grad (zero activation memory)
            with torch.no_grad():
                hid = self._last_hidden(
                    self._run_backbone(self.embed(input_ids), attention_mask, input_ids, 0)
                )
        else:
            hid = self._last_hidden(
                self._run_backbone(self.embed(input_ids), attention_mask, input_ids, 0)
            )
        last = attention_mask.sum(1).long().clamp(min=1) - 1  # index of each row's last real token
        return hid[torch.arange(hid.size(0), device=hid.device), last]  # [B, h]

    def head_project(self, feats: torch.Tensor) -> torch.Tensor:
        """Project pooled features through the trainable head -> normalized latent. [B, h] -> [B, d]."""
        return F.normalize(self.head.out_proj(feats.float()), p=2, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Read input text, emit the latent. Returns [B, d]."""
        if self.pool_mode == "last":  # POOL path: no emit query; last real-token hidden -> out_proj
            return self.head_project(self._pool_hidden(input_ids, attention_mask))
        nl = self.head.n_latents
        rev = self.embed(input_ids)
        q = self.head.q.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(rev.dtype)
        emb = torch.cat([rev, q], 1)
        am = torch.cat(
            [
                attention_mask,
                torch.ones(
                    input_ids.size(0), nl, device=input_ids.device, dtype=attention_mask.dtype
                ),
            ],
            1,
        )
        hid = self._last_hidden(
            self._run_backbone(emb, am, input_ids, 0)
        )  # real tokens front, query appended
        z = self.head(hid[:, -nl:, :])  # [B, n_latents, d]
        return z.squeeze(1) if nl == 1 else z

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[str, list[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        device: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Sentence-Transformer-compatible. This is the method SetFit calls on its body."""
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        was_training = self.training
        self.eval()
        out: list[torch.Tensor] = []
        with (
            torch.no_grad()
        ):  # eval-only: LoRA params require grad, so w/o this every forward builds
            for i in range(
                0, len(texts), batch_size
            ):  # a throwaway autograd graph -> slower + huge VRAM (caps batch)
                enc = self.tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(self.device)
                z = self(enc["input_ids"], enc["attention_mask"])
                if normalize_embeddings:
                    z = F.normalize(z, p=2, dim=-1)
                out.append(z.float().cpu())  # fp32 so .numpy() works even with a bf16 backbone
        if was_training:
            self.train()
        emb = torch.cat(out)
        emb = emb[0] if single else emb
        return emb.numpy() if convert_to_numpy else emb

    def emit(self, sentences: Union[str, list[str]], **kw: Unknown) -> torch.Tensor:
        # kw is a genuine passthrough to encode() -> Unknown (gradual) so it forwards into encode's typed
        # params without ANN401 tripping on Any. cast the union return (encode -> ndarray | Tensor).
        return cast("torch.Tensor", self.encode(sentences, convert_to_numpy=False, **kw))

    # --- auxiliary supervised heads (langset.heads) --------------------------------------------------------------
    def seed_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-sequence backbone hidden for a LEFT-padded batch -> [B, h]. Left padding puts every row's last real
        token at the final column, so the pooled/final hidden is simply `hid[:, -1]` (this is the "hidden" read
        site a value/time head reads; distinct from `_pool_hidden`, which assumes RIGHT padding). Grad flows into
        the backbone/LoRA when they are trainable, so the head can SHAPE the representation."""
        hid = self._last_hidden(
            self._run_backbone(self.embed(input_ids), attention_mask, input_ids, 0)
        )
        return hid[:, -1]

    def add_aux_head(self, module: nn.Linear, spec: dict[str, Any]) -> None:
        """Register a PERSISTED auxiliary head so `save_pretrained` serializes it and `head_output` can read it back.
        `spec` is the head's metadata (name / reads / loss / in_dim / out_dim / classes). Called by the trainer for
        each persisted Head; the module object is SHARED with the trainer so in-place training keeps it current."""
        name = str(spec["name"])
        self.aux_heads[name] = module
        self.aux_head_specs[name] = dict(spec)

    @torch.no_grad()
    def head_output(
        self,
        name: str,
        sentences: Union[str, list[str]],
        batch_size: int = 32,
        reduce: str = "mean",
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Query a PERSISTED auxiliary head at inference — the readout that makes a value/time head useful.
        `reads="hidden"` heads read the pooled seed hidden -> one [out_dim] vector per sentence ([N, out_dim]).
        `reads="recon"` heads read every emitted latent of the rollout: reduce="mean" -> [N, out_dim] (mean over a
        row's emitted latents); reduce="none" -> a per-row list of [Li, out_dim] (the dense per-tick readout).
        For a `loss="mse"` value head out_dim=1 (a scalar per state); for a `loss="ce"` head the columns are class
        logits (argmax -> `aux_head_specs[name]['classes']`)."""
        if name not in self.aux_heads:
            raise KeyError(f"no persisted head {name!r}; have {sorted(self.aux_heads)}")
        if reduce not in (
            "mean",
            "none",
        ):  # fail loud: a typo'd reduce must not silently pick the mean path
            raise ValueError(f"head_output reduce must be 'mean' or 'none'; got {reduce!r}")
        module = cast(nn.Linear, self.aux_heads[name])
        reads = self.aux_head_specs[name]["reads"]
        if reads not in (
            "hidden",
            "recon",
        ):  # guard the read site rather than fall through to recon
            raise ValueError(f"head {name!r} has an unexpected read site {reads!r}")
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        was_training = self.training
        self.eval()
        if reads == "hidden":
            rows: list[torch.Tensor] = []
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    padding_side="left",  # matches training: last real token at the final column
                    return_tensors="pt",
                ).to(self.device)
                pooled = self.seed_hidden(enc["input_ids"], enc["attention_mask"])
                rows.append(module(pooled.float()).cpu())
            if was_training:
                self.train()
            out = torch.cat(rows)
            return out[0] if single else out
        # reads == "recon": roll out, apply the head to each emitted latent, reduce over the row's latents.
        lat, lengths = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            self.rollout(texts, return_lengths=True),
        )
        per_row = module(lat.float())  # [B, L, out_dim]
        if was_training:
            self.train()
        if reduce == "none":
            seq = [per_row[r, : int(lengths[r])].cpu() for r in range(len(texts))]
            return seq[0] if single else seq
        means: list[torch.Tensor] = []
        for r in range(len(texts)):
            n = max(int(lengths[r]), 1)
            means.append(per_row[r, :n].mean(0).cpu())
        pooled_out = torch.stack(means)
        return pooled_out[0] if single else pooled_out

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new: int = 200) -> str:
        """Greedy text generation via the TIED input embedding (the lm_head is stripped). Used to MEASURE whether
        [LEARN]/train_base actually injected knowledge — ask the trained model a question and read its answer."""
        self.eval()
        tok, dev = self.tokenizer, self.device
        msgs = [{"role": "user", "content": prompt}]
        try:
            enc = tok.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        ids = cast("Unknown", enc)["input_ids"].to(
            dev
        )  # tokenizer BatchEncoding: apply_chat_template union stub edge
        eos = int(tok.eos_token_id or 0)
        out: list[int] = []
        for _ in range(max_new):
            hid = self._last_hidden(
                self._run_backbone(self.embed(ids), torch.ones_like(ids), ids, 0)
            )[:, -1]
            nxt = int(F.linear(hid.float(), self.embed.weight.float()).argmax(-1))
            if nxt == eos:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
        return tok.decode(out, skip_special_tokens=True).strip()

    # ---- multi-latent autoregressive rollout (drives the feedback / stop_logit seams) ----
    @torch.no_grad()
    def rollout(
        self,
        text: Union[str, list[str]],
        max_steps: int = 8,
        stop_threshold: float = 0.0,
        return_lengths: bool = False,
        return_confidence: bool = False,
        temperature: float = 0.0,
        return_soft: bool = False,
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    ]:
        """Autoregressive emission: real text tokens and emitted latents share ONE hidden stream. Each emitted
        latent is fed back into the sequence via `head.feedback` (the latent lives in the backbone's own hidden
        space, so an emitted latent and a real token embedding are the same kind of vector — they intermingle).
        Terminates per-row on the natural-EOS `stop_logit`. Returns [B, L, d] (or [L, d] for a single string),
        zero-padding halted rows; pass `return_lengths` for the per-row emit count. `return_confidence` (FSQ path
        only) additionally returns a dict of per-step NATIVE emission-confidence [B, L] tensors — the model's OWN
        co-trained uncertainty: 'code' = mean per-dim FSQ digit-softmax certainty, 'stop' = P(STOP) at that step,
        'dig0' = digit-0 level certainty. Returned as (lat, lengths, conf) when set. `temperature` (FSQ path only):
        0 = deterministic argmax (default); >0 = SAMPLE each FSQ digit from softmax(logits/temperature), so K rolls
        of one seed fan into a DISTRIBUTION of plausible futures (McDonald's multi-rollout). `return_soft` returns
        (lat, lengths, soft_lat, ent) where soft_lat is the EXPECTED latent E[digit]->up_proj (the calibrated
        superposition, read from the full digit softmax rather than the argmax) and ent is the mean per-dim softmax
        entropy (the model's native emission fuzziness). On the single-latent (out_proj) path soft_lat==lat and ent==0."""
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        dev = self.device
        # token-native (multi-latent) FSQ path (codebook). Single-latent falls to the out_proj else-branch below.
        codebook = self.head.multi_latent
        stop_idx = (
            (self.head.n_codes if self.head.code_emit else self.head.fsq_levels) if codebook else -1
        )  # STOP is the extra class in dim-0's softmax (codebook mode: the class past the last code)
        pad_side = (
            "left" if codebook else self.tokenizer.padding_side
        )  # left-pad so [:, -1] is the last real token
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            padding_side=pad_side,
            return_tensors="pt",
        ).to(dev)
        seq = self.embed(enc["input_ids"])  # [B, S, h] — real TEXT tokens
        am = enc["attention_mask"]
        b = seq.size(0)
        alive = torch.ones(b, dtype=torch.bool, device=dev)
        lengths = torch.zeros(b, dtype=torch.long, device=dev)
        cols: list[torch.Tensor] = []
        conf_code: list[torch.Tensor] = []  # native per-step confidences (FSQ path)
        conf_stop: list[torch.Tensor] = []
        conf_dig0: list[torch.Tensor] = []
        soft_cols: list[torch.Tensor] = []  # SOFT emission: E[latent] under digit softmax
        ent_cols: list[torch.Tensor] = []  # native uncertainty: mean per-dim softmax entropy
        was_training = self.training
        self.eval()
        for _step in range(max_steps):
            if codebook:  # AR: read the LAST real token's hidden
                hid = self._last_hidden(self._run_backbone(seq, am, enc["input_ids"], 0))[
                    :, -1
                ]  # PLE-safe AR read
                dim_lg, stop_lg = self.head.emit_logits(hid)  # [B, fsq_dim, L], [B, 1]
                d0_full = torch.cat(
                    [dim_lg[:, 0, :], stop_lg], -1
                )  # [B, L+1] over [L levels; STOP]
                # CODEBOOK: termination is FACTORIZED out of the emission softmax. Folding STOP in (as the FSQ
                # path does) is fair when dim-0's own target is one-hot, but a set-valued target is 1/k-diffuse
                # and sharing one normalizer couples them: the gradient suppressing STOP at a member position
                # scales with P(STOP), which weakens as the set widens. Independent sigmoid: "continue or stop"
                # is decided on its own logit and never rides on how wide the set is.
                code_stop = self.head.code_emit
                pick = (
                    dim_lg[:, 0, :] if code_stop else d0_full
                )  # codebook: members only, STOP is separate
                if temperature and temperature > 0:  # SAMPLE each digit (stochastic futures)
                    dim0 = torch.multinomial(
                        torch.softmax(pick.float() / temperature, -1), 1
                    ).squeeze(-1)
                    rest = (
                        dim0.new_zeros(b, 0)  # codebook: fsq_dim == 1, so there are no other dims
                        if code_stop
                        else torch.multinomial(
                            torch.softmax(dim_lg[:, 1:, :].float() / temperature, -1).reshape(
                                -1, self.head.fsq_levels
                            ),
                            1,
                        ).reshape(b, -1)
                    )
                else:  # deterministic argmax (default)
                    dim0 = pick.argmax(-1)
                    rest = dim_lg[:, 1:, :].argmax(-1)  # [B, 0] under codebook emission
                emit_now = (
                    alive & (torch.sigmoid(stop_lg.squeeze(-1)) < 0.5)  # independent terminator
                    if code_stop
                    else alive & (dim0 != stop_idx)  # halt on STOP (or already-dead rows)
                )
                if return_confidence:  # the model's OWN emission uncertainty
                    p0 = torch.softmax(pick.float(), -1)  # [B, L+1] (codebook: [B, n_codes])
                    conf_stop.append(  # P(STOP): its own sigmoid under codebook, else the folded class
                        torch.sigmoid(stop_lg.squeeze(-1)) if code_stop else p0[:, -1]
                    )
                    conf_dig0.append((p0 if code_stop else p0[:, :-1]).max(-1).values)
                    conf_code.append(
                        torch.softmax(dim_lg.float(), -1).max(-1).values.mean(-1)
                    )  # mean code certainty
                if return_soft:  # the SUPERPOSITION reads out HERE, not from argmax
                    p = torch.softmax(
                        dim_lg.float(), -1
                    )  # [B, fsq_dim, L] per-dim digit distribution
                    if (
                        self.head.code_emit
                    ):  # ONE distribution over the alphabet: mixture + its entropy
                        pc = p.squeeze(-2)  # [B, n_codes]
                        soft_z = F.normalize(pc @ self.head.code, dim=-1)  # mixture of codes [B, d]
                        ent = -(pc.clamp_min(1e-9).log() * pc).sum(
                            -1
                        )  # [B] entropy = how many codes it is holding
                    else:
                        levels = torch.arange(
                            self.head.fsq_levels, device=dim_lg.device, dtype=torch.float32
                        )
                        soft_z = self.head.reconstruct(
                            (p * levels).sum(-1)
                        )  # E[digit] -> up_proj = expected latent [B, d]
                        ent = (-(p.clamp_min(1e-9).log() * p).sum(-1)).mean(
                            -1
                        )  # [B] mean per-dim entropy (native fuzziness)
                    soft_cols.append(
                        torch.where(emit_now.unsqueeze(1), soft_z, torch.zeros_like(soft_z))
                    )
                    ent_cols.append(torch.where(emit_now, ent, torch.zeros_like(ent)))
                digits = torch.cat(
                    [dim0.clamp(max=stop_idx - 1).unsqueeze(-1), rest], -1
                )  # [B, fsq_dim] (codebook: [B, 1], the top code index)
                z = (  # the emitted latent: FSQ commits to the argmax code, codebook to the mixture
                    self.head.commit(dim_lg, hid=hid)
                    if self.head.code_emit
                    else self.head.reconstruct(digits)
                )  # [B, d]
                cols.append(torch.where(emit_now.unsqueeze(1), z, torch.zeros_like(z)))
                lengths = lengths + emit_now.long()
                seq = torch.cat([seq, self.head.feedback(z).unsqueeze(1).to(seq.dtype)], 1)
                am = torch.cat([am, emit_now.long().unsqueeze(1)], 1)
                alive = emit_now
            else:
                q = (
                    self.head.q[:1].unsqueeze(0).expand(b, 1, -1).to(seq.dtype)
                )  # one read-query token
                am_q = torch.cat([am, am.new_ones(b, 1)], 1)
                hid = self._last_hidden(
                    self._run_backbone(torch.cat([seq, q], 1), am_q, enc["input_ids"], 0)
                )[:, -1]  # [B, h] PLE-safe
                z = self.head(hid.unsqueeze(1)).squeeze(1)  # [B, d] emit via out_proj
                stop = (
                    self.head.stop_logit(hid, self.embed) > stop_threshold
                )  # single-latent: emit-then-check on the
                emit_now = alive  # natural-EOS logit of the hidden that emitted
                cols.append(torch.where(emit_now.unsqueeze(1), z, torch.zeros_like(z)))
                if return_soft:  # single-latent out_proj emission IS the soft
                    soft_cols.append(
                        torch.where(emit_now.unsqueeze(1), z, torch.zeros_like(z))
                    )  # readout; entropy undefined
                    ent_cols.append(torch.zeros(b, device=dev))  # -> 0
                lengths = lengths + emit_now.long()
                seq = torch.cat(
                    [seq, self.head.feedback(z).unsqueeze(1).to(seq.dtype)], 1
                )  # feed back the emitted latent
                am = torch.cat(
                    [am, emit_now.long().unsqueeze(1)], 1
                )  # mask stopped/dead positions out of attention
                alive = alive & ~stop
            if not bool(alive.any()):
                break
        if was_training:
            self.train()
        lat = torch.stack(cols, 1) if cols else seq.new_zeros(b, 0, self.latent_dim)  # [B, L, d]
        if return_confidence:  # per-step native confidences [B, L]

            def _st(xs: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(xs, 1) if xs else lat.new_zeros(b, 0)

            conf = {"code": _st(conf_code), "stop": _st(conf_stop), "dig0": _st(conf_dig0)}
            if single:
                conf = {k: v[0, : int(lengths[0])] for k, v in conf.items()}
        if single:
            lat = lat[0, : int(lengths[0])]
        if return_soft:  # SOFT readout: expected latent + native entropy
            soft_lat = (
                torch.stack(soft_cols, 1) if soft_cols else lat.new_zeros(b, 0, self.latent_dim)
            )
            ent_t = torch.stack(ent_cols, 1) if ent_cols else lat.new_zeros(b, 0)
            if single:
                soft_lat, ent_t = soft_lat[0, : int(lengths[0])], ent_t[0, : int(lengths[0])]
            return lat, lengths, soft_lat, ent_t
        if return_confidence:
            return lat, lengths, conf
        if return_lengths:
            return lat, lengths
        return lat

    def rollout_train(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced AR pass for training: feed the TRUE latents back through `feedback` and predict the next
        emission at each future position in ONE forward pass (no python loop). Returns (preds [B, L, d],
        stop_logits [B, L]) — the per-step emission and its natural-EOS logit. (Continuous, single-latent path.)"""
        bsz, s_len = input_ids.size(0), input_ids.size(1)
        n = target_latents.size(1)
        seed = self.embed(input_ids)  # [B, S, h]
        fb = self.head.feedback(target_latents.to(seed.dtype))  # [B, L, h] — true latents fed back
        seq = torch.cat([seed, fb], 1)
        am = torch.cat([attention_mask, attention_mask.new_ones(bsz, n)], 1)
        hid = self._last_hidden(
            self._run_backbone(seq, am, input_ids, 0)
        )  # PLE-safe teacher-forced read
        hf = hid[:, s_len - 1 : s_len - 1 + n]  # [B, L, h] — predicts future positions
        preds = self.head(hf)  # [B, L, d]
        stop = self.head.stop_logit(hf, self.embed)  # [B, L]
        return preds, stop

    def rollout_train_codebook(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_latents: torch.Tensor,
        tau: float = 0.07,
        train_hops: Optional[int] = None,
        ss_prob: float = 0.0,
        ss_sample: bool = False,
        ss_mask: Optional[torch.Tensor] = None,
        kv_cache: bool = False,
        return_emit_hidden: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Token-native AR pass (multi-latent, FSQ). Quantizes the TRUE target latents to per-dim digits and predicts
        the digits at each of L+1 positions (targets 0..L-1 then a STOP at position L). Returns (dim_logits
        [B, L+1, fsq_dim, fsq_levels], stop_logits [B, L+1, 1], digits [B, L, fsq_dim], recon [B, L, d]) — the STOP
        label is appended by the trainer.

        `ss_prob`=0 (default): pure TEACHER FORCING in ONE forward pass — every position predicted from the true
        prefix, gradients flow at most ONE hop (byte-identical to before). `ss_prob`>0: SCHEDULED SAMPLING for the
        first `train_hops` positions (None = all) — with prob `ss_prob` each of those positions is fed the model's
        OWN emitted latent instead of the ground truth, so the emitter learns to consume its own (imperfect)
        predictions. This is the exposure-bias fix that makes MULTI-HOP rollout trained rather than emergent. Self-
        fed latents are DETACHED (standard scheduled sampling; no BPTT through FSQ argmax). Cost = train_hops+1
        backbone passes (positions past train_hops are teacher-forced in one pass). `ss_sample` samples the self-fed
        digits instead of argmax.

        `ss_mask` (optional [B, H] bool): the PRECOMPUTED per-(row, hop) self-feed decisions. When given, the loop
        uses `ss_mask[:, h]` in place of a fresh `torch.rand < ss_prob` draw. This makes the rollout DETERMINISTIC
        given the mask, so GradCache's phase-1 (no_grad, full batch) and phase-2 (grad, per chunk) forwards produce
        identical `recon` and the cached gradients line up exactly. None = sample as usual (byte-identical)."""
        assert self.head.multi_latent
        bsz, s_len = input_ids.size(0), input_ids.size(1)
        n = target_latents.size(1)
        digits, recon = self.head.encode(target_latents.reshape(-1, target_latents.size(-1)))
        digits = digits.view(bsz, n, -1)  # [B, L, fsq_dim]
        recon = recon.view(bsz, n, -1)  # [B, L, d] — clean feedback + recon target
        H = n if train_hops is None else max(0, min(int(train_hops), n))
        if ss_prob <= 0 or n == 0 or H == 0:  # TEACHER-FORCED one-shot (default, fast)
            seed = self.embed(input_ids)
            fb = self.head.feedback(
                recon.detach().to(seed.dtype)
            )  # [B, L, h] — feedback (no grad through fb)
            seq = torch.cat([seed, fb], 1)
            am = torch.cat([attention_mask, attention_mask.new_ones(bsz, n)], 1)
            hid = self._last_hidden(
                self._run_backbone(seq, am, input_ids, 0)
            )  # PLE-safe teacher-forced read
            hf = hid[:, s_len - 1 : s_len - 1 + n + 1]  # [B, L+1, h] — +1 to predict the STOP
            dim_lg, stop_lg = self.head.emit_logits(hf)  # [B, L+1, fsq_dim, L], [B, L+1, 1]
            return (
                (dim_lg, stop_lg, digits, recon, hf)
                if return_emit_hidden
                else (dim_lg, stop_lg, digits, recon)
            )
        # SCHEDULED-SAMPLING multi-hop path
        dev = recon.device
        if (
            kv_cache
        ):  # KV-CACHE rollout: forward the prompt ONCE, then each latent token ALONE against the
            # cached prefix K/V. Numerically identical to the recompute loop below given the same ss decisions
            # (cached single-token hiddens match a full forward to ~1e-5), but activation memory is ~1 prompt
            # forward + n single tokens instead of n full-prefix forwards — it kills the O(ticks) blowup that
            # forces grad_ckpt, so this path trains WITHOUT checkpointing. PLE (Gemma-E) unsupported: its
            # per_layer_inputs would need the cached span; the maze/SmolLM backbones are non-PLE.
            assert not self._ple_dim, "kv_cache rollout does not support PLE (Gemma-E) backbones"
            # HF force-disables use_cache under gradient checkpointing, which would silently null the cache and
            # feed each latent token with NO history. kv_cache REPLACES grad_ckpt (it removes the O(ticks) blowup
            # that grad_ckpt was paying for), so the two are mutually exclusive — fail loudly, don't degrade.
            assert not getattr(self.backbone, "is_gradient_checkpointing", False), (
                "kv_cache rollout is incompatible with gradient checkpointing (HF disables the cache under it); "
                "kv_cache replaces grad_ckpt — turn grad_ckpt OFF"
            )

            def _feed(
                dl_h: torch.Tensor, t: int
            ) -> torch.Tensor:  # the latent to advance tick t -> t+1
                recon_pred = self.head.commit(dl_h, ss_sample, hid=hid).detach()
                if t < H:  # self-feed region: own emission or ground truth by the ss decision
                    if (
                        ss_mask is not None
                    ):  # shared per-(row,hop) replay (deterministic; matches non-cached)
                        use_own = ss_mask[:, t].to(device=dev, dtype=torch.bool).unsqueeze(1)
                    else:
                        use_own = (torch.rand(bsz, device=dev) < ss_prob).unsqueeze(1)
                    return torch.where(use_own, recon_pred, recon[:, t].detach())
                return recon[:, t].detach()  # teacher-forced region: always the true latent

            seq0 = self.embed(input_ids)
            out = self._run_backbone(seq0, attention_mask, input_ids, 0, use_cache=True)
            pkv = getattr(out, "past_key_values", None)
            hid = self._last_hidden(out)[:, -1]  # last real prompt token -> emits tick 0
            cur_am = attention_mask
            dim_parts = []
            stop_parts = []
            hid_parts = []
            for t in range(H):  # emit tick t, then feed one latent to advance to tick t+1
                dl, sl = self.head.emit_logits(hid)
                dim_parts.append(dl.unsqueeze(1))
                stop_parts.append(sl.unsqueeze(1))
                hid_parts.append(hid.unsqueeze(1))
                fb = self.head.feedback(_feed(dl, t).to(seq0.dtype)).unsqueeze(1)  # [B, 1, h]
                cur_am = torch.cat([cur_am, cur_am.new_ones(bsz, 1)], 1)
                out = self._run_backbone(fb, cur_am, None, 0, past_key_values=pkv, use_cache=True)
                pkv = getattr(out, "past_key_values", None)
                hid = self._last_hidden(out)[:, -1]
            if H < n:
                # The scheduled-sampling prefix must stay serial: each self-fed token depends on the preceding
                # emission.  Once that prefix ends, however, the remaining *known* feedback tokens are ordinary
                # teacher forcing.  Feed the whole tail against the cached prefix in one call, rather than making
                # n-H needless one-token cache calls. `hid` predicts tick H; the tail outputs predict H+1..n.
                fb_tail = self.head.feedback(recon[:, H:].detach().to(seq0.dtype))
                tail_am = torch.cat([cur_am, cur_am.new_ones(bsz, n - H)], 1)
                tail_out = self._run_backbone(
                    fb_tail, tail_am, None, 0, past_key_values=pkv, use_cache=True
                )
                tail_hid = self._last_hidden(tail_out)
                hid_tail = torch.cat([hid.unsqueeze(1), tail_hid], 1)
                dl_tail, sl_tail = self.head.emit_logits(hid_tail)
                dim_parts.append(dl_tail)
                stop_parts.append(sl_tail)
                hid_parts.append(hid_tail)
            else:
                dl, sl = self.head.emit_logits(hid)  # tick n = the STOP position
                dim_parts.append(dl.unsqueeze(1))
                stop_parts.append(sl.unsqueeze(1))
                hid_parts.append(hid.unsqueeze(1))
            out4 = (torch.cat(dim_parts, 1), torch.cat(stop_parts, 1), digits, recon)
            return (*out4, torch.cat(hid_parts, 1)) if return_emit_hidden else out4
        seq = self.embed(input_ids)
        am = attention_mask
        dim_parts: list[torch.Tensor] = []
        stop_parts: list[torch.Tensor] = []
        hid_parts: list[torch.Tensor] = []
        for h in range(H):  # AR self-feed region
            hid = self._last_hidden(self._run_backbone(seq, am, input_ids, 0))[:, -1]
            dl, sl = self.head.emit_logits(hid)  # [B, fsq_dim, L], [B, 1]
            dim_parts.append(dl.unsqueeze(1))
            stop_parts.append(sl.unsqueeze(1))
            hid_parts.append(hid.unsqueeze(1))
            recon_pred = self.head.commit(
                dl, ss_sample, hid=hid
            ).detach()  # own emitted latent (detached)
            if (
                ss_mask is not None
            ):  # GradCache: replay the SHARED per-(row,hop) decisions (deterministic rollout)
                use_own = (
                    ss_mask[:, h].to(device=dev, dtype=torch.bool).unsqueeze(1)
                )  # normalize: caller may pass CPU/int
            else:
                use_own = (torch.rand(bsz, device=dev) < ss_prob).unsqueeze(1)
            feed_h = torch.where(use_own, recon_pred, recon[:, h].detach())
            seq = torch.cat([seq, self.head.feedback(feed_h.to(seq.dtype)).unsqueeze(1)], 1)
            am = torch.cat([am, am.new_ones(bsz, 1)], 1)
        if H < n:  # teacher-force positions H..n-1 in one pass
            fb_rest = self.head.feedback(recon[:, H:].detach().to(seq.dtype))
            seq = torch.cat([seq, fb_rest], 1)
            am = torch.cat([am, am.new_ones(bsz, n - H)], 1)
        hid_all = self._last_hidden(self._run_backbone(seq, am, input_ids, 0))
        hf_rest = hid_all[:, s_len - 1 + H : s_len - 1 + n + 1]  # positions H..n (incl STOP at n)
        dl_rest, sl_rest = self.head.emit_logits(hf_rest)  # [B, n-H+1, fsq_dim, L], [B, n-H+1, 1]
        dim_lg = (
            torch.cat(dim_parts + [dl_rest], 1) if dim_parts else dl_rest
        )  # [B, n+1, fsq_dim, L]
        stop_lg = torch.cat(stop_parts + [sl_rest], 1) if stop_parts else sl_rest  # [B, n+1, 1]
        if return_emit_hidden:
            emit_hid = torch.cat(hid_parts + [hf_rest], 1) if hid_parts else hf_rest
            return dim_lg, stop_lg, digits, recon, emit_hid
        return dim_lg, stop_lg, digits, recon

    def get_sentence_embedding_dimension(self) -> int:
        return self.latent_dim

    def as_sentence_transformer(self) -> SentenceTransformer:
        """Wrap as a `sentence_transformers.SentenceTransformer` so it drops into SetFit as `model_body`."""
        from langset.st_module import to_sentence_transformer

        return to_sentence_transformer(self)

    # ---- persistence (LoRA + head + config; backbone rebuilt from ids) ----
    def save_pretrained(self, path: Union[str, Path]) -> None:
        import json

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if (
            self._pretrained and not self._full_ft
        ):  # pretrained + frozen base: backbone rebuilds from `llm_model` -> LoRA only
            weights = {
                "head": self.head.state_dict(),
                "lora": {k: v.cpu() for k, v in self.backbone.state_dict().items() if "lora" in k},
            }
        else:  # random-init OR train_base full-FT: no source to rebuild from -> FULL backbone
            weights = {
                "head": self.head.state_dict(),
                "backbone": {k: v.cpu() for k, v in self.backbone.state_dict().items()},
            }
        if (
            self.aux_heads
        ):  # PERSISTED auxiliary heads (langset.heads): weights here, metadata in config.json
            weights["aux_heads"] = {
                name: {k: v.cpu() for k, v in mod.state_dict().items()}
                for name, mod in self.aux_heads.items()
            }
        if hasattr(
            self, "emission_bridge"
        ):  # parallel-query emission family: persist its module's state_dict
            weights["emission_bridge"] = {
                k: v.cpu() for k, v in self.emission_bridge.state_dict().items()
            }
        torch.save(weights, p / "langset.pt")
        (p / "config.json").write_text(
            json.dumps(
                {
                    "llm_model": self.llm_model,
                    "latent_dim": self.latent_dim,
                    "n_latents": self.head.n_latents,
                    "max_len": self.max_len,
                    "multi_latent": self.multi_latent,
                    "fsq_dim": self.fsq_dim,
                    "fsq_levels": self.fsq_levels,
                    "lora_top_k": self._lora_top_k,
                    "fsq_emit": self.head.fsq_emit,
                    "code_emit": self.head.code_emit,
                    "n_codes": self.head.n_codes,
                    "code_tau": self.head.code_tau,
                    "res_dim": self.head.res_dim,
                    "concept_spans": getattr(self.head, "concept_spans", None),
                    "concept_names": getattr(self.head, "concept_names", None),
                    "pool_mode": self.pool_mode,
                    "pretrained": self._pretrained,
                    "full_ft": self._full_ft,
                    "tokenizer_id": self._tokenizer_id,
                    "arch_overrides": self._arch_overrides,
                    "aux_heads": self.aux_head_specs,  # {} unless persisted heads were registered (back-compat)
                }
            )
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        lora_r: int = 16,
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
    ) -> "LangSetModel":
        import json

        p = Path(path)
        cfg = json.loads((p / "config.json").read_text())
        if cfg.get("pretrained", True):
            m = cls.from_pretrained(
                cfg["llm_model"],
                latent_dim=cfg["latent_dim"],
                n_latents=cfg.get("n_latents", 1),
                lora_r=lora_r,
                max_len=cfg["max_len"],
                multi_latent=cfg.get("multi_latent", False),
                fsq_dim=cfg.get("fsq_dim", 128),
                fsq_levels=cfg.get("fsq_levels", 8),
                lora_top_k=int(cfg.get("lora_top_k", 0)),
                fsq_emit=cfg.get("fsq_emit", False),
                code_emit=cfg.get("code_emit", False),
                n_codes=int(cfg.get("n_codes", 0)),
                code_tau=float(cfg.get("code_tau", 0.07)),
                res_dim=int(cfg.get("res_dim", 0)),
                pool_mode=cfg.get("pool_mode", ""),
                device=device,
                attn_implementation=attn_implementation,
            )  # 'eager' to read attention weights
        else:  # random-init: rebuild the same arch from scratch, then load full weights
            m = cls.from_scratch(
                cfg["llm_model"],
                tokenizer_id=cfg.get("tokenizer_id"),
                latent_dim=cfg["latent_dim"],
                n_latents=cfg.get("n_latents", 1),
                max_len=cfg["max_len"],
                multi_latent=cfg.get("multi_latent", False),
                fsq_dim=cfg.get("fsq_dim", 128),
                fsq_levels=cfg.get("fsq_levels", 8),
                fsq_emit=cfg.get("fsq_emit", False),
                arch_overrides=cfg.get("arch_overrides"),
                device=device,
                attn_implementation=attn_implementation,
            )
        m._full_ft = bool(
            cfg.get("full_ft", False)
        )  # restore the flag so a re-save round-trips the same way
        if cfg.get("concept_spans"):
            m.head.concept_spans = [tuple(x) for x in cfg["concept_spans"]]
            m.head.concept_names = list(cfg.get("concept_names") or [])
        sd = torch.load(p / "langset.pt", map_location=m.device, weights_only=False)
        # select by the ACTUAL payload (not the flag): a full-FT pretrained model persists "backbone", not "lora".
        m.backbone.load_state_dict(sd["backbone"] if "backbone" in sd else sd["lora"], strict=False)
        m.head.load_state_dict(sd["head"])
        for name, spec in cfg.get(
            "aux_heads", {}
        ).items():  # rebuild + reload PERSISTED auxiliary heads
            mod = nn.Linear(int(spec["in_dim"]), int(spec["out_dim"])).to(m.device)
            mod.load_state_dict({k: v.to(m.device) for k, v in sd["aux_heads"][name].items()})
            m.add_aux_head(mod, spec)
        if (
            "emission_bridge" in sd
        ):  # stash raw state; a QueryBridgeEmission reloads it on re-attach (no import cycle)
            m._emission_bridge_state = sd["emission_bridge"]
        m.eval()
        return m
