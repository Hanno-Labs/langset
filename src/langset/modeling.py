"""Model and emission-head implementations for LangSet.

:class:`LangSetModel` combines a language-model backbone with a learned emission interface. The emission head can
project backbone hidden states into a single embedding, multiple continuous latents, or autoregressive codebook
states. Optional LoRA adaptation supports parameter-efficient fine-tuning.

The model provides Sentence-Transformer-compatible ``encode``, ``get_sentence_embedding_dimension``, and
``as_sentence_transformer`` methods. Codebook-configured multi-latent models additionally support autoregressive
state emission through ``rollout``.
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
    """
    Wraps the backbone model, providing a common interface for accessing hidden states.
    """

    config: PretrainedConfig

    def __call__(self, **kwargs: object) -> _HiddenOutput: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def parameters(self, recurse: bool = ...) -> Iterator[nn.Parameter]: ...
    def state_dict(self, *args: object, **kwargs: object) -> dict[str, Any]: ...
    def load_state_dict(self, *args: object, **kwargs: object) -> object: ...


class EmitHead(nn.Module):
    """Read learned query-token hidden states as continuous or named-state vectors.

    The ordinary embedding path projects hidden states directly into one continuous vector. When
    ``code_emit=True``, the head instead emits a distribution over a fixed, named codebook and feeds the
    resulting superposition back into the backbone for autoregressive state rollout. QueryBridge supplies the
    separate open-ended continuous multi-vector path and does not use this codebook head.
    """

    def __init__(
        self,
        h: int,
        d: int,
        n_latents: int = 1,
        dropout: float = 0.0,
        eos_id: int = 0,
        multi_latent: bool = False,
        code_emit: bool = False,
        n_codes: int = 0,
        code_tau: float = 0.07,
        res_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.q = nn.Parameter(torch.randn(n_latents, h) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(h, d)
        self.in_proj = nn.Linear(d, h)
        self.eos_id = eos_id
        self.multi_latent = multi_latent
        self.code_emit = code_emit
        self.n_codes, self.code_tau = n_codes, code_tau
        self.res_dim = int(res_dim)
        self.state_dim = d - self.res_dim
        self.stop_proj: Optional[nn.Linear] = nn.Linear(h, 1) if code_emit else None
        if code_emit:
            assert n_codes > 0, "code_emit requires n_codes > 0"
            assert 0 <= self.res_dim < d, (
                f"res_dim must satisfy 0 <= res_dim < latent_dim ({d}); got {res_dim}"
            )
            self.register_buffer("code", torch.zeros(n_codes, self.state_dim))
            self.concept_spans: list[tuple[int, int, int, int]] = [(0, n_codes, 0, self.state_dim)]
            self.concept_names: list[str] = []
            self.query_proj: Optional[nn.Linear] = nn.Linear(h, self.state_dim)
            self.res_proj: Optional[nn.Linear] = (
                nn.Linear(h, self.res_dim) if self.res_dim else None
            )
        else:
            self.query_proj = None
            self.res_proj = None

    def forward(self, hid_emit: torch.Tensor) -> torch.Tensor:
        if self.code_emit:
            logits, _ = self.emit_logits(hid_emit)
            state = F.normalize(self.concept_probs(logits) @ self.code, p=2, dim=-1)
            if self.res_dim == 0:
                return state
            return torch.cat([state, self.residual(hid_emit)], -1) * (0.5**0.5)
        return F.normalize(self.out_proj(self.drop(hid_emit.float())), p=2, dim=-1)

    def feedback(self, latent: torch.Tensor) -> torch.Tensor:
        return self.in_proj(latent.float()).to(latent.dtype)

    def set_code(self, code: torch.Tensor) -> None:
        """Install a fixed codebook [n_codes, state_dim] with normalized rows."""
        assert self.code_emit and tuple(code.shape) == tuple(self.code.shape), (
            f"set_code expects [{self.n_codes}, {self.code.size(-1)}], got {tuple(code.shape)}"
        )
        self.code.copy_(F.normalize(code.float(), dim=-1).to(self.code.device))

    def set_concepts(self, facets: "list[tuple[str, torch.Tensor, int]]") -> None:
        """Install named concept facets as independent codebook spans."""
        total_m = sum(c.shape[0] for _, c, _ in facets)
        total_d = sum(d for _, _, d in facets)
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

    def concept_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalize each named facet independently and concatenate the probabilities."""
        flat = logits.float().squeeze(-2)
        out = torch.zeros_like(flat)
        for m_lo, m_hi, _, _ in self.concept_spans:
            out[..., m_lo:m_hi] = flat[..., m_lo:m_hi].softmax(-1)
        return out

    def residual(self, hid: torch.Tensor) -> torch.Tensor:
        """Return the normalized unnamed residual portion of an emission."""
        if self.res_proj is None:
            return hid.new_zeros(*hid.shape[:-1], 0)
        return F.normalize(self.res_proj(self.drop(hid.float())), dim=-1)

    def encode(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map target vectors to nearest code indices for teacher-forcing bookkeeping."""
        assert self.code_emit, "encode(target) is only defined for a codebook emission head"
        state = F.normalize(target.float()[..., : self.state_dim], dim=-1)
        idx = (state @ self.code.t()).argmax(-1, keepdim=True)
        return idx, target.float()

    def reconstruct(self, codes: torch.Tensor) -> torch.Tensor:
        """Map code indices back to their fixed state vectors."""
        assert self.code_emit
        return self.code[codes.long().squeeze(-1)]

    def emit_logits(self, hid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return code logits ``[..., 1, n_codes]`` and an independent STOP logit."""
        assert self.code_emit and self.query_proj is not None and self.stop_proj is not None
        query = F.normalize(self.query_proj(self.drop(hid.float())), dim=-1)
        return (
            (query @ self.code.t()).unsqueeze(-2) / self.code_tau,
            self.stop_proj(hid.float()),
        )

    def commit(self, logits: torch.Tensor, hid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Commit a code distribution to the superposed state vector fed into the next step."""
        assert self.code_emit
        state = F.normalize(self.concept_probs(logits) @ self.code, dim=-1)
        if self.res_dim == 0:
            return state
        assert hid is not None, "commit(): residual emission requires the emit hidden"
        return torch.cat([state, self.residual(hid)], -1) * (0.5**0.5)

    def stop_logit(self, hidden: torch.Tensor, tok_embed: nn.Module) -> torch.Tensor:
        """Alignment to the model's EOS embedding for the ordinary continuous path."""
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
    """Return the text transformer inside a PEFT-wrapped generation model.

    Calling the text tower directly avoids the vocabulary projection and any vision tower while retaining LoRA
    modules injected into the language layers. The returned module exposes hidden states rather than LM logits.
    """
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
        # Restrict LoRA to the top K transformer layers when requested. A value of zero leaves all eligible layers
        # adapted. Emission reads the final hidden state, so upper-layer adapters can directly shape its representation.
        return (
            list(range(max(0, n_layers - lora_top_k), n_layers))
            if lora_top_k and n_layers
            else None
        )

    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    dt = torch.bfloat16 if bf16 else torch.float32

    if not pretrained:
        # Build the requested architecture from configuration without loading pretrained weights. The complete
        # backbone is trainable, and `vocab_size` keeps its new embedding table aligned with a decoupled tokenizer.
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
        # SDPA avoids materializing the eager attention score matrix. Some multimodal wrappers reject
        # `attention_dropout`, so retry without that keyword when necessary.
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

    # Treat an explicitly selected nonstandard attention implementation as a strict requirement. Default SDPA and
    # eager modes may fall back to one another for model compatibility, but an explicit optimized kernel must not be
    # silently replaced with a slower implementation.
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
    # The model consumes hidden states only. Returning the text tower avoids allocating full-vocabulary logits with
    # shape `[batch, sequence, vocabulary]` on every forward.
    peft = get_peft_model(base, lora)
    if train_base:
        # `train_base` requests full fine-tuning rather than adapter-only training, so next-token replay can update
        # all backbone parameters.
        for p in peft.parameters():
            p.requires_grad_(True)
    if grad_ckpt:
        # Trade additional computation for lower activation memory. Caching must be disabled, and input gradients are
        # enabled so checkpointed segments remain connected when embeddings are frozen and only LoRA is trainable.
        base.config.use_cache = False
        peft.gradient_checkpointing_enable()
        peft.enable_input_require_grads()
    return _text_tower(
        cast("_Backbone", peft)
    )  # PeftModel|PeftMixedModel don't structurally match the Protocol


class LangSetModel(nn.Module):
    """Wrap a language-model backbone with continuous or codebook-based emission heads.

    Depending on its configuration, a model can produce a single embedding, multiple learned latent vectors,
    or an autoregressive sequence of named-state emissions. Use :meth:`from_pretrained` or :meth:`from_scratch`
    to construct a model, :meth:`encode` for Sentence-Transformer-compatible embeddings, and :meth:`rollout` for
    autoregressive codebook emission.
    """

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
        # Gemma E-series models use per-layer embeddings indexed by token ID. Pass `per_layer_inputs` explicitly:
        # real tokens receive their PLE values, while synthetic emit and feedback positions receive zeros. A zero
        # `_ple_dim` identifies a backbone without per-layer embeddings.
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
        eos_id = int(tokenizer.eos_token_id or 0)
        self.head = EmitHead(
            self.h,
            latent_dim,
            n_latents,
            dropout,
            eos_id=eos_id,
            multi_latent=multi_latent,
            code_emit=code_emit,
            n_codes=n_codes,
            code_tau=code_tau,
            res_dim=res_dim,
        )
        self.llm_model = llm_model
        self.max_len = max_len
        self._lora_top_k = 0  # overwritten by from_pretrained; persisted in config
        # `pool_mode="last"` bypasses the learned emit query and projects the final real-token hidden state. When
        # the backbone is frozen, this allows head-only training without retaining a backbone autograd graph. The
        # empty mode uses the learned-query emission path.
        self.pool_mode = pool_mode
        self._frozen_bb = (
            False  # set by from_pretrained(freeze_backbone=True); gates the no-grad backbone read
        )
        # A pretrained model can rebuild its base weights from `llm_model`, so persistence needs only its adapters.
        # A randomly initialized model has no external weight source and must persist the complete backbone.
        self._pretrained = True
        # Full fine-tuning changes base weights that cannot be reconstructed from `llm_model`; those runs persist the
        # complete backbone just like randomly initialized models.
        self._full_ft = False
        self._tokenizer_id: Optional[str] = (
            None  # decoupled HF tokenizer id (None => same as llm_model/arch)
        )
        self._arch_overrides: Optional[dict] = (
            None  # config shrink applied to the from-scratch backbone
        )
        # Persisted auxiliary heads are registered by name with metadata describing their read site, loss, dimensions,
        # and optional classes. Transient trainer-only heads are not registered here.
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
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
        train_base: bool = False,
        grad_ckpt: bool = False,
        lora_top_k: int = 0,
        pool_mode: str = "",
        freeze_backbone: bool = False,
        code_emit: bool = False,
        n_codes: int = 0,
        code_tau: float = 0.07,
        res_dim: int = 0,
    ) -> "LangSetModel":
        """Construct a LangSet model from a pretrained Hugging Face model.

        The backbone is adapted with LoRA by default. Set ``train_base=True`` for full fine-tuning, or combine
        ``pool_mode="last"`` with ``freeze_backbone=True`` to train only the output projection over frozen
        backbone features. When ``latent_dim`` is omitted, it defaults to the backbone hidden size.

        Set ``code_emit=True`` to enable codebook-based emission and configure ``n_codes`` for the codebook size.
        Set ``multi_latent=True`` when the model will be trained with a multi-latent emission strategy. The
        returned model is moved to ``device``, or to CUDA when available if no device is specified.
        """
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
            pool_mode=pool_mode,
            code_emit=code_emit,
            n_codes=n_codes,
            code_tau=code_tau,
            res_dim=res_dim,
        )
        model._lora_top_k = int(lora_top_k)  # persisted in config so load() rebuilds same adapters
        model._full_ft = bool(
            train_base
        )  # full fine-tuning requires persisting the complete backbone
        if pool_mode == "last":  # initialize the projection as an identity mapping
            torch.nn.init.eye_(
                model.head.out_proj.weight
            )  # `head_project` initially preserves the normalized final-token hidden state
            torch.nn.init.zeros_(model.head.out_proj.bias)
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
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
        grad_ckpt: bool = False,
        arch_overrides: Optional[dict] = None,
        code_emit: bool = False,
        n_codes: int = 0,
        code_tau: float = 0.07,
        res_dim: int = 0,
    ) -> "LangSetModel":
        """Construct a LangSet model with randomly initialized backbone weights.

        ``arch`` identifies a Hugging Face configuration, but its pretrained weights are not loaded. The complete
        backbone remains trainable and no LoRA adapters are added. ``tokenizer_id`` defaults to ``arch``, and the
        new input embedding table is sized for that tokenizer. Use ``arch_overrides`` to replace configuration
        fields such as the number of layers, hidden size, or attention-head count. The returned model is moved to
        ``device``, or to CUDA when available if no device is specified.
        """
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
            code_emit=code_emit,
            n_codes=n_codes,
            code_tau=code_tau,
            res_dim=res_dim,
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
        """Return each row's final real-token hidden state with shape ``[batch, hidden_size]``.

        When the backbone is frozen, the forward pass runs without gradient
        tracking so callers can cache these features for projection-head training.
        """
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
        """Project ``[batch, hidden_size]`` pooled features to normalized latent vectors."""
        return F.normalize(self.head.out_proj(feats.float()), p=2, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Emit latent vectors from a tokenized batch.

        Returns ``[batch, latent_dim]`` when one latent is configured, or
        ``[batch, n_latents, latent_dim]`` otherwise.
        """
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
        """Encode one or more sentences using the Sentence-Transformer-compatible interface.

        Inputs are tokenized in batches and truncated to :attr:`max_len`. A single string returns one embedding;
        a sequence returns a batch. Results are NumPy arrays by default or CPU tensors when
        ``convert_to_numpy=False``. ``show_progress_bar`` and ``device`` are accepted for interface compatibility
        but are currently ignored.
        """
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        was_training = self.training
        self.eval()
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
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
        """Encode text and return the result as a PyTorch tensor.

        Additional keyword arguments are forwarded to :meth:`encode`. This is equivalent to calling
        ``encode(..., convert_to_numpy=False)``.
        """
        # `kw` is a typed passthrough to `encode`; `Unknown` avoids weakening the public signature to `Any`.
        return cast("torch.Tensor", self.encode(sentences, convert_to_numpy=False, **kw))

    # --- auxiliary supervised heads (langset.heads) --------------------------------------------------------------
    def seed_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the final-token backbone hidden state for each sequence.

        The batch must be left-padded so every sequence's final real token occupies the last column. The result
        has shape ``[batch, hidden_size]`` and remains connected to trainable backbone parameters.
        """
        hid = self._last_hidden(
            self._run_backbone(self.embed(input_ids), attention_mask, input_ids, 0)
        )
        return hid[:, -1]

    def add_aux_head(self, module: nn.Linear, spec: dict[str, Any]) -> None:
        """Register an auxiliary linear head and its persistence metadata.

        ``spec`` describes the head name, read site, loss, input and output dimensions, and optional class labels.
        Registered heads are serialized by :meth:`save_pretrained` and can be queried with :meth:`head_output`.
        """
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
        """Run a persisted auxiliary head on one or more sentences.

        Heads with ``reads="hidden"`` consume one pooled seed representation per sentence. Heads with
        ``reads="recon"`` consume every latent produced by :meth:`rollout`; ``reduce="mean"`` averages each
        sequence, while ``reduce="none"`` preserves the per-step outputs. MSE heads typically return scalar
        values. Classification heads return logits whose columns correspond to
        ``aux_head_specs[name]["classes"]``.

        Raises:
            KeyError: If ``name`` is not registered.
            ValueError: If the reduction or the head's read site is unsupported.
        """
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
        """Generate text greedily using the tied input embeddings as the output projection.

        Generation stops at the tokenizer's EOS token or after ``max_new`` tokens. The model is left in evaluation
        mode after this call.
        """
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

    # ---- named-state autoregressive rollout -----------------------------------------------------
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
        """Autoregressively emit codebook-based latent states.

        Each emitted state is fed back into the backbone until its stop logit exceeds ``stop_threshold`` or
        ``max_steps`` is reached. This method requires a model constructed with ``code_emit=True``.
        ``temperature <= 0`` uses the learned code distribution unchanged; positive values rescale its logits.
        Batched outputs zero-pad halted rows, while outputs for a single string are trimmed to their emitted length.

        By default, returns the emitted latents. ``return_lengths`` adds per-row lengths; ``return_confidence``
        returns ``(latents, lengths, confidence)``; and ``return_soft`` returns
        ``(latents, lengths, committed_latents, entropy)``. These flags are checked in that precedence order:
        ``return_soft``, then ``return_confidence``, then ``return_lengths``.
        """
        if not self.head.code_emit:
            raise ValueError(
                "rollout() requires a named codebook emitter. Use model.encode() for one continuous vector, "
                "or QueryBridgeEmission.emit_infer() for an open-ended continuous vector set."
            )
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        dev = self.device
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            padding_side="left",
            return_tensors="pt",
        ).to(dev)
        seq = self.embed(enc["input_ids"])
        am = enc["attention_mask"]
        b = seq.size(0)
        alive = torch.ones(b, dtype=torch.bool, device=dev)
        lengths = torch.zeros(b, dtype=torch.long, device=dev)
        cols: list[torch.Tensor] = []
        ent_cols: list[torch.Tensor] = []
        conf_code: list[torch.Tensor] = []
        conf_stop: list[torch.Tensor] = []
        was_training = self.training
        self.eval()
        for _ in range(max_steps):
            hid = self._last_hidden(self._run_backbone(seq, am, enc["input_ids"], 0))[:, -1]
            logits, stop_logits = self.head.emit_logits(hid)
            scaled = logits if temperature <= 0 else logits / temperature
            probs = self.head.concept_probs(scaled)
            state = F.normalize(probs @ self.head.code, dim=-1)
            z = (
                torch.cat([state, self.head.residual(hid)], -1) * (0.5**0.5)
                if self.head.res_dim
                else state
            )
            stop = stop_logits.squeeze(-1) > stop_threshold
            emit_now = alive & ~stop
            z = torch.where(emit_now.unsqueeze(-1), z, torch.zeros_like(z))
            cols.append(z)
            p = probs.clamp_min(1e-9)
            ent_cols.append(
                torch.where(emit_now, -(p.log() * p).sum(-1), torch.zeros(b, device=dev))
            )
            conf_code.append(probs.max(-1).values)
            conf_stop.append(torch.sigmoid(stop_logits.squeeze(-1)))
            lengths = lengths + emit_now.long()
            seq = torch.cat([seq, self.head.feedback(z).unsqueeze(1).to(seq.dtype)], 1)
            am = torch.cat([am, emit_now.long().unsqueeze(1)], 1)
            alive = emit_now
            if not bool(alive.any()):
                break
        if was_training:
            self.train()
        lat = torch.stack(cols, 1) if cols else seq.new_zeros(b, 0, self.latent_dim)
        ent = torch.stack(ent_cols, 1) if ent_cols else lat.new_zeros(b, 0)
        conf = {
            "code": torch.stack(conf_code, 1) if conf_code else lat.new_zeros(b, 0),
            "stop": torch.stack(conf_stop, 1) if conf_stop else lat.new_zeros(b, 0),
        }
        if single:
            n = int(lengths[0])
            lat, ent = lat[0, :n], ent[0, :n]
            conf = {k: v[0, :n] for k, v in conf.items()}
        if return_soft:
            return lat, lengths, lat, ent
        if return_confidence:
            return lat, lengths, conf
        if return_lengths:
            return lat, lengths
        return lat

    def rollout_train_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_latents: torch.Tensor,
        train_hops: Optional[int] = None,
        ss_prob: float = 0.0,
        ss_mask: Optional[torch.Tensor] = None,
        kv_cache: bool = False,
        return_emit_hidden: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Run an autoregressive concept/codebook training pass.

        Targets are mapped to nearest code indices for code-classification objectives while their exact vectors are
        retained for teacher forcing. Returns code logits with shape ``[B, L + 1, 1, n_codes]``, stop logits, code
        indices, and exact target vectors. When ``return_emit_hidden=True``, the corresponding emission hidden states
        are appended to the result.

        With ``ss_prob=0``, all positions are predicted from the true prefix in one forward pass. With a positive
        probability, the first ``train_hops`` positions may consume detached model emissions instead of ground-truth
        latents; later positions remain teacher-forced. This scheduled-sampling path requires multiple backbone
        forwards.

        ``ss_mask`` optionally supplies precomputed ``[B, H]`` self-feed decisions. Reusing the same mask makes
        GradCache's detached and gradient-enabled passes produce aligned emissions and cached gradients.
        """
        assert self.head.multi_latent and self.head.code_emit
        bsz, s_len = input_ids.size(0), input_ids.size(1)
        n = target_latents.size(1)
        codes, recon = self.head.encode(target_latents.reshape(-1, target_latents.size(-1)))
        codes = codes.view(bsz, n, -1)  # [B, L, 1]
        recon = recon.view(bsz, n, -1)  # [B, L, d] — clean feedback + recon target
        H = n if train_hops is None else max(0, min(int(train_hops), n))
        if ss_prob <= 0 or n == 0 or H == 0:  # one-pass teacher forcing
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
            code_lg, stop_lg = self.head.emit_logits(hf)  # [B, L+1, 1, n_codes], [B, L+1, 1]
            return (
                (code_lg, stop_lg, codes, recon, hf)
                if return_emit_hidden
                else (code_lg, stop_lg, codes, recon)
            )
        # Scheduled-sampling path: self-fed positions are evaluated serially.
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
                recon_pred = self.head.commit(dl_h, hid=hid).detach()
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
            out4 = (torch.cat(dim_parts, 1), torch.cat(stop_parts, 1), codes, recon)
            return (*out4, torch.cat(hid_parts, 1)) if return_emit_hidden else out4
        seq = self.embed(input_ids)
        am = attention_mask
        dim_parts: list[torch.Tensor] = []
        stop_parts: list[torch.Tensor] = []
        hid_parts: list[torch.Tensor] = []
        for h in range(H):  # AR self-feed region
            hid = self._last_hidden(self._run_backbone(seq, am, input_ids, 0))[:, -1]
            dl, sl = self.head.emit_logits(hid)  # [B, 1, n_codes], [B, 1]
            dim_parts.append(dl.unsqueeze(1))
            stop_parts.append(sl.unsqueeze(1))
            hid_parts.append(hid.unsqueeze(1))
            recon_pred = self.head.commit(dl, hid=hid).detach()  # own emitted latent (detached)
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
        dl_rest, sl_rest = self.head.emit_logits(hf_rest)  # [B, n-H+1, 1, n_codes], [B, n-H+1, 1]
        code_lg = (
            torch.cat(dim_parts + [dl_rest], 1) if dim_parts else dl_rest
        )  # [B, n+1, 1, n_codes]
        stop_lg = torch.cat(stop_parts + [sl_rest], 1) if stop_parts else sl_rest  # [B, n+1, 1]
        if return_emit_hidden:
            emit_hid = torch.cat(hid_parts + [hf_rest], 1) if hid_parts else hf_rest
            return code_lg, stop_lg, codes, recon, emit_hid
        return code_lg, stop_lg, codes, recon

    def get_sentence_embedding_dimension(self) -> int:
        """Return the configured latent embedding dimension."""
        return self.latent_dim

    def as_sentence_transformer(self) -> SentenceTransformer:
        """Return a SentenceTransformer wrapper suitable for APIs such as SetFit."""
        from langset.st_module import to_sentence_transformer

        return to_sentence_transformer(self)

    # ---- persistence (LoRA + head + config; backbone rebuilt from ids) ----
    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the LangSet configuration and trainable state to a directory.

        Pretrained models whose base weights can be reconstructed store the emission head and LoRA adapters.
        Randomly initialized or fully fine-tuned models store the complete backbone. Persisted auxiliary heads and
        emission-bridge state are included when present.
        """
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
                    "lora_top_k": self._lora_top_k,
                    "emission_family": (
                        "query_bridge"
                        if hasattr(self, "emission_bridge")
                        else "codebook"
                        if self.head.code_emit
                        else "continuous"
                    ),
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
        """Load a model saved by :meth:`save_pretrained`.

        Reconstructs the original pretrained or randomly initialized backbone, restores the saved trainable state,
        rebuilds persisted auxiliary heads, moves the model to ``device``, and returns it in evaluation mode.
        Checkpoints using the removed FSQ emitter are rejected.
        """
        import json

        p = Path(path)
        cfg = json.loads((p / "config.json").read_text())
        sd = torch.load(p / "langset.pt", map_location=device or "cpu", weights_only=False)
        legacy_fsq = cfg.get("fsq_emit", False) or (
            cfg.get("multi_latent", False)
            and not cfg.get("code_emit", False)
            and "emission_bridge" not in sd
            and "fsq_dim" in cfg
        )
        if legacy_fsq:
            raise ValueError(
                "This checkpoint uses the removed FSQ emitter. Retrain with ConceptObjective, "
                "StateResidualObjective, CodeSoftmaxObjective, or QueryBridgeEmission."
            )
        if cfg.get("pretrained", True):
            m = cls.from_pretrained(
                cfg["llm_model"],
                latent_dim=cfg["latent_dim"],
                n_latents=cfg.get("n_latents", 1),
                lora_r=lora_r,
                max_len=cfg["max_len"],
                multi_latent=cfg.get("multi_latent", False),
                lora_top_k=int(cfg.get("lora_top_k", 0)),
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
                arch_overrides=cfg.get("arch_overrides"),
                code_emit=cfg.get("code_emit", False),
                n_codes=int(cfg.get("n_codes", 0)),
                code_tau=float(cfg.get("code_tau", 0.07)),
                res_dim=int(cfg.get("res_dim", 0)),
                device=device,
                attn_implementation=attn_implementation,
            )
        m._full_ft = bool(
            cfg.get("full_ft", False)
        )  # restore the flag so a re-save round-trips the same way
        if cfg.get("concept_spans"):
            m.head.concept_spans = [tuple(x) for x in cfg["concept_spans"]]
            m.head.concept_names = list(cfg.get("concept_names") or [])
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
