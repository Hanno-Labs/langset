"""LangSetModel: an LLM (LoRA) + a learned emit head that maps input text -> a latent in a bespoke geometry.

The latent lives in the model's OWN hidden space, and the output is Sentence-Transformer-shaped (`encode`,
`get_sentence_embedding_dimension`, `as_sentence_transformer`) so the trained model drops straight into SetFit
as a `model_body`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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

    def __init__(self, h: int, d: int, n_latents: int = 1, dropout: float = 0.0, eos_id: int = 0,
                 multi_latent: bool = False, fsq_dim: int = 128, fsq_levels: int = 8) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.q = nn.Parameter(torch.randn(n_latents, h) * 0.02)   # one query token per emitted latent
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(h, d)                           # hidden -> latent (targets/bank/continuous)
        self.in_proj = nn.Linear(d, h)                            # latent -> hidden (inverse projection)
        self.eos_id = eos_id
        self.multi_latent = multi_latent
        self.fsq_dim, self.fsq_levels = fsq_dim, fsq_levels
        self.down_proj = nn.Linear(d, fsq_dim) if multi_latent else None        # d -> FSQ bottleneck (learned)
        self.up_proj = nn.Linear(fsq_dim, d) if multi_latent else None          # FSQ -> d (learned decoder)
        self.level_proj = nn.Linear(h, fsq_dim * fsq_levels) if multi_latent else None   # per-dim digit logits
        self.stop_proj = nn.Linear(h, 1) if multi_latent else None              # STOP logit (folded into dim-0 softmax)

    def forward(self, hid_emit: torch.Tensor) -> torch.Tensor:    # [B, n_latents, h] -> [B, n_latents, d]
        return F.normalize(self.out_proj(self.drop(hid_emit.float())), p=2, dim=-1)

    def feedback(self, latent: torch.Tensor) -> torch.Tensor:     # [B, ..., d] -> [B, ..., h]
        return self.in_proj(latent.float()).to(latent.dtype)     # head stays fp32; match the backbone's dtype (bf16)

    def fsq(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Finite scalar quantize z [.., fsq_dim] -> (zq straight-through, digits). Both live in the SAME [0, L-1]
        space so training (up_proj(zq)) and inference (up_proj(digits)) reconstruct identically."""
        lvl = self.fsq_levels - 1
        zb = (torch.tanh(z) + 1.0) * 0.5 * lvl                    # [0, L-1]
        zr = zb.round().clamp(0, lvl)
        zq = zb + (zr - zb).detach()                             # straight-through, in [0, L-1]
        return zq, zr.long()

    def encode(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Target embedding -> (digits [.., fsq_dim], recon [.., d]). Grad flows to down/up_proj (recon loss)."""
        assert self.down_proj is not None and self.up_proj is not None
        zq, digits = self.fsq(self.down_proj(t.float()))
        return digits, self.up_proj(zq)

    def reconstruct(self, digits: torch.Tensor) -> torch.Tensor:  # digits [.., fsq_dim] in [0, L-1] -> [.., d]
        assert self.up_proj is not None
        return self.up_proj(digits.float())

    def emit_logits(self, hid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[.., h] -> (per-dim digit logits [.., fsq_dim, fsq_levels], STOP logit [.., 1])."""
        assert self.level_proj is not None and self.stop_proj is not None
        lg = self.level_proj(self.drop(hid.float())).unflatten(-1, (self.fsq_dim, self.fsq_levels))
        return lg, self.stop_proj(hid.float())

    def stop_logit(self, hidden: torch.Tensor, tok_embed: nn.Module) -> torch.Tensor:
        """Alignment of the hidden state to the model's real EOS embedding (continuous-mode terminator)."""
        emb_eos = cast(torch.Tensor, tok_embed.weight)[self.eos_id].float()
        return hidden.float() @ emb_eos


def _cfg_int(config: Any, name: str) -> int:
    """Read a scalar (hidden_size / vocab_size) that may live on a composite config's text sub-config."""
    v = getattr(config, name, None)
    if v is None and hasattr(config, "text_config"):
        v = getattr(config.text_config, name, None)
    if v is None and hasattr(config, "get_text_config"):
        v = getattr(config.get_text_config(), name, None)
    if v is None:
        raise AttributeError(f"config has no {name}")
    return int(v)


def _text_tower(model: Any) -> Any:
    """Descend a (peft-wrapped) causal/conditional-generation model to its TEXT transformer — the module that
    returns hidden states directly, with NO lm_head. Skips the huge-vocab logits projection (Gemma's 262k-vocab
    lm_head over a full sequence OOMs — we only ever read hidden states) and any vision tower. LoRA is injected
    in-place on the language Linears, so calling the text tower directly still applies it."""
    node = getattr(getattr(model, "base_model", model), "model", model)   # peft LoraModel -> underlying HF model
    for _ in range(4):
        nxt = getattr(node, "language_model", None)                       # VLM container -> text tower
        if nxt is not None and nxt is not node:
            node = nxt; continue
        if hasattr(node, "lm_head") and hasattr(node, "model"):           # ForCausalLM/CondGen -> inner text model
            node = node.model; continue
        break
    return node


def build_backbone(llm_model: str, lora_r: int, dropout: float, bf16: bool, dev: str) -> Any:
    if llm_model.startswith("unsloth/"):        # Unsloth patches Gemma-4 clippable-linears so PEFT LoRA can attach
        from unsloth import FastModel  # type: ignore[import-untyped]  # (must be imported before transformers)
        base, _ = FastModel.from_pretrained(model_name=llm_model, dtype=None, max_seq_length=4096,
                                            load_in_4bit=True, full_finetuning=False)
        # Gemma-4 E-series shares K/V module-scoped: with grad-checkpointing, our 2+ forwards-before-backward
        # (contrastive + recon) corrupt the first graph on recompute -> GC OFF for it. Every other model is fine
        # with GC and NEEDS it to fit (3 no-GC forwards on a 4B model exceed 178GB) -> keep GC ON.
        use_gc = False if "gemma-4-e" in llm_model.lower() else "unsloth"
        peft_model = FastModel.get_peft_model(
            base, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=lora_r, lora_alpha=lora_r, lora_dropout=dropout, bias="none", random_state=3407,
            use_gradient_checkpointing=use_gc)
        return _text_tower(peft_model)          # hidden states w/o the 262k-vocab lm_head (OOM) or vision path
    from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]
    dt = torch.bfloat16 if bf16 else torch.float32
    kw: dict[str, Any] = {"dtype": dt, "attn_implementation": "sdpa"}
    try:
        base = AutoModelForCausalLM.from_pretrained(llm_model, attention_dropout=dropout, **kw)
    except TypeError:                        # multimodal wrappers (e.g. Gemma4ForConditionalGeneration) reject it
        base = AutoModelForCausalLM.from_pretrained(llm_model, **kw)
    if hasattr(base, "language_model"):      # unwrap conditional-generation wrapper to the text tower
        base = base.language_model
    base = base.to(dev)
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=dropout, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(base, lora)


class LangSetModel(nn.Module):
    """LLM backbone (LoRA) + EmitHead. The latent lives in the model's own hidden space; the geometry is defined
    by the `target_text` the Trainer contrasts against (see Trainer)."""

    def __init__(self, backbone: Any, tokenizer: Any, latent_dim: int, n_latents: int,
                 llm_model: str, dropout: float = 0.0, max_len: int = 512,
                 multi_latent: bool = False, fsq_dim: int = 128, fsq_levels: int = 8) -> None:
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
        # The Unsloth path loads a text tower that always returns `last_hidden_state`, so we don't ask the backbone
        # to collect (and keep) every layer's hidden states — a big memory win. A plain ForCausalLM has no
        # `last_hidden_state`, so it still needs output_hidden_states to expose the final layer.
        self._need_ohs = not str(llm_model).startswith("unsloth/")
        self.latent_dim = latent_dim
        self.n_latents = n_latents
        self.multi_latent = multi_latent
        self.fsq_dim = fsq_dim
        self.fsq_levels = fsq_levels
        eos_id = int(tokenizer.eos_token_id or 0)
        self.head = EmitHead(self.h, latent_dim, n_latents, dropout, eos_id=eos_id,
                             multi_latent=multi_latent, fsq_dim=fsq_dim, fsq_levels=fsq_levels)
        self.llm_model = llm_model
        self.max_len = max_len

    # ---- construction ----
    @classmethod
    def from_pretrained(cls, llm_model: str, *, latent_dim: Optional[int] = None, n_latents: int = 1,
                        lora_r: int = 16, dropout: float = 0.0, bf16: bool = False, max_len: int = 512,
                        multi_latent: bool = False, fsq_dim: int = 128, fsq_levels: int = 8,
                        device: Optional[str] = None) -> "LangSetModel":
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(llm_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        backbone = build_backbone(llm_model, lora_r, dropout, bf16, dev)
        if latent_dim is None:                                   # default: emit in the backbone's own hidden space
            latent_dim = _cfg_int(backbone.config, "hidden_size")
        model = cls(backbone, tok, latent_dim, n_latents, llm_model, dropout, max_len,
                    multi_latent, fsq_dim, fsq_levels)
        if llm_model.startswith("unsloth/"):                     # 4bit backbone is already placed; move only the head
            model.head.to(dev)
            return model
        return model.to(dev)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # ---- forward / inference ----
    def _run_backbone(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                      real_ids: Optional[torch.Tensor] = None, real_start: int = 0) -> Any:
        """Backbone forward that stays correct on Per-Layer-Embedding models (Gemma E-series). For PLE models we
        build `per_layer_inputs` ourselves: the real-token span [real_start : real_start+len] gets its true
        token-ID lookup; synthetic positions (emit query / fed-back latents / recon soft tokens) get zeros, so
        their per-layer contribution is projection-only and the crashing embed->ID reverse lookup never runs.
        A no-op for non-PLE backbones (identical to a plain inputs_embeds forward)."""
        kw: dict[str, Any] = {}
        if self._ple_dim:
            b, s = inputs_embeds.shape[:2]
            ple = inputs_embeds.new_zeros(b, s, self._n_layers, self._ple_dim)
            if real_ids is not None:
                real = self.backbone.get_per_layer_inputs(real_ids, None).to(ple.dtype)
                ple[:, real_start:real_start + real_ids.size(1)] = real
            kw["per_layer_inputs"] = ple
        return self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             output_hidden_states=self._need_ohs, **kw)

    @staticmethod
    def _last_hidden(out: Any) -> torch.Tensor:
        h = getattr(out, "last_hidden_state", None)          # text tower returns this; a ForCausalLM does not
        return h if h is not None else out.hidden_states[-1]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Read input text, emit the latent. Returns [B, d]."""
        nl = self.head.n_latents
        rev = self.embed(input_ids)
        q = self.head.q.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(rev.dtype)
        emb = torch.cat([rev, q], 1)
        am = torch.cat([attention_mask,
                        torch.ones(input_ids.size(0), nl, device=input_ids.device,
                                   dtype=attention_mask.dtype)], 1)
        hid = self._last_hidden(self._run_backbone(emb, am, input_ids, 0))   # real tokens front, query appended
        z = self.head(hid[:, -nl:, :])                          # [B, n_latents, d]
        return z.squeeze(1) if nl == 1 else z

    @torch.no_grad()
    def encode(self, sentences: Union[str, list[str]], batch_size: int = 32, convert_to_numpy: bool = True,
               normalize_embeddings: bool = True, show_progress_bar: bool = False,
               device: Optional[str] = None) -> Union[np.ndarray, torch.Tensor]:
        """Sentence-Transformer-compatible. This is the method SetFit calls on its body."""
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        was_training = self.training
        self.eval()
        out: list[torch.Tensor] = []
        with torch.no_grad():               # eval-only: LoRA params require grad, so w/o this every forward builds
            for i in range(0, len(texts), batch_size):   # a throwaway autograd graph -> slower + huge VRAM (caps batch)
                enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                     max_length=self.max_len, return_tensors="pt").to(self.device)
                z = self(enc["input_ids"], enc["attention_mask"])
                if normalize_embeddings:
                    z = F.normalize(z, p=2, dim=-1)
                out.append(z.float().cpu())     # fp32 so .numpy() works even with a bf16 backbone
        if was_training:
            self.train()
        emb = torch.cat(out)
        emb = emb[0] if single else emb
        return emb.numpy() if convert_to_numpy else emb

    def emit(self, sentences: Union[str, list[str]], **kw: Any) -> torch.Tensor:
        return self.encode(sentences, convert_to_numpy=False, **kw)  # type: ignore[return-value]

    # ---- multi-latent autoregressive rollout (drives the feedback / stop_logit seams) ----
    @torch.no_grad()
    def rollout(self, text: Union[str, list[str]], max_steps: int = 8, stop_threshold: float = 0.0,
                return_lengths: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Autoregressive emission: real text tokens and emitted latents share ONE hidden stream. Each emitted
        latent is fed back into the sequence via `head.feedback` (the latent lives in the backbone's own hidden
        space, so an emitted latent and a real token embedding are the same kind of vector — they intermingle).
        Terminates per-row on the natural-EOS `stop_logit`. Returns [B, L, d] (or [L, d] for a single string),
        zero-padding halted rows; pass `return_lengths` for the per-row emit count."""
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        dev = self.device
        codebook = self.head.multi_latent                                   # token-native (multi-latent) FSQ path
        stop_idx = self.head.fsq_levels if codebook else -1                 # STOP is the extra class in dim-0's softmax
        pad_side = "left" if codebook else self.tokenizer.padding_side      # left-pad so [:, -1] is the last real token
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len,
                             padding_side=pad_side, return_tensors="pt").to(dev)
        seq = self.embed(enc["input_ids"])                                  # [B, S, h] — real TEXT tokens
        am = enc["attention_mask"]
        b = seq.size(0)
        alive = torch.ones(b, dtype=torch.bool, device=dev)
        lengths = torch.zeros(b, dtype=torch.long, device=dev)
        cols: list[torch.Tensor] = []
        was_training = self.training
        self.eval()
        for _step in range(max_steps):
            if codebook:                                                     # AR: read the LAST real token's hidden
                hid = self._last_hidden(self._run_backbone(seq, am, enc["input_ids"], 0))[:, -1]  # PLE-safe AR read
                dim_lg, stop_lg = self.head.emit_logits(hid)                # [B, fsq_dim, L], [B, 1]
                dim0 = torch.cat([dim_lg[:, 0, :], stop_lg], -1).argmax(-1)  # [B] over [L levels; STOP]
                emit_now = alive & (dim0 != stop_idx)                       # halt on STOP (or already-dead rows)
                digits = torch.cat([dim0.clamp(max=self.head.fsq_levels - 1).unsqueeze(-1),
                                    dim_lg[:, 1:, :].argmax(-1)], -1)       # [B, fsq_dim]
                z = self.head.reconstruct(digits)                          # [B, d]
                cols.append(torch.where(emit_now.unsqueeze(1), z, torch.zeros_like(z)))
                lengths = lengths + emit_now.long()
                seq = torch.cat([seq, self.head.feedback(z).unsqueeze(1).to(seq.dtype)], 1)
                am = torch.cat([am, emit_now.long().unsqueeze(1)], 1)
                alive = emit_now
            else:
                q = self.head.q[:1].unsqueeze(0).expand(b, 1, -1).to(seq.dtype)  # one read-query token
                am_q = torch.cat([am, am.new_ones(b, 1)], 1)
                hid = self._last_hidden(self._run_backbone(torch.cat([seq, q], 1), am_q,
                                                           enc["input_ids"], 0))[:, -1]   # [B, h] PLE-safe
                z = self.head(hid.unsqueeze(1)).squeeze(1)                   # [B, d] emit via out_proj
                cols.append(torch.where(alive.unsqueeze(1), z, torch.zeros_like(z)))
                lengths = lengths + alive.long()
                stop = self.head.stop_logit(hid, self.embed) > stop_threshold  # emit-then-check: the hidden that emitted
                seq = torch.cat([seq, self.head.feedback(z).unsqueeze(1).to(seq.dtype)], 1)  # this latent also carries
                am = torch.cat([am, alive.long().unsqueeze(1)], 1)         # its natural-EOS logit (aligns with the last-
                alive = alive & ~stop                                      # position EOS label in rollout_train)
            if not bool(alive.any()):
                break
        if was_training:
            self.train()
        lat = torch.stack(cols, 1) if cols else seq.new_zeros(b, 0, self.latent_dim)   # [B, L, d]
        if single:
            lat = lat[0, : int(lengths[0])]
        if return_lengths:
            return lat, lengths
        return lat

    def rollout_train(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      target_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced AR pass for training: feed the TRUE latents back through `feedback` and predict the next
        emission at each future position in ONE forward pass (no python loop). Returns (preds [B, L, d],
        stop_logits [B, L]) — the per-step emission and its natural-EOS logit. (Continuous, single-latent path.)"""
        bsz, s_len = input_ids.size(0), input_ids.size(1)
        n = target_latents.size(1)
        seed = self.embed(input_ids)                                        # [B, S, h]
        fb = self.head.feedback(target_latents.to(seed.dtype))              # [B, L, h] — true latents fed back
        seq = torch.cat([seed, fb], 1)
        am = torch.cat([attention_mask, attention_mask.new_ones(bsz, n)], 1)
        hid = self._last_hidden(self._run_backbone(seq, am, input_ids, 0))   # PLE-safe teacher-forced read
        hf = hid[:, s_len - 1: s_len - 1 + n]                               # [B, L, h] — predicts future positions
        preds = self.head(hf)                                              # [B, L, d]
        stop = self.head.stop_logit(hf, self.embed)                        # [B, L]
        return preds, stop

    def rollout_train_codebook(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                               target_latents: torch.Tensor, tau: float = 0.07
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token-native teacher-forced AR pass (multi-latent, FSQ). Quantizes the TRUE target latents to per-dim
        digits, feeds the clean reconstruction back, and predicts the digits at each of L+1 positions (targets
        0..L-1 then a STOP at position L). Returns (dim_logits [B, L+1, fsq_dim, fsq_levels], stop_logits
        [B, L+1, 1], digits [B, L, fsq_dim], recon [B, L, d]) — predictions, the target digits, and the FSQ
        reconstruction (whose grad trains down/up_proj). The STOP label is appended by the trainer."""
        assert self.head.multi_latent
        bsz, s_len = input_ids.size(0), input_ids.size(1)
        n = target_latents.size(1)
        digits, recon = self.head.encode(target_latents.reshape(-1, target_latents.size(-1)))
        digits = digits.view(bsz, n, -1)                                    # [B, L, fsq_dim]
        recon = recon.view(bsz, n, -1)                                      # [B, L, d] — clean feedback + recon target
        seed = self.embed(input_ids)
        fb = self.head.feedback(recon.detach().to(seed.dtype))             # [B, L, h] — feedback (no grad through fb)
        seq = torch.cat([seed, fb], 1)
        am = torch.cat([attention_mask, attention_mask.new_ones(bsz, n)], 1)
        hid = self._last_hidden(self._run_backbone(seq, am, input_ids, 0))   # PLE-safe teacher-forced read
        hf = hid[:, s_len - 1: s_len - 1 + n + 1]                           # [B, L+1, h] — +1 to predict the STOP
        dim_lg, stop_lg = self.head.emit_logits(hf)                        # [B, L+1, fsq_dim, L], [B, L+1, 1]
        return dim_lg, stop_lg, digits, recon

    def get_sentence_embedding_dimension(self) -> int:
        return self.latent_dim

    def as_sentence_transformer(self) -> Any:
        """Wrap as a `sentence_transformers.SentenceTransformer` so it drops into SetFit as `model_body`."""
        from langset.st_module import to_sentence_transformer
        return to_sentence_transformer(self)

    # ---- persistence (LoRA + head + config; backbone rebuilt from ids) ----
    def save_pretrained(self, path: Union[str, Path]) -> None:
        import json
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save({"head": self.head.state_dict(),
                    "lora": {k: v.cpu() for k, v in self.backbone.state_dict().items() if "lora" in k}},
                   p / "langset.pt")
        (p / "config.json").write_text(json.dumps({
            "llm_model": self.llm_model, "latent_dim": self.latent_dim,
            "n_latents": self.head.n_latents, "max_len": self.max_len,
            "multi_latent": self.multi_latent, "fsq_dim": self.fsq_dim, "fsq_levels": self.fsq_levels}))

    @classmethod
    def load(cls, path: Union[str, Path], *, lora_r: int = 16, device: Optional[str] = None) -> "LangSetModel":
        import json
        p = Path(path); cfg = json.loads((p / "config.json").read_text())
        m = cls.from_pretrained(cfg["llm_model"], latent_dim=cfg["latent_dim"], n_latents=cfg.get("n_latents", 1),
                                lora_r=lora_r, max_len=cfg["max_len"], multi_latent=cfg.get("multi_latent", False),
                                fsq_dim=cfg.get("fsq_dim", 128), fsq_levels=cfg.get("fsq_levels", 8), device=device)
        sd = torch.load(p / "langset.pt", map_location=m.device, weights_only=False)
        m.backbone.load_state_dict(sd["lora"], strict=False)
        m.head.load_state_dict(sd["head"])
        m.eval()
        return m
