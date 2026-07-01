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

    `dropout` is trained ON when >0 so MC-dropout could later serve as an uncertainty signal. The head stays fp32
    even when the backbone is bf16.
    """

    def __init__(self, h: int, d: int, n_latents: int = 1, dropout: float = 0.0, eos_id: int = 0) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.q = nn.Parameter(torch.randn(n_latents, h) * 0.02)   # one query token per emitted latent
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(h, d)                           # hidden -> latent
        self.in_proj = nn.Linear(d, h)                            # latent -> hidden (inverse projection)
        self.eos_id = eos_id

    def forward(self, hid_emit: torch.Tensor) -> torch.Tensor:    # [B, n_latents, h] -> [B, n_latents, d]
        return F.normalize(self.out_proj(self.drop(hid_emit.float())), p=2, dim=-1)

    def feedback(self, latent: torch.Tensor) -> torch.Tensor:     # [B, ..., d] -> [B, ..., h]
        return self.in_proj(latent)

    def stop_logit(self, hidden: torch.Tensor, tok_embed: nn.Module) -> torch.Tensor:
        """Alignment of the hidden state to the model's real EOS embedding."""
        emb_eos = cast(torch.Tensor, tok_embed.weight)[self.eos_id].float()
        return hidden.float() @ emb_eos


def build_backbone(llm_model: str, lora_r: int, dropout: float, bf16: bool, dev: str) -> Any:
    from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]
    dt = torch.bfloat16 if bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        llm_model, torch_dtype=dt, attention_dropout=dropout, attn_implementation="sdpa").to(dev)  # type: ignore[arg-type]
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=dropout, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(base, lora)


class LangSetModel(nn.Module):
    """LLM backbone (LoRA) + EmitHead. The latent lives in the model's own hidden space; the geometry is defined
    by the `target_text` the Trainer contrasts against (see Trainer)."""

    def __init__(self, backbone: Any, tokenizer: Any, latent_dim: int, n_latents: int,
                 llm_model: str, dropout: float = 0.0, max_len: int = 512) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.embed = backbone.get_input_embeddings()
        self.h = int(backbone.config.hidden_size)
        self.latent_dim = latent_dim
        self.n_latents = n_latents
        eos_id = int(tokenizer.eos_token_id or 0)
        self.head = EmitHead(self.h, latent_dim, n_latents, dropout, eos_id=eos_id)
        self.llm_model = llm_model
        self.max_len = max_len

    # ---- construction ----
    @classmethod
    def from_pretrained(cls, llm_model: str, *, latent_dim: Optional[int] = None, n_latents: int = 1,
                        lora_r: int = 16, dropout: float = 0.0, bf16: bool = False,
                        max_len: int = 512, device: Optional[str] = None) -> "LangSetModel":
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(llm_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        backbone = build_backbone(llm_model, lora_r, dropout, bf16, dev)
        if latent_dim is None:                                   # default: emit in the backbone's own hidden space
            latent_dim = int(backbone.config.hidden_size)
        return cls(backbone, tok, latent_dim, n_latents, llm_model, dropout, max_len).to(dev)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # ---- forward / inference ----
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Read input text, emit the latent. Returns [B, d]."""
        nl = self.head.n_latents
        rev = self.embed(input_ids)
        q = self.head.q.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(rev.dtype)
        emb = torch.cat([rev, q], 1)
        am = torch.cat([attention_mask,
                        torch.ones(input_ids.size(0), nl, device=input_ids.device,
                                   dtype=attention_mask.dtype)], 1)
        hid = self.backbone(inputs_embeds=emb, attention_mask=am, output_hidden_states=True).hidden_states[-1]
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
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)
            z = self(enc["input_ids"], enc["attention_mask"])
            if normalize_embeddings:
                z = F.normalize(z, p=2, dim=-1)
            out.append(z.cpu())
        if was_training:
            self.train()
        emb = torch.cat(out)
        emb = emb[0] if single else emb
        return emb.numpy() if convert_to_numpy else emb

    def emit(self, sentences: Union[str, list[str]], **kw: Any) -> torch.Tensor:
        return self.encode(sentences, convert_to_numpy=False, **kw)  # type: ignore[return-value]

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
            "n_latents": self.head.n_latents, "max_len": self.max_len}))

    @classmethod
    def load(cls, path: Union[str, Path], *, lora_r: int = 16, device: Optional[str] = None) -> "LangSetModel":
        import json
        p = Path(path); cfg = json.loads((p / "config.json").read_text())
        m = cls.from_pretrained(cfg["llm_model"], latent_dim=cfg["latent_dim"], n_latents=cfg.get("n_latents", 1),
                                lora_r=lora_r, max_len=cfg["max_len"], device=device)
        sd = torch.load(p / "langset.pt", map_location=m.device, weights_only=False)
        m.backbone.load_state_dict(sd["lora"], strict=False)
        m.head.load_state_dict(sd["head"])
        m.eval()
        return m
