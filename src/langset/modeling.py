"""LangSetModel: an LLM (LoRA) + a learned emit head that maps input text -> a latent in a bespoke geometry.

The output is Sentence-Transformer-shaped (`encode`, `get_sentence_embedding_dimension`, `as_sentence_transformer`)
so the trained model drops straight into SetFit as a `model_body`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class EmitHead(nn.Module):
    """Learned query tokens whose post-backbone hidden states are read out as the latent (the LLM's vector 'mouth').

    `dropout` is trained ON when >0 so MC-dropout could later serve as an uncertainty signal; it's also plain
    regularization. The head stays fp32 even when the backbone is bf16.
    """

    def __init__(self, h: int, d: int, n_emit: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_emit = n_emit
        self.q = nn.Parameter(torch.randn(n_emit, h) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_emit * h, d)

    def forward(self, hid_emit: torch.Tensor) -> torch.Tensor:           # [B, n_emit, h] -> [B, d]
        flat = self.drop(hid_emit.reshape(hid_emit.size(0), -1).float())
        return F.normalize(self.out(flat), p=2, dim=-1)


def build_backbone(llm_model: str, lora_r: int, dropout: float, bf16: bool, dev: str) -> Any:
    from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]
    dt = torch.bfloat16 if bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        llm_model, torch_dtype=dt, attention_dropout=dropout, attn_implementation="sdpa").to(dev)
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=dropout, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(base, lora)


class LangSetModel(nn.Module):
    """LLM backbone (LoRA) + EmitHead. `from_pretrained` infers the latent dim from the bootstrap encoder, so the
    emitted geometry starts in that encoder's space (the Trainer then EMA-specializes it away)."""

    def __init__(self, backbone: Any, tokenizer: Any, latent_dim: int, n_emit: int,
                 llm_model: str, bootstrap_model: str, dropout: float = 0.0, max_len: int = 512) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.embed = backbone.get_input_embeddings()
        self.h = int(backbone.config.hidden_size)
        self.latent_dim = latent_dim
        self.head = EmitHead(self.h, latent_dim, n_emit, dropout)
        self.llm_model = llm_model
        self.bootstrap_model = bootstrap_model
        self.max_len = max_len
        self._bootstrap: Any = None

    # ---- construction ----
    @classmethod
    def from_pretrained(cls, llm_model: str, bootstrap_model: str, *, latent_dim: Optional[int] = None,
                        n_emit: int = 4, lora_r: int = 16, dropout: float = 0.0, bf16: bool = False,
                        max_len: int = 512, device: Optional[str] = None) -> "LangSetModel":
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(llm_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if latent_dim is None:                                   # target space dim = bootstrap encoder dim
            _st = SentenceTransformer(bootstrap_model)            # ST renamed the method in 4.x -> support both
            _dim = getattr(_st, "get_embedding_dimension", None) or _st.get_sentence_embedding_dimension
            latent_dim = int(_dim())
        backbone = build_backbone(llm_model, lora_r, dropout, bf16, dev)
        m = cls(backbone, tok, latent_dim, n_emit, llm_model, bootstrap_model, dropout, max_len).to(dev)
        return m

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def bootstrap_encoder(self) -> Any:
        """Lazily-loaded target encoder (used by the Trainer for targets + the beats-bootstrap baseline)."""
        if self._bootstrap is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            self._bootstrap = SentenceTransformer(self.bootstrap_model, device=str(self.device))
        return self._bootstrap

    # ---- forward / inference ----
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        rev = self.embed(input_ids)
        q = self.head.q.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(rev.dtype)
        emb = torch.cat([rev, q], 1)
        am = torch.cat([attention_mask,
                        torch.ones(input_ids.size(0), self.head.n_emit, device=input_ids.device,
                                   dtype=attention_mask.dtype)], 1)
        hid = self.backbone(inputs_embeds=emb, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        return self.head(hid[:, -self.head.n_emit:, :])

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

    # ---- persistence (LoRA + head + config; backbone/bootstrap rebuilt from ids) ----
    def save_pretrained(self, path: Union[str, Path]) -> None:
        import json
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save({"head": self.head.state_dict(),
                    "lora": {k: v.cpu() for k, v in self.backbone.state_dict().items() if "lora" in k}},
                   p / "langset.pt")
        (p / "config.json").write_text(json.dumps({
            "llm_model": self.llm_model, "bootstrap_model": self.bootstrap_model,
            "latent_dim": self.latent_dim, "n_emit": self.head.n_emit, "max_len": self.max_len}))

    @classmethod
    def load(cls, path: Union[str, Path], *, lora_r: int = 16, device: Optional[str] = None) -> "LangSetModel":
        import json
        p = Path(path); cfg = json.loads((p / "config.json").read_text())
        m = cls.from_pretrained(cfg["llm_model"], cfg["bootstrap_model"], latent_dim=cfg["latent_dim"],
                                n_emit=cfg["n_emit"], lora_r=lora_r, max_len=cfg["max_len"], device=device)
        sd = torch.load(p / "langset.pt", map_location=m.device, weights_only=False)
        m.backbone.load_state_dict(sd["lora"], strict=False)
        m.head.load_state_dict(sd["head"])
        m.eval()
        return m
