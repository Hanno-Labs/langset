"""The langset training loop, in isolation — the LLM -> embedding path with nothing else attached.

This is a SINGLE self-contained file that distills langset's single-latent (embedding) path down to its
essentials so you can read the whole thing top to bottom:

    text  ->  LoRA'd LLM backbone  ->  learned query token  ->  last hidden  ->  projection head  ->  latent
    and the self-contrastive loop that shapes that latent: emit(input_text) should land where emit(target_text) does.

It is a faithful reduction of `src/langset/modeling.py` + `src/langset/trainer.py`, with every peripheral feature
STRIPPED OUT so the core is visible. Removed (each lives in the real trainer): the multi-latent / FSQ world-model
path, the recon + text-replay ([LEARN]) auxiliaries, preempt-resume checkpointing, wandb, hard negatives,
false-negative masking, the frozen-pool fast path, Per-Layer-Embedding (Gemma) handling, view-fusing, and the
dependency-injected strategy seams. What remains is the primary objective and the eval/selection that guards it.

Minimal external dependencies: torch, transformers, peft, numpy. (No datasets, sentence-transformers, or setfit.)

Run:
    python examples/isolated_training_loop.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ------------------------------------------------------------------------------------------------------------
# Config — the handful of knobs the core loop actually reads (cf. langset.training_args.TrainingArguments).
# ------------------------------------------------------------------------------------------------------------
@dataclass
class Config:
    llm_model: str = "HuggingFaceTB/SmolLM2-135M"   # any HF causal LM
    lora_r: int = 16
    max_len: int = 64
    epochs: int = 20
    batch_size: int = 8
    lr: float = 5e-4
    tau: float = 0.07                # contrastive temperature
    lam_uniform: float = 0.1         # light uniformity aux (spread latents on the sphere)
    val_frac: float = 0.2
    seed: int = 0


# ------------------------------------------------------------------------------------------------------------
# The emit head — the LLM's "vector mouth". A learned query token is appended to the input; after the backbone
# runs, that query's final hidden state is projected to the latent and L2-normalized. (cf. EmitHead in
# modeling.py — the multi_latent / FSQ branches are dropped here; this is the single continuous vector.)
# ------------------------------------------------------------------------------------------------------------
class EmitHead(nn.Module):
    def __init__(self, h: int, d: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, h) * 0.02)   # one learned query token
        self.out_proj = nn.Linear(h, d)                   # hidden -> latent

    def forward(self, hid_query: torch.Tensor) -> torch.Tensor:   # [B, h] -> [B, d]
        return F.normalize(self.out_proj(hid_query.float()), p=2, dim=-1)


# ------------------------------------------------------------------------------------------------------------
# The model — LLM backbone (LoRA) + emit head. `forward(input_ids, mask)` IS the LLM -> embedding path.
# ------------------------------------------------------------------------------------------------------------
class LangSetModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(cfg.llm_model)
        # LoRA the attention + MLP projections; the base stays frozen. Only the adapters + head train.
        # (The real build_backbone also strips the lm_head to avoid the full-vocab OOM on big models and reads
        # `last_hidden_state` directly; here we just keep output_hidden_states=True and read the final layer,
        # which is simpler and fine for small models.)
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        self.backbone = get_peft_model(base, lora)
        self.embed = self.backbone.get_input_embeddings()
        self.h = base.config.hidden_size
        self.head = EmitHead(self.h, self.h)              # emit in the backbone's own hidden space (latent_dim = h)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def _final_hidden(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            output_hidden_states=True)
        return out.hidden_states[-1]                      # [B, S, h]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Read input text, emit ONE latent. [B, S] -> [B, d]. This is the whole LLM -> embedding path."""
        tok_emb = self.embed(input_ids)                                          # real text tokens
        q = self.head.q.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(tok_emb.dtype)
        seq = torch.cat([tok_emb, q], dim=1)                                     # append the query token
        am = torch.cat([attention_mask, attention_mask.new_ones(input_ids.size(0), 1)], dim=1)
        hid = self._final_hidden(seq, am)
        return self.head(hid[:, -1, :])                                          # the query's final hidden -> latent

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Sentence-Transformer-shaped convenience: texts -> normalized latents as numpy."""
        was_training = self.training
        self.eval()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.cfg.max_len,
                             return_tensors="pt").to(self.device)
        z = self(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy()
        if was_training:
            self.train()
        return z


# ------------------------------------------------------------------------------------------------------------
# Selection metrics — NEVER select on training loss (a contrastive objective can minimize it by COLLAPSING the
# geometry). Score held-out input-view <-> target-view retrieval, penalized by collapse. (cf. langset.selection.)
# ------------------------------------------------------------------------------------------------------------
def retrieval_mrr(pred: np.ndarray, bank: np.ndarray) -> float:
    """Each pred[i] should retrieve bank[i] among the val set (rank-based, scale-free, collapse-sensitive)."""
    order = np.argsort(-(pred @ bank.T), axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) for i in range(len(pred))])
    return float((1.0 / (ranks + 1)).mean())


def collapse_score(x: np.ndarray) -> float:
    """Mean off-diagonal cosine of the emissions. ->1 means collapsed; a healthy space sits low."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    s = xn @ xn.T
    np.fill_diagonal(s, np.nan)
    return float(np.nanmean(s))


_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4    # collapse below this isn't penalized; above it, selection is tanked


# ------------------------------------------------------------------------------------------------------------
# The training loop, in isolation.
#
# One objective: emit(input_text) should land where emit(target_text) lands. Both views live in the model's OWN
# space (self-contrastive) and the target_text DEFINES the geometry. In-batch negatives force separation so the
# space can't collapse; a light uniformity term keeps latents spread on the sphere. Select on held-out MRR.
# ------------------------------------------------------------------------------------------------------------
def train(model: LangSetModel, data: list[dict], cfg: Config) -> LangSetModel:
    dev = model.device
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tok = model.tokenizer

    input_text = [str(r["input_text"]) for r in data]
    target_text = [str(r["target_text"]) for r in data]

    def tok_to(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        e = tok(texts, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    ids, mask = tok_to(input_text)          # input view
    t2_ids, t2_mask = tok_to(target_text)   # target view (defines where each input should land)

    # held-out split for selection
    perm = rng.permutation(len(input_text))
    n_val = max(4, int(len(perm) * cfg.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    best_score, best_state = -1e9, None

    for ep in range(cfg.epochs):
        model.train()
        order = tr_idx[rng.permutation(len(tr_idx))]
        tot = nb = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = torch.tensor(order[i:i + cfg.batch_size], device=dev)

            pred = model(ids[idx], mask[idx])          # emit(input_text)  [B, d]
            target = model(t2_ids[idx], t2_mask[idx])  # emit(target_text) [B, d]

            # InfoNCE: each row's own target is the positive; every other target in the batch is a negative.
            logits = (pred @ target.t()) / cfg.tau     # [B, B]
            B = len(idx)
            loss = F.cross_entropy(logits, torch.arange(B, device=dev))          # primary: contrastive

            if cfg.lam_uniform > 0 and B > 1:                                    # aux: uniformity (spread)
                sq = torch.pdist(F.normalize(pred, p=2, dim=-1), p=2).pow(2)
                loss = loss + cfg.lam_uniform * sq.mul(-2.0).exp().mean().log()

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1

        # ---- validate in the CURRENT geometry: held-out input-view vs target-view retrieval + collapse ----
        emit_in = model.encode([input_text[j] for j in val_idx])
        emit_tg = model.encode([target_text[j] for j in val_idx])
        mrr = retrieval_mrr(emit_in, emit_tg)
        collapse = collapse_score(emit_in)
        sel_score = mrr - _COLLAPSE_PENALTY * max(0.0, collapse - _COLLAPSE_FLOOR)
        print(f"ep{ep:02d} loss={tot / nb:.3f} mrr={mrr:.3f} collapse={collapse:.3f} sel={sel_score:.3f}",
              flush=True)

        if sel_score > best_score:      # keep the best-so-far weights (head + LoRA), restore at the end
            best_score = sel_score
            best_state = {"head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
                          "lora": {k: v.detach().cpu().clone()
                                   for k, v in model.backbone.state_dict().items() if "lora" in k}}

    if best_state is not None:
        model.head.load_state_dict(best_state["head"])
        model.backbone.load_state_dict(best_state["lora"], strict=False)
    model.eval()
    print(f"done. best sel={best_score:.3f}", flush=True)
    return model


# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # rows: input_text (what you read at inference) -> target_text (the description that DEFINES the geometry)
    data = [
        {"input_text": "Sleep — Dopesmoker", "target_text": "glacial detuned doom-metal riffs, sludgy and hypnotic"},
        {"input_text": "Aphex Twin — Selected Ambient Works", "target_text": "warm melodic ambient techno, analog pads"},
        {"input_text": "Burial — Untrue", "target_text": "crackly nocturnal UK garage, pitched vocal ghosts"},
        {"input_text": "Electric Wizard — Dopethrone", "target_text": "down-tuned fuzz doom, occult and crushing"},
    ] * 8  # tiny toy set

    cfg = Config(epochs=20, batch_size=8)
    model = LangSetModel(cfg)
    train(model, data, cfg)

    z = model.encode(["Bongripper — Satan Worshipping Doom"])   # input text -> latent
    print("latent shape:", z.shape)
