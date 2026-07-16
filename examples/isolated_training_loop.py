"""The langset training loop, in isolation — the LLM -> embedding path with nothing else attached.

This is a SINGLE self-contained file that distills langset's single-latent (embedding) path down to its
essentials so you can read the whole thing top to bottom:

    text  ->  LoRA'd LLM backbone  ->  learned query token  ->  last hidden  ->  projection head  ->  embedding
    and the self-contrastive loop that shapes it: emit(input_text) should land where emit(target_text) does.

It is a faithful reduction of `src/langset/modeling.py` + `src/langset/trainer.py`, with every peripheral feature
STRIPPED OUT so the core is visible. Removed (each lives in the real trainer): the multi-latent / FSQ world-model
path, the recon + text-replay ([LEARN]) auxiliaries, preempt-resume checkpointing, wandb, hard negatives,
false-negative masking, the frozen-pool fast path, Per-Layer-Embedding (Gemma) handling, view-fusing, and the
dependency-injected strategy seams. What remains is the primary objective and the eval/selection that guards it.

Minimal external dependencies: torch, transformers, peft, numpy. (No datasets, sentence-transformers, or setfit.)

--------------------------------------------------------------------------------------------------------------
VOCABULARY (for an engineer new to ML) — the terms used below, once:
  - tensor            a multi-dimensional array (like a numpy array) that also tracks gradients for training.
  - embedding /       a fixed-length list of numbers (here 576) that represents a piece of text. "Close"
    latent / vector   vectors mean "similar meaning". We keep them length-1, so a dot product = cosine similarity.
  - token / tokenizer the tokenizer chops text into integer IDs (sub-word pieces); the model only sees integers.
  - forward pass      run data through the model to get an output (here: text -> embedding).
  - loss              one number measuring how wrong the output is. Training = making this number smaller.
  - gradient /        backprop computes, for every trainable number, which direction changes the loss. The
    backprop         optimizer then nudges each number that way. Repeat -> the model improves.
  - LoRA              a cheap way to fine-tune: freeze the giant pretrained model, train tiny add-on matrices.
  - contrastive       train by "pull matching pairs together, push non-matching pairs apart" (no labels needed).
  - collapse          the failure mode where the model makes ALL embeddings identical (loss looks low, but the
                      embeddings are useless). We measure it and refuse to select a collapsed checkpoint.
--------------------------------------------------------------------------------------------------------------

Run:
    python examples/isolated_training_loop.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F   # F = the functional API (F.normalize, F.cross_entropy, ...); standard alias
from torch import nn              # nn = neural-network building blocks (nn.Linear, nn.Module, ...); standard alias


# ------------------------------------------------------------------------------------------------------------
# Config — the handful of knobs the core loop actually reads (cf. langset.training_args.TrainingArguments).
# ------------------------------------------------------------------------------------------------------------
@dataclass
class Config:
    llm_model: str = "HuggingFaceTB/SmolLM2-135M"   # any HF causal LM; this one is small enough to run on CPU
    lora_r: int = 16                 # LoRA "rank": how much capacity the trainable add-on gets (bigger = more)
    max_len: int = 64                # truncate/pad every text to this many tokens
    epochs: int = 20                 # how many times we sweep the whole training set
    batch_size: int = 8              # how many rows we process per optimizer step
    learning_rate: float = 5e-4      # how big each parameter nudge is
    tau: float = 0.07                # contrastive "temperature" (see the loss): smaller = pickier/sharper
    lam_uniform: float = 0.1         # weight of the light "spread the embeddings out" auxiliary term
    val_frac: float = 0.2            # fraction of rows held out (never trained on) to score checkpoints
    seed: int = 0                    # fix randomness so runs are reproducible


# ------------------------------------------------------------------------------------------------------------
# The emit head — the LLM's "vector mouth". A learned query token is appended to the input; after the backbone
# runs, that query's final hidden state is projected to the embedding and length-normalized. (cf. EmitHead in
# modeling.py — the multi_latent / FSQ branches are dropped here; this is the single continuous vector.)
# ------------------------------------------------------------------------------------------------------------
class EmitHead(nn.Module):
    """The small trainable layer that reads ONE embedding vector out of the big (mostly frozen) LLM."""

    def __init__(self, hidden_size: int, latent_dim: int) -> None:
        super().__init__()
        # A learned "query token": a single trainable vector, the same width as a real token embedding. We
        # append it to the input; the LLM attends over the text and writes a summary into this token's slot.
        # nn.Parameter = a tensor the optimizer is allowed to train. randn(...) * 0.02 = small random start.
        self.query_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        # A plain linear layer (output = input @ W + b) mapping the LLM's hidden width -> our embedding width.
        self.out_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:   # [batch, hidden] -> [batch, latent]
        # Project to the embedding width, then L2-normalize so every embedding has length 1 (lives on the unit
        # sphere). Once vectors are length-1, a dot product between two of them IS their cosine similarity.
        return F.normalize(self.out_proj(query_hidden.float()), p=2, dim=-1)


# ------------------------------------------------------------------------------------------------------------
# The model — LLM backbone (LoRA) + emit head. `forward(input_ids, mask)` IS the LLM -> embedding path.
# ------------------------------------------------------------------------------------------------------------
class LangSetModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        # The tokenizer converts text <-> integer token IDs the model understands.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
        if self.tokenizer.pad_token_id is None:                    # some models have no dedicated PAD token;
            self.tokenizer.pad_token = self.tokenizer.eos_token    # reuse end-of-sequence for padding.

        # Download the pretrained LLM. Its weights stay frozen — we never update the base model's parameters.
        base_llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model)
        # LoRA = Low-Rank Adaptation. Rather than fine-tune billions of weights, freeze them and inject small
        # trainable matrices into the named layers. We train ONLY those adapters + our head => cheap, few-shot.
        # target_modules are the attention/MLP projection layers inside each transformer block to adapt.
        lora_config = LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, bias="none",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"])
        self.backbone = get_peft_model(base_llm, lora_config)      # frozen LLM + trainable LoRA adapters
        self.embed = self.backbone.get_input_embeddings()          # the token-ID -> vector lookup table
        self.hidden_size = base_llm.config.hidden_size             # width of the LLM's internal vectors (576 here)
        # Emit in the LLM's own hidden width, so the embedding width == hidden_size.
        self.head = EmitHead(self.hidden_size, self.hidden_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        # (The real build_backbone also strips the LLM's word-prediction head to save memory and reads
        # `last_hidden_state` directly; here we keep output_hidden_states=True and take the final layer,
        # which is simpler and fine for a small model.)

    def _final_hidden(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Run the LLM. We feed it vectors directly (inputs_embeds) rather than token IDs, because our appended
        # query token is a vector with no ID. output_hidden_states=True exposes every layer's activations; we
        # take the last layer. Returned shape: [batch, seq_len, hidden].
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                output_hidden_states=True)
        return outputs.hidden_states[-1]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Read input text, emit ONE embedding. [batch, seq_len] -> [batch, latent]. The whole LLM->embedding path.

        attention_mask has a 1 for every real token and a 0 for padding, so the LLM ignores padded positions.
        """
        batch_size = input_ids.size(0)
        token_embeds = self.embed(input_ids)                       # [batch, seq_len, hidden] — the text tokens
        # Broadcast the single shared query token to every row: [1, hidden] -> [batch, 1, hidden].
        query = self.head.query_token.unsqueeze(0).expand(batch_size, -1, -1).to(token_embeds.dtype)
        seq = torch.cat([token_embeds, query], dim=1)              # append the query token AFTER the text
        # Extend the mask with a 1 for the query position (it's a real slot the LLM should attend to).
        seq_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, 1)], dim=1)
        hidden = self._final_hidden(seq, seq_mask)                 # [batch, seq_len+1, hidden]
        # Take the LAST position — our query token — and turn its hidden state into the embedding.
        return self.head(hidden[:, -1, :])

    @torch.no_grad()                                               # inference only: don't track gradients here
    def encode(self, texts: list[str]) -> np.ndarray:
        """Convenience: a list of strings -> normalized embeddings as a numpy array [n_texts, latent]."""
        was_training = self.training
        self.eval()                                                # switch off training-only behavior (dropout, etc.)
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.cfg.max_len,
                                 return_tensors="pt").to(self.device)
        embeddings = self(encoded["input_ids"], encoded["attention_mask"]).float().cpu().numpy()
        if was_training:
            self.train()                                           # restore whatever mode we were in
        return embeddings


# ------------------------------------------------------------------------------------------------------------
# Selection metrics — NEVER select on training loss (a contrastive objective can minimize it by COLLAPSING the
# geometry). Score held-out input-view <-> target-view retrieval, penalized by collapse. (cf. langset.selection.)
# ------------------------------------------------------------------------------------------------------------
def retrieval_mrr(pred: np.ndarray, bank: np.ndarray) -> float:
    """Retrieval quality on held-out data. For each input embedding pred[i], rank ALL target embeddings by
    similarity; the correct answer is bank[i]. MRR = mean of 1/(rank of the correct target). 1.0 = always #1.
    Rank-based, so it's scale-free and drops toward chance if the space has collapsed."""
    # pred @ bank.T = every input-vs-every-target cosine similarity (vectors are unit length). Shape [n, n].
    similarity = pred @ bank.T
    # argsort(-similarity) orders the targets best-first within each row.
    order = np.argsort(-similarity, axis=1)
    # For row `query_row`, find WHERE the correct target (column query_row) landed => its rank (0 = top).
    ranks = np.array([int(np.where(order[query_row] == query_row)[0][0]) for query_row in range(len(pred))])
    return float((1.0 / (ranks + 1)).mean())


def collapse_score(vectors: np.ndarray) -> float:
    """How bunched-together are the embeddings? Returns the average cosine similarity between DIFFERENT
    embeddings. ~1.0 = collapsed (all pointing the same way = useless); a healthy space sits much lower."""
    unit_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)   # re-normalize (defensive)
    cosine_sims = unit_vectors @ unit_vectors.T                    # all-pairs cosine similarity
    np.fill_diagonal(cosine_sims, np.nan)                          # drop self-similarity (always 1)
    return float(np.nanmean(cosine_sims))


_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4    # collapse below this isn't penalized; above it, selection is tanked


# ------------------------------------------------------------------------------------------------------------
# The training loop, in isolation.
#
# One objective: emit(input_text) should land where emit(target_text) lands. Both views live in the model's OWN
# space (self-contrastive) and the target_text DEFINES the geometry. In-batch negatives force separation so the
# space can't collapse; a light uniformity term keeps embeddings spread on the sphere. Select on held-out MRR.
# ------------------------------------------------------------------------------------------------------------
def train(model: LangSetModel, data: list[dict], cfg: Config) -> LangSetModel:
    device = model.device
    torch.manual_seed(cfg.seed)                    # seed both RNGs so the run is reproducible
    rng = np.random.default_rng(cfg.seed)
    tokenizer = model.tokenizer

    # Pull the two columns out of the dataset rows.
    input_text = [str(row["input_text"]) for row in data]
    target_text = [str(row["target_text"]) for row in data]

    def tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # -> (token_ids [n, seq_len], attention_mask [n, seq_len]). padding=True pads to the longest in the list.
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt")
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    # Pre-tokenize both "views" of every row once. The INPUT view is what you have at inference time; the TARGET
    # view is a description that DEFINES where that input should land in embedding space.
    input_ids, input_mask = tokenize(input_text)
    target_ids, target_mask = tokenize(target_text)

    # Hold out a fraction of rows we never train on — used only to score and select checkpoints.
    shuffled = rng.permutation(len(input_text))
    n_val = max(4, int(len(shuffled) * cfg.val_frac))
    val_idx, train_idx = shuffled[:n_val], shuffled[n_val:]

    # AdamW is the optimizer: it reads each parameter's gradient and nudges the parameter to lower the loss.
    # We hand it ONLY the trainable parameters (LoRA adapters + head); the frozen base has requires_grad=False.
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad],
                                  lr=cfg.learning_rate)
    best_score, best_state = -1e9, None

    for epoch in range(cfg.epochs):                # one epoch = one pass over the training rows
        model.train()                              # enable training-only behavior (dropout, etc.)
        epoch_order = train_idx[rng.permutation(len(train_idx))]     # reshuffle the rows each epoch
        running_loss, num_batches = 0.0, 0.0
        for batch_start in range(0, len(epoch_order), cfg.batch_size):
            batch_idx = torch.tensor(epoch_order[batch_start:batch_start + cfg.batch_size], device=device)

            # Forward pass: turn this batch's input texts and target texts into embeddings.
            pred = model(input_ids[batch_idx], input_mask[batch_idx])       # emit(input_text)  [batch, latent]
            target = model(target_ids[batch_idx], target_mask[batch_idx])   # emit(target_text) [batch, latent]
            batch_len = len(batch_idx)

            # ---- The contrastive objective (InfoNCE), the heart of the loop ----
            # similarity[i, j] = how close input i's embedding is to target j's embedding (unit-vector dot =
            # cosine). Dividing by tau (temperature) sharpens the scores; a smaller tau makes the model pickier.
            similarity = (pred @ target.t()) / cfg.tau                      # [batch, batch]
            # We WANT input i to match its own target i (the diagonal) and NOT the others. That's just a
            # classification problem: "for row i, pick column i." The other batch_len-1 targets are the
            # negatives — free, no mining needed ("in-batch negatives"). cross_entropy pushes the diagonal up
            # and the off-diagonal down. This is what prevents collapse: identical vectors can't win this game.
            labels = torch.arange(batch_len, device=device)                # the correct column for each row = its index
            loss = F.cross_entropy(similarity, labels)                      # primary term

            # Optional light "uniformity" term: actively spread the input embeddings apart on the sphere (a
            # known trick for healthier embedding geometry). It rewards larger average pairwise distance.
            if cfg.lam_uniform > 0 and batch_len > 1:
                pairwise_sq_dist = torch.pdist(F.normalize(pred, p=2, dim=-1), p=2).pow(2)
                loss = loss + cfg.lam_uniform * pairwise_sq_dist.mul(-2.0).exp().mean().log()

            # ---- Backward pass + parameter update ----
            optimizer.zero_grad()      # clear gradients left over from the previous step
            loss.backward()            # backprop: compute d(loss)/d(parameter) for every trainable parameter
            optimizer.step()           # AdamW nudges each parameter a little to reduce the loss
            running_loss += float(loss.detach())
            num_batches += 1

        # ---- End of epoch: score on held-out data and remember the best checkpoint ----
        # Never select on training loss — a collapsed model has low loss but useless embeddings. Instead score
        # real retrieval on the val rows, and subtract a penalty when the space is collapsing.
        val_input_emb = model.encode([input_text[sample_idx] for sample_idx in val_idx])
        val_target_emb = model.encode([target_text[sample_idx] for sample_idx in val_idx])
        mrr = retrieval_mrr(val_input_emb, val_target_emb)
        collapse = collapse_score(val_input_emb)
        sel_score = mrr - _COLLAPSE_PENALTY * max(0.0, collapse - _COLLAPSE_FLOOR)
        print(f"ep{epoch:02d} loss={running_loss / num_batches:.3f} mrr={mrr:.3f} "
              f"collapse={collapse:.3f} sel={sel_score:.3f}", flush=True)

        if sel_score > best_score:     # snapshot the best-scoring weights so far (just the head + LoRA adapters)
            best_score = sel_score
            best_state = {"head": {name: tensor.detach().cpu().clone()
                                   for name, tensor in model.head.state_dict().items()},
                          "lora": {name: tensor.detach().cpu().clone()
                                   for name, tensor in model.backbone.state_dict().items() if "lora" in name}}

    # Restore the best checkpoint (later epochs can regress once collapse creeps back in).
    if best_state is not None:
        model.head.load_state_dict(best_state["head"])
        model.backbone.load_state_dict(best_state["lora"], strict=False)   # strict=False: only LoRA keys are present
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

    embedding = model.encode(["Bongripper — Satan Worshipping Doom"])   # unseen input text -> embedding
    print("embedding shape:", embedding.shape)
