"""Trainer: fit the LLM emitter so `emit(input_text)` lands where `emit(target_text)` does — a native
self-contrastive objective (both views in the model's own space, in-batch negatives). The target text DEFINES
the geometry. Two light aux terms keep it grounded and spread; selection is collapse-aware.

Dataset rows: `input_text` (what you have at inference) + `target_text` (a description of the same item that
defines where it should land). Pass a `datasets.Dataset` or `list[dict]`; use `column_mapping` to rename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from langset import selection
from langset.modeling import LangSetModel
from langset.training_args import TrainingArguments

_RECON_K = 8            # soft-prompt tokens the latent expands into for the recon decoder
_RECON_MAXLEN = 128     # target_text tokens the recon aux reconstructs
_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4   # collapse below this isn't penalized; above it, selection is tanked


def _columns(dataset: Any) -> dict[str, list[Any]]:
    if hasattr(dataset, "column_names"):                       # datasets.Dataset
        return {c: list(dataset[c]) for c in dataset.column_names}
    rows = list(dataset)                                       # list[dict]
    return {k: [r[k] for r in rows] for k in rows[0]}


class Trainer:
    def __init__(self, model: LangSetModel, args: TrainingArguments, train_dataset: Any,
                 eval_dataset: Optional[Any] = None, column_mapping: Optional[dict[str, str]] = None) -> None:
        self.model = model
        self.args = args
        cols = _columns(train_dataset)
        inv = {v: k for k, v in (column_mapping or {}).items()}   # user-col -> canonical
        get = lambda canon: cols[inv.get(canon, canon)]           # type: ignore[index]  # noqa: E731
        self.input_text = [str(x) for x in get("input_text")]
        self.target_text = [str(x) for x in get("target_text")]
        if args.verbose:
            print(f"[langset] {len(self.input_text)} rows", flush=True)

    def train(self) -> LangSetModel:
        a, m = self.args, self.model
        dev = m.device
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        tok = m.tokenizer

        def tok_to(texts: list[str], mx: int) -> tuple[torch.Tensor, torch.Tensor]:
            e = tok(texts, padding=True, truncation=True, max_length=mx, return_tensors="pt")
            return e["input_ids"].to(dev), e["attention_mask"].to(dev)

        ids, mask = tok_to(self.input_text, a.max_len)            # input view
        t2_ids, t2_mask = tok_to(self.target_text, a.max_len)     # target view (self-contrastive target)
        tr_ids, tr_mask = tok_to(self.target_text, _RECON_MAXLEN)  # target tokens for the recon aux

        n = len(self.input_text)
        perm = rng.permutation(n)
        n_val = max(4, int(n * a.val_frac))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        # recon aux: latent -> K soft tokens -> backbone decodes target_text (token CE). Grounds the latent in the
        # text. `connector` is TRAINING-ONLY scaffolding (not saved; inference just emits the latent).
        hsz, vsz = m.h, int(m.backbone.config.vocab_size)
        connector = torch.nn.Linear(m.latent_dim, _RECON_K * hsz).to(dev)

        def recon_loss(latent: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            ti, tm = tr_ids[rows], tr_mask[rows]
            temb = m.embed(ti)
            soft = connector(latent).view(latent.size(0), _RECON_K, hsz).to(temb.dtype)
            seq = torch.cat([soft, temb], dim=1)
            am = torch.cat([torch.ones(latent.size(0), _RECON_K, device=dev, dtype=tm.dtype), tm], dim=1)
            logits = m.backbone(inputs_embeds=seq, attention_mask=am).logits
            pred_lg = logits[:, _RECON_K - 1:_RECON_K - 1 + ti.size(1), :].float()
            return F.cross_entropy(pred_lg.reshape(-1, vsz), ti.masked_fill(tm == 0, -100).reshape(-1),
                                   ignore_index=-100)

        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad] + list(connector.parameters()),
                                lr=a.lr)
        run = None
        if a.report_to == "wandb":
            import wandb  # type: ignore[import-untyped]
            run = wandb.init(project=a.wandb_project, config=vars(a))

        best_score, best_state, no_improve = -1e9, None, 0
        for ep in range(a.epochs):
            m.train()
            order = tr_idx[rng.permutation(len(tr_idx))]
            tot = nb = 0.0
            for i in range(0, len(order), a.batch_size):
                idx = torch.tensor(order[i:i + a.batch_size], device=dev)
                pred = m(ids[idx], mask[idx])
                target = m(t2_ids[idx], t2_mask[idx])            # self-contrastive: emit(target_text), same space
                logits = (pred @ target.t()) / a.tau            # in-batch negatives force separation (no collapse)
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))          # primary
                loss = loss + a.lam_recon * recon_loss(pred, idx)                            # aux: grounding
                if a.lam_uniform > 0 and len(idx) > 1:                                       # aux: uniformity
                    sq = torch.pdist(F.normalize(pred, p=2, dim=-1), p=2).pow(2)
                    loss = loss + a.lam_uniform * sq.mul(-2.0).exp().mean().log()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1

            if ep % a.eval_every:
                continue
            # validate in the CURRENT geometry: input-view vs target-view retrieval + collapse + held-out recon.
            emit_in = np.asarray(m.encode([self.input_text[j] for j in val_idx], normalize_embeddings=True))
            emit_tg = np.asarray(m.encode([self.target_text[j] for j in val_idx], normalize_embeddings=True))
            mrr = selection.retrieval_mrr(emit_in, emit_tg)["mrr"]
            collapse = selection.collapse_score(emit_in)
            with torch.no_grad():
                rv, tot_v = 0.0, 0
                for s in range(0, len(val_idx), a.batch_size):
                    vb = torch.tensor(val_idx[s:s + a.batch_size], device=dev)
                    rv += float(recon_loss(m(ids[vb], mask[vb]), vb)) * len(vb); tot_v += len(vb)
                recon_val = rv / tot_v
            # recon_val is teacher-forced -> blind to collapse; hard-penalize high collapse so a collapsed epoch
            # can never win.
            sel_score = -recon_val - _COLLAPSE_PENALTY * max(0.0, collapse - _COLLAPSE_FLOOR)
            if a.verbose:
                print(f"ep{ep:02d} loss={tot/nb:.3f} mrr={mrr:.3f} collapse={collapse:.3f} "
                      f"recon_val={recon_val:.3f} sel={sel_score:.3f}", flush=True)
            if run is not None:
                run.log({"loss": tot / nb, "mrr": mrr, "collapse": collapse, "recon_val": recon_val,
                         "sel_score": sel_score, "epoch": ep})

            if sel_score > best_score:
                best_score = sel_score
                best_state = {"head": {k: v.detach().cpu().clone() for k, v in m.head.state_dict().items()},
                              "lora": {k: v.detach().cpu().clone()
                                       for k, v in m.backbone.state_dict().items() if "lora" in k}}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    if a.verbose:
                        print(f"[langset] early stop at ep{ep} (best {best_score:.3f})", flush=True)
                    break

        if best_state is not None:                            # restore best
            m.head.load_state_dict(best_state["head"])
            m.backbone.load_state_dict(best_state["lora"], strict=False)
        m.eval()
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(a.output_dir)
        if run is not None:
            run.finish()
        if a.verbose:
            print(f"[langset] done. best={best_score:.3f} -> {a.output_dir}", flush=True)
        return m
