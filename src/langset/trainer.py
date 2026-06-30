"""Trainer: bootstrap the target geometry from the bootstrap encoder, contrastively fit the LLM emitter, and
EMA-specialize the geometry away from the seed. Selects/early-stops on held-out geometry (see selection.py),
never on training loss.

Dataset rows: `input_text`, `target_text`, plus optional geometry-label columns (EVAL-ONLY probes).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from langset import selection
from langset.modeling import LangSetModel
from langset.training_args import TrainingArguments


def _columns(dataset: Any) -> dict[str, list[Any]]:
    if hasattr(dataset, "column_names"):                       # datasets.Dataset
        return {c: list(dataset[c]) for c in dataset.column_names}
    rows = list(dataset)                                       # list[dict]
    return {k: [r[k] for r in rows] for k in rows[0]}


class Trainer:
    def __init__(self, model: LangSetModel, args: TrainingArguments, train_dataset: Any,
                 eval_dataset: Optional[Any] = None, column_mapping: Optional[dict[str, str]] = None,
                 label_columns: Optional[list[str]] = None) -> None:
        self.model = model
        self.args = args
        cols = _columns(train_dataset)
        cmap = column_mapping or {}
        inv = {v: k for k, v in cmap.items()}                 # user-col -> canonical
        get = lambda canon: cols[inv.get(canon, canon)]       # noqa: E731
        self.input_text = [str(x) for x in get("input_text")]
        self.target_text = [str(x) for x in get("target_text")]
        used = {inv.get("input_text", "input_text"), inv.get("target_text", "target_text")}
        lc = label_columns if label_columns is not None else [c for c in cols if c not in used]
        self.label_cols = {c: [str(x) for x in cols[c]] for c in lc}
        if args.verbose:
            print(f"[langset] {len(self.input_text)} rows | label columns (eval-only): {list(self.label_cols)}",
                  flush=True)

    def train(self) -> LangSetModel:
        a, m = self.args, self.model
        dev = m.device
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)

        # 1. bootstrap targets = bootstrap_encoder(target_text)  (the seed geometry)
        tgt_np = m.bootstrap_encoder.encode(self.target_text, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        tgt = torch.tensor(tgt_np, device=dev)
        n = len(self.input_text)
        perm = rng.permutation(n)
        n_val = max(4, int(n * a.val_frac))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        tok = m.tokenizer
        enc_in = tok(self.input_text, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        ids = enc_in["input_ids"].to(dev)
        mask = enc_in["attention_mask"].to(dev)

        ema = tgt.clone()                                     # EMA self-target (the "specialize" step)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr)
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
                target = ema[idx] if a.ema else tgt[idx]
                logits = (pred @ target.t()) / a.tau          # InfoNCE, in-batch negatives
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))
                if a.lam_anchor > 0:                          # pull toward the frozen bootstrap (0 = fully specialize)
                    loss = loss + a.lam_anchor * (1 - (pred * tgt[idx]).sum(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if a.ema:
                    with torch.no_grad():
                        upd = a.ema_m * ema[idx] + (1 - a.ema_m) * pred.detach()
                        ema[idx] = F.normalize(upd, p=2, dim=-1)
                tot += float(loss.detach()); nb += 1

            if ep % a.eval_every:
                continue
            # 2. validate in the CURRENT geometry: input-view vs target-view co-location (never vs the seed)
            vi = self.input_text and [self.input_text[j] for j in val_idx]
            vt = [self.target_text[j] for j in val_idx]
            emit_in = m.encode(vi, normalize_embeddings=True)
            emit_tg = m.encode(vt, normalize_embeddings=True)
            labels_val = {k: [v[j] for j in val_idx] for k, v in self.label_cols.items()}
            metrics = selection.evaluate(emit_in, emit_tg, labels_val, tgt_np[val_idx], a.select)
            if a.verbose:
                extra = f" purity_mean={metrics.get('purity_mean'):.3f} beats={metrics.get('beats_bootstrap')}" \
                    if "purity_mean" in metrics else ""
                print(f"ep{ep:02d} loss={tot/nb:.3f} mrr={metrics['mrr']:.3f} collapse={metrics['collapse']:.3f}"
                      f"{extra} score={metrics['score']:.3f} [{metrics['select_mode']}]", flush=True)
            if run is not None:
                run.log({"loss": tot / nb, **metrics, "epoch": ep})

            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_state = {"head": {k: v.detach().cpu().clone() for k, v in m.head.state_dict().items()},
                              "lora": {k: v.detach().cpu().clone()
                                       for k, v in m.backbone.state_dict().items() if "lora" in k}}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    if a.verbose:
                        print(f"[langset] early stop at ep{ep} (best score {best_score:.3f})", flush=True)
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
            print(f"[langset] done. best score={best_score:.3f} -> {a.output_dir}", flush=True)
        return m
