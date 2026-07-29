# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "peft", "datasets", "numpy", "wandb"]
# ///
"""ner-multi-latent: text in -> a VARIABLE-LENGTH sequence of entity latents out, terminated by a learned STOP.

This is langset's MULTI-LATENT training demo (`multi_latent=True`). A sentence goes in; the model
autoregressively emits one latent per entity (ordered by appearance) via `rollout_train_codebook`, feeding each
emitted CODE back into its own hidden stream, and stops when it emits the STOP code. NER is deliberately
NON-PREDICTIVE — every entity is present in the input — so this isolates "does multi-latent emission train
cleanly" from any forecasting question.

Multi-latent emission is TOKEN-NATIVE via FSQ (finite scalar quantization): the stop-grad EMA-twin's target
embedding is projected into a FIXED grid (fsq_dim dims x fsq_levels levels), and the model predicts each dim's
digit plus a STOP folded into dim-0's softmax. The grid is not learned and cannot drift/collapse (unlike a VQ
codebook), so it stays stable while the encoder + EMA twin keep SPECIALIZING; near-continuous precision comes
from the grid resolution. The objective is per-dim cross-entropy + a cosine reconstruction loss (trains the
down/up projections) — no lam_* knobs, no stop threshold. Count/termination is free (STOP wins dim-0's softmax).

Training is DELEGATED to the langset library `Trainer` (multi_latent path): we hand it `{input_text,
target_texts:[...]}` rows and it runs the exact EMA-twin + FSQ + learned-STOP loop and selects on retrieval MRR.
This example then runs its own GROUND-TRUTH entity-level F1 read on the held-out validation split (the NER-
specific quality signal the generic library eval can't express), plus decode_acc (FSQ fidelity) and tf_acc.

  uv run examples/ner-multi-latent/train.py                                    # full (GPU recommended)
  uv run examples/ner-multi-latent/train.py --limit 300 --epochs 3 --device cpu --no-wandb   # quick smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # for prepare.py
sys.path.insert(0, str(HERE.parent.parent / "src"))
from langset import LangSetModel, Trainer, TrainingArguments  # noqa: E402
import prepare  # noqa: E402


def _ent_text(surface: str, typ: str) -> str:
    return f"{typ}: {surface}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--max_entities", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=16)     # rollout cap at eval
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.07)                              # code-classification temperature
    ap.add_argument("--latent_dim", type=int, default=0)                             # 0 => backbone hidden size
    ap.add_argument("--fsq_dim", type=int, default=128)                              # FSQ bottleneck dims (internal)
    ap.add_argument("--fsq_levels", type=int, default=8)                            # FSQ levels per dim (internal)
    ap.add_argument("--ema_m", type=float, default=0.99)                            # EMA-target momentum (internal knob)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out", type=str, default=str(HERE / "langset-ner"))
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    return ap


@torch.no_grad()
def evaluate_f1(model: LangSetModel, val_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, float]:
    """GROUND-TRUTH entity-level F1 on the held-out validation split. The library selects on retrieval MRR during
    training; this is the NER-specific read the generic eval can't express. Uses the trained model itself (post-
    specialization) as the target-space encoder for the entity bank."""
    dev = model.device
    tok = model.tokenizer
    head = model.head
    d = model.latent_dim
    model.eval()

    def emit(texts: list[str]) -> torch.Tensor:
        e = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        return F.normalize(model(e["input_ids"], e["attention_mask"]).float(), dim=-1)

    # gold entity bank: unique "TYPE: surface" over val -> emit -> [Nbank, d], tagged (surface_lower, type)
    bank_texts: list[str] = []; bank_tags: list[tuple[str, str]] = []; seen: set[str] = set()
    gold: list[set[tuple[str, str]]] = []
    for r in val_rows:
        gp: set[tuple[str, str]] = set()
        for (s, t) in r["entities"][:args.max_entities]:
            gp.add((s.lower(), t))
            et = _ent_text(s, t)
            if et not in seen:
                seen.add(et); bank_texts.append(et); bank_tags.append((s.lower(), t))
        gold.append(gp)
    zb = emit(bank_texts)                                                     # [Nbank, d] target space
    # DIAGNOSTIC decode_acc: entity -> FSQ encode -> reconstruction -> nearest bank entity. FSQ fidelity;
    # with a FIXED grid this should stay high and STABLE across training (no codebook drift to decay it).
    _, recon = head.encode(zb)                                               # [Nbank, d] reconstructions
    near = (F.normalize(recon, dim=-1) @ zb.t()).argmax(-1)                   # nearest bank entity per recon
    decode_acc = float((near == torch.arange(len(bank_tags), device=dev)).float().mean())
    # DIAGNOSTIC tf_acc: teacher-forced per-dim digit accuracy (predictor quality, isolated from free rollout).
    tf_ok = tf_tot = 0
    for i in range(0, len(val_rows), args.batch_size):
        vb = val_rows[i:i + args.batch_size]
        se = tok([r["input_text"] for r in vb], padding=True, truncation=True, max_length=args.max_len,
                 padding_side="left", return_tensors="pt").to(dev)
        el = [[_ent_text(s, t) for (s, t) in r["entities"][:args.max_entities]] for r in vb]
        lm = max(len(x) for x in el)
        ft = emit([x for lst in el for x in lst])
        tl = torch.zeros(len(vb), lm, d, device=dev); kk = 0
        for r, lst in enumerate(el):
            tl[r, :len(lst)] = ft[kk:kk + len(lst)]; kk += len(lst)
        dim_lg, _sl, digits, _rc = model.rollout_train_codebook(se["input_ids"], se["attention_mask"], tl, args.tau)
        pr = dim_lg[:, :lm].argmax(-1)                                        # [b, lm, fsq_dim]
        for r, lst in enumerate(el):
            tf_ok += float((pr[r, :len(lst)] == digits[r, :len(lst)]).float().mean(-1).sum())  # mean over dims
            tf_tot += len(lst)
    tf_acc = tf_ok / max(tf_tot, 1)
    tp = fp = fn = 0
    emitted: list[torch.Tensor] = []
    demo: list[str] = []
    for i in range(0, len(val_rows), args.batch_size):                       # BATCH the rollout
        chunk = val_rows[i:i + args.batch_size]
        lats, lens = model.rollout([r["input_text"] for r in chunk],
                                   max_steps=args.max_steps, return_lengths=True)  # [B, Lmax, d], [B]
        for k in range(len(chunk)):
            gp = gold[i + k]
            pred_set: set[tuple[str, str]] = set()
            for j in range(int(lens[k])):
                v = F.normalize(lats[k, j].float(), dim=-1)
                pred_set.add(bank_tags[int((zb @ v).argmax())])
                emitted.append(v.cpu())
            tp += len(pred_set & gp); fp += len(pred_set - gp); fn += len(gp - pred_set)
            if len(demo) < 4:                                                # live qualitative decode sample
                demo.append(f"    {chunk[k]['input_text'][:52]!r}\n      gold={sorted(f'{t}:{s}' for s, t in gp)}"
                            f"\n      pred={sorted(f'{t}:{s}' for s, t in pred_set)}")
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    for line in demo:
        print(line, flush=True)
    return {"f1": f1, "precision": prec, "recall": rec, "decode_acc": decode_acc, "tf_acc": tf_acc,
            "avg_emitted": len(emitted) / max(len(val_rows), 1)}


def run(args: argparse.Namespace, commit_fn: Optional[Callable[[], None]] = None) -> None:
    torch.manual_seed(args.seed)                             # LoRA init + dropout — else runs aren't comparable
    train_rows = prepare.rows(args.limit, "train")
    val_rows = prepare.rows(max(args.limit // 4, 200) if args.limit else 800, "validation")
    print(f"[ner] {len(train_rows)} train / {len(val_rows)} val sentences (>=1 entity)", flush=True)

    model = LangSetModel.from_pretrained(args.llm, lora_r=16, max_len=args.max_len, multi_latent=True,
                                         latent_dim=(args.latent_dim or None),   # None => backbone hidden size
                                         fsq_dim=args.fsq_dim, fsq_levels=args.fsq_levels,
                                         device=(args.device or None))

    # DOGFOOD: hand the multi-latent rows to the langset library Trainer — it runs the EMA-twin + FSQ + learned-STOP
    # loop and selects on retrieval MRR, so this example never re-copies the training loop.
    rows = [{"input_text": r["input_text"],
             "target_texts": [_ent_text(s, t) for (s, t) in r["entities"][:args.max_entities]]}
            for r in train_rows]
    targs = TrainingArguments(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, tau=args.tau, max_len=args.max_len,
        ema_m=args.ema_m, max_target_items=args.max_entities, max_steps=args.max_steps, val_frac=0.1,
        output_dir=args.out, report_to=("wandb" if args.wandb else None),
        wandb_project="langset-ner-multi-latent", seed=args.seed, verbose=True)
    model = Trainer(model, targs, rows, on_checkpoint=commit_fn).train()

    # GROUND-TRUTH entity-F1 on the held-out validation split (the NER-specific quality read).
    m = evaluate_f1(model, val_rows, args)
    print(f"[ner] held-out entity-F1={m['f1']:.3f} (p={m['precision']:.3f} r={m['recall']:.3f}) "
          f"tf_acc={m['tf_acc']:.3f} decode_acc={m['decode_acc']:.3f} emit={m['avg_emitted']:.2f}", flush=True)

    demo = model.rollout("Angela Merkel met Tim Cook at Apple headquarters in Cupertino .", max_steps=args.max_steps)
    print(f"[demo] emitted {demo.shape[0]} entity latents for a 3-entity sentence -> {tuple(demo.shape)}", flush=True)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
