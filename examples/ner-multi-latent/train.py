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

Reports entity-level F1 (a GROUND-TRUTH read of multi-latent quality), avg emitted count, decode_acc (FSQ
reconstruction fidelity — should stay high and STABLE, no drift), and tf_acc (teacher-forced digit accuracy).

  uv run examples/ner-multi-latent/train.py                                    # full (GPU recommended)
  uv run examples/ner-multi-latent/train.py --limit 300 --epochs 3 --device cpu --no-wandb   # quick smoke
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # for prepare.py
sys.path.insert(0, str(HERE.parent.parent / "src"))
from langset import LangSetModel  # noqa: E402
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


def run(args: argparse.Namespace, commit_fn: Optional[Callable[[], None]] = None) -> None:
    torch.manual_seed(args.seed)                             # LoRA init + dropout — else runs aren't comparable
    train_rows = prepare.rows(args.limit, "train")
    val_rows = prepare.rows(max(args.limit // 4, 200) if args.limit else 800, "validation")
    print(f"[ner] {len(train_rows)} train / {len(val_rows)} val sentences (>=1 entity)", flush=True)

    model = LangSetModel.from_pretrained(args.llm, lora_r=16, max_len=args.max_len, multi_latent=True,
                                         latent_dim=(args.latent_dim or None),   # None => backbone hidden size
                                         fsq_dim=args.fsq_dim, fsq_levels=args.fsq_levels,
                                         device=(args.device or None))
    dev = model.device
    tok = model.tokenizer
    d = model.latent_dim
    head = model.head
    fsq_dim, fsq_levels = head.fsq_dim, head.fsq_levels
    stop_idx = fsq_levels                                    # STOP is the extra class in dim-0's softmax
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # EMA target twin: a stop-grad, slowly-updated copy supplies the target latents (BYOL/JEPA). Multi-latent
    # emission MANDATES it — with the online model emitting its own target (both sides move) the geometry
    # collapsed on a bad init (f1 0.05, mean-cosine 0.85). It is NOT a flag: single-latent langset core doesn't
    # need it, but any >1-latent trainer turns it on automatically so nobody foot-guns themselves by forgetting.
    ema_model: Any = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    ema_o = [po for po in model.parameters() if po.requires_grad]
    ema_e = [pe for pe, po in zip(ema_model.parameters(), model.parameters()) if po.requires_grad]

    def ema_update() -> None:
        with torch.no_grad():
            torch._foreach_mul_(ema_e, args.ema_m)
            torch._foreach_add_(ema_e, ema_o, alpha=1.0 - args.ema_m)

    run = None
    if args.wandb:
        import wandb  # type: ignore[import-untyped]
        run = wandb.init(project="langset-ner-multi-latent", config=vars(args))

    def emit_texts(texts: list[str], grad: bool, mdl: Optional[Any] = None) -> torch.Tensor:
        """Single-latent emission of each text -> [N, d] (normalized). The target latent; from the EMA twin
        (stop-grad) when one is passed, else from the online model."""
        m2 = mdl if mdl is not None else model
        e = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return m2(e["input_ids"], e["attention_mask"])

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
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
        zb = F.normalize(emit_texts(bank_texts, grad=False, mdl=ema_model).float(), dim=-1)   # target space
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
            ft = emit_texts([x for lst in el for x in lst], grad=False, mdl=ema_model)
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
        for i in range(0, len(val_rows), args.batch_size):                       # BATCH the rollout (was 1-at-a-
            chunk = val_rows[i:i + args.batch_size]                              # time -> GPU-starved on a big GPU)
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
        model.train()
        return {"f1": f1, "precision": prec, "recall": rec, "decode_acc": decode_acc, "tf_acc": tf_acc,
                "avg_emitted": len(emitted) / max(len(val_rows), 1)}

    rng = torch.Generator().manual_seed(args.seed)
    best_f1 = -1.0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_rows), generator=rng).tolist()
        tot = nb = 0.0
        for i in range(0, len(order), args.batch_size):
            batch = [train_rows[k] for k in order[i:i + args.batch_size]]
            seeds = [r["input_text"] for r in batch]
            se = tok(seeds, padding=True, truncation=True, max_length=args.max_len,
                     padding_side="left", return_tensors="pt").to(dev)      # left-pad: hid[s_len-1] = last real token
            ent_lists = [[_ent_text(s, t) for (s, t) in r["entities"][:args.max_entities]] for r in batch]
            lmax = max(len(x) for x in ent_lists)
            flat_texts = [txt for lst in ent_lists for txt in lst]
            flat_tgt = emit_texts(flat_texts, grad=False, mdl=ema_model)          # [ΣL, d] stop-grad EMA targets
            b = len(batch)
            target_lat = torch.zeros(b, lmax, d, device=dev)
            valid = torch.zeros(b, lmax, dtype=torch.bool, device=dev)
            lens: list[int] = []
            k = 0
            for r, lst in enumerate(ent_lists):
                nl = len(lst); lens.append(nl)
                target_lat[r, :nl] = flat_tgt[k:k + nl]; valid[r, :nl] = True
                k += nl
            # token-native FSQ: predict each entity's per-dim digits, then a STOP folded into dim-0's softmax.
            dim_lg, stop_lg, digits, recon = model.rollout_train_codebook(se["input_ids"], se["attention_mask"],
                                                                          target_lat, args.tau)
            dim0 = torch.cat([dim_lg[:, :, 0, :], stop_lg], -1)                   # [b, lmax+1, L+1] — digit-0 + STOP
            lab0 = torch.full((b, lmax + 1), -100, dtype=torch.long, device=dev)
            lab_rest = torch.full((b, lmax, fsq_dim - 1), -100, dtype=torch.long, device=dev)
            for r, nl in enumerate(lens):
                lab0[r, :nl] = digits[r, :nl, 0]; lab0[r, nl] = stop_idx          # emit digit-0, then STOP after last
                lab_rest[r, :nl] = digits[r, :nl, 1:]
            loss_stop = F.cross_entropy(dim0.reshape(-1, fsq_levels + 1), lab0.reshape(-1), ignore_index=-100)
            loss_dims = F.cross_entropy(dim_lg[:, :lmax, 1:, :].reshape(-1, fsq_levels),
                                        lab_rest.reshape(-1), ignore_index=-100)
            recon_loss = (1.0 - F.cosine_similarity(recon[valid], target_lat[valid], dim=-1)).mean()  # trains down/up
            loss = loss_stop + loss_dims + recon_loss                            # one objective — no lam_* knobs
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update()                                                         # EMA twin tracks the online model
            tot += float(loss.detach()); nb += 1

        m = evaluate()
        tag = ""
        if m["f1"] > best_f1:                                    # save-on-best; commit so it's live-downloadable
            best_f1 = m["f1"]
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            if commit_fn is not None:
                commit_fn()
            tag = f"  <- best, saved to {args.out}"
        print(f"ep{ep:02d} loss={tot/nb:.3f} f1={m['f1']:.3f} (p={m['precision']:.3f} r={m['recall']:.3f}) "
              f"tf_acc={m['tf_acc']:.3f} decode_acc={m['decode_acc']:.3f} emit={m['avg_emitted']:.2f}{tag}",
              flush=True)
        if run is not None:
            run.log({"loss": tot / nb, **m, "best_f1": best_f1, "epoch": ep})

    demo = model.rollout("Angela Merkel met Tim Cook at Apple headquarters in Cupertino .", max_steps=args.max_steps)
    print(f"[demo] emitted {demo.shape[0]} entity latents for a 3-entity sentence -> {tuple(demo.shape)}", flush=True)
    if run is not None:
        run.finish()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
