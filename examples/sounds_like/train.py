# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "peft", "sentence-transformers", "datasets", "numpy"]
# ///
"""sounds-like, the langset way: an album review -> a "how it sounds" latent, few-shot. The trained model is a
SentenceTransformer body (see setfit_compose.py).

Training rows are reconstructed from PUBLIC data (this repo ships no review text) — see prepare.py:
  input_text  = public Pitchfork review text  (mediumrarely/pitchfork-reviews)
  target_text = our sonic fingerprint         (Hanno-Labs/sounds-like-fingerprints)

  uv run examples/sounds_like/train.py                                       # full (GPU recommended)
  uv run examples/sounds_like/train.py --limit 200 --epochs 5 --device cpu   # quick smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # for prepare.py
sys.path.insert(0, str(HERE.parent.parent / "src"))
from langset import LangSetModel, Trainer, TrainingArguments  # noqa: E402
from prepare import joined_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out", type=str, default=str(HERE / "langset-music"))
    args = ap.parse_args()

    rows = joined_rows(args.limit)
    print(f"[sounds-like] {len(rows)} rows (public reviews x our fingerprints)", flush=True)
    model = LangSetModel.from_pretrained(args.llm, lora_r=16, max_len=args.max_len, device=(args.device or None))
    targs = TrainingArguments(epochs=args.epochs, batch_size=32, max_len=args.max_len, output_dir=args.out)
    Trainer(model, targs, train_dataset=rows).train()
    emb = model.as_sentence_transformer().encode(["detuned sludgy doom at a glacial tempo", "bubblegum synthpop"])
    print(f"[setfit-shape] as_sentence_transformer().encode -> {emb.shape}")


if __name__ == "__main__":
    main()
