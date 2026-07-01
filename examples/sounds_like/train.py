# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "peft", "sentence-transformers", "numpy"]
# ///
"""sounds-like, the langset way: an album review -> a "how it sounds" latent, few-shot. The trained model is a
SentenceTransformer body (see setfit_compose.py).

  input_text  = album review
  target_text = a sonic description that DEFINES the geometry ("how it sounds")

  uv run examples/sounds_like/train.py                                       # full (GPU recommended)
  uv run examples/sounds_like/train.py --limit 200 --epochs 5 --device cpu   # quick smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from langset import LangSetModel, Trainer, TrainingArguments  # noqa: E402


def load_rows(prepared: str, limit: int = 0) -> list[dict]:
    recs = json.loads(Path(prepared).read_text())
    if limit:
        recs = recs[:limit]
    return [{"input_text": r["review"], "target_text": r["fingerprint"]} for r in recs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=str, default=str(HERE / "data/prepared.json"))
    ap.add_argument("--llm", type=str, default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out", type=str, default=str(HERE / "langset-music"))
    args = ap.parse_args()

    rows = load_rows(args.prepared, args.limit)
    print(f"[sounds-like] {len(rows)} rows", flush=True)
    model = LangSetModel.from_pretrained(args.llm, lora_r=16, max_len=args.max_len, device=(args.device or None))
    targs = TrainingArguments(epochs=args.epochs, batch_size=32, max_len=args.max_len, output_dir=args.out)
    Trainer(model, targs, train_dataset=rows).train()
    emb = model.as_sentence_transformer().encode(["detuned sludgy doom at a glacial tempo", "bubblegum synthpop"])
    print(f"[setfit-shape] as_sentence_transformer().encode -> {emb.shape}")


if __name__ == "__main__":
    main()
