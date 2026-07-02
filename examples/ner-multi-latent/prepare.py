"""CoNLL-2003 -> ordered-entity rows for langset multi-latent training.

Each row is one sentence plus the list of its entities IN ORDER OF APPEARANCE:
    {"input_text": "EU rejects German call to boycott British lamb .",
     "entities":   [("EU", "ORG"), ("German", "MISC"), ("British", "MISC")]}

NER is deliberately a NON-predictive multi-latent task: every entity is present in the input, so this
isolates "does multi-latent emission train" from "does forecasting train" — the point of the demo.
"""
from __future__ import annotations

from typing import Any


def _spans(tokens: list[str], tag_ids: list[int], names: list[str]) -> list[tuple[str, str]]:
    """IOB2 tag ids -> ordered (surface, type) spans."""
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_type = ""
    for tok, tid in zip(tokens, tag_ids):
        tag = names[tid]                                   # e.g. "B-ORG", "I-ORG", "O"
        if tag == "O":
            if cur:
                out.append((" ".join(cur), cur_type)); cur = []
            continue
        pre, typ = tag.split("-", 1)
        if pre == "B" or typ != cur_type:
            if cur:
                out.append((" ".join(cur), cur_type))
            cur = [tok]; cur_type = typ
        else:                                              # I- continuing the same type
            cur.append(tok)
    if cur:
        out.append((" ".join(cur), cur_type))
    return out


def rows(limit: int = 0, split: str = "train", min_entities: int = 1) -> list[dict[str, Any]]:
    """Load CoNLL-2003 and return ordered-entity rows (sentences with >= min_entities entities)."""
    from datasets import load_dataset  # type: ignore[import-untyped]
    ds = None
    for name in ("tomaarsen/conll2003", "eriktks/conll2003", "conll2003"):   # parquet mirrors first
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception:  # noqa: BLE001 — try the next mirror
            continue
    if ds is None:
        raise RuntimeError("could not load CoNLL-2003 from any known HF id")
    names = ds.features["ner_tags"].feature.names            # ['O','B-PER','I-PER',...]
    out: list[dict[str, Any]] = []
    for ex in ds:
        ents = _spans(ex["tokens"], ex["ner_tags"], names)
        if len(ents) < min_entities:
            continue
        out.append({"input_text": " ".join(ex["tokens"]), "entities": ents})
        if limit and len(out) >= limit:
            break
    return out
