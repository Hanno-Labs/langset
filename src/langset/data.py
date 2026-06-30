"""Dataset contract for langset.

A training example is a row with:
  - `input_text`   : the text you'll have at inference (e.g. a name, a query, a review)
  - `target_text`  : a description of the SAME item that defines where it should land (embedded by the
                     bootstrap encoder to seed the geometry). Point this at something `input_text` can't
                     trivially regenerate, or you're just distilling a text encoder.
  - any number of optional label columns : EVAL-ONLY geometry probes (kNN-purity at validation). They never
                     enter training — training on them would collapse the geometry onto the labels.

Pass a `datasets.Dataset` or a `list[dict]` to `Trainer`; use `column_mapping` to rename your columns onto
`input_text` / `target_text`.
"""
from __future__ import annotations

from typing import Any, Optional


def from_records(records: list[dict[str, Any]], input_key: str, target_key: str,
                 label_keys: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Convenience: project a list of dicts onto the langset contract."""
    label_keys = label_keys or []
    out = []
    for r in records:
        row = {"input_text": str(r[input_key]), "target_text": str(r[target_key])}
        for k in label_keys:
            row[k] = r.get(k, "unknown")
        out.append(row)
    return out
