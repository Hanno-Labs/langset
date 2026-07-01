"""Dataset contract for langset.

A training example is a row with:
  - `input_text`   : the text you'll have at inference (e.g. a name, a query, a review)
  - `target_text`  : a description of the SAME item that DEFINES where it should land. The geometry is whatever
                     these descriptions describe — point them at the axis you care about. Point them at something
                     `input_text` can't trivially regenerate, or you're just distilling a text encoder.

Pass a `datasets.Dataset` or a `list[dict]` to `Trainer`; use `column_mapping` to rename your columns onto
`input_text` / `target_text`.
"""
from __future__ import annotations

from typing import Any


def from_records(records: list[dict[str, Any]], input_key: str, target_key: str) -> list[dict[str, Any]]:
    """Convenience: project a list of dicts onto the langset contract."""
    return [{"input_text": str(r[input_key]), "target_text": str(r[target_key])} for r in records]
