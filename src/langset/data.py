"""Dataset contracts and small helpers for preparing langset training data.

The core training contract is a row with two views of the same item:

- ``input_text`` is the text available at inference time, such as a name,
  query, or review.
- ``target_text`` is a description of that same item that defines the geometry
  the model should learn. Choose a target that exposes the property of
  interest and cannot be trivially regenerated from ``input_text``; otherwise
  the objective mostly distills a text encoder.

Pass a ``datasets.Dataset`` or ``list[dict]`` to :class:`langset.Trainer`. Use
``column_mapping`` when an existing dataset uses different column names, or
use :func:`from_records` to project arbitrary records onto the basic contract.
"""

from __future__ import annotations

from typing import Any


def from_records(
    records: list[dict[str, Any]], input_key: str, target_key: str
) -> list[dict[str, Any]]:
    """Convenience: project a list of dicts onto the langset contract."""
    return [{"input_text": str(r[input_key]), "target_text": str(r[target_key])} for r in records]
