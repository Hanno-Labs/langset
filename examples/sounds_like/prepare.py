"""Reconstruct the sounds-like training rows from PUBLIC sources — this repo ships no review text.

We join, on artist+album:
  - public Pitchfork review text   -> `input_text`   (mediumrarely/pitchfork-reviews, ~99% coverage, full text)
  - our LLM sonic fingerprints     -> `target_text`  (Hanno-Labs/sounds-like-fingerprints)

The fingerprint is what DEFINES the geometry; the review is what the model reads at inference.
"""
from __future__ import annotations

import re

REVIEWS = "mediumrarely/pitchfork-reviews"            # public review text (input_text)
FINGERPRINTS = "Hanno-Labs/sounds-like-fingerprints"  # our sonic fingerprints (target_text)


def _key(artist: str, album: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(artist).lower()) + "|" + re.sub(r"[^a-z0-9]", "", str(album).lower())


def joined_rows(limit: int = 0) -> list[dict]:
    """Join public reviews to our fingerprints -> langset training rows (input_text, target_text, +metadata)."""
    from datasets import load_dataset  # type: ignore[import-untyped]
    reviews: dict[str, str] = {}
    for r in load_dataset(REVIEWS, split="train"):
        text = str(r.get("review_text") or "")
        if text.strip():
            reviews[_key(r["artist"], r["album"])] = text
    rows: list[dict] = []
    for r in load_dataset(FINGERPRINTS, split="train"):
        rev = reviews.get(_key(r["artist"], r["album"]))
        if rev and r.get("sounds_like"):
            rows.append({"input_text": rev, "target_text": r["sounds_like"],
                         "artist": r["artist"], "album": r["album"], "genre": r["genre"]})
    return rows[:limit] if limit else rows
