"""Masked self-prediction (JEPA-style) data construction — langset's generic JEPA mode.

langset already trains the JEPA way on the *target* side: the default `EMATwinTarget` is a stop-grad EMA copy
of the model that supplies the target latents (BYOL/JEPA teacher, no collapse). What made a langset encoder
JEPA-vs-not was never a flag — it was WHAT you put in `target_text`:

  * target_text = an external LABEL (e.g. "the legal moves here")  -> contrastive-to-label; the encoder keeps
    only label-relevant structure and DISCARDS the rest (this is how our chess affordance encoder collapsed
    near-identical positions — one move barely changes the label).
  * target_text = a HELD-OUT PART of the SAME text, predicted from the VISIBLE part -> real JEPA: no label,
    the encoder is FORCED to represent what it can't see, so fine detail survives.

This module gives you the second one GENERICALLY. You hand it whole texts and a `Masker` (the algorithm that
decides what to hide); it returns langset rows `{input_text: visible, target_text: full-or-hidden}`. Train
those with the normal (single-latent, default EMATwinTarget) `Trainer` and you have a JEPA encoder for ANY
text domain — prose, structured records, board renders, code — by swapping the masker.

    # PREFERRED: hand the Trainer RAW text; it masks fresh every epoch. No pre-built masked data.
    Trainer(model, TrainingArguments(), [{"text": t} for t in texts]).train()   # auto-masks (word @0.15)
    # pick the algorithm / ratio: TrainingArguments(masker="span", mask_ratio=0.2)
    # custom masker (e.g. protect chess move-numbers): TrainingArguments(masker=TokenMasker(0.15, protect=...))

The Trainer path re-masks EVERY epoch (unlimited diversity from a small corpus) — you never enumerate masked
copies. `build_masked` / `build_masked_pairs` below still exist for materializing a fixed masked dataset when
you want one, but for training just give `text`.

A Masker is any callable `(text: str, rng: random.Random) -> (visible_text, hidden_text)`.
"""

from __future__ import annotations

import random
from typing import Callable, Protocol

Masker = Callable[[str, random.Random], tuple[str, str]]


class _MaskerBase(Protocol):
    def __call__(self, text: str, rng: random.Random) -> tuple[str, str]: ...


class SpanMasker:
    """Hide a CONTIGUOUS run of whitespace tokens — the general-text default (a phrase, a clause). The encoder
    must infer the missing span from both sides, so it has to represent surrounding meaning, not just keywords."""

    def __init__(self, ratio: float = 0.15, sentinel: str = "[MASK]") -> None:
        self.ratio, self.sentinel = ratio, sentinel

    def __call__(self, text: str, rng: random.Random) -> tuple[str, str]:
        toks = text.split()
        if len(toks) < 2:
            return text, ""
        k = max(1, min(len(toks) - 1, round(len(toks) * self.ratio)))
        start = rng.randint(0, len(toks) - k)
        hidden = toks[start : start + k]
        visible = toks[:start] + [self.sentinel] + toks[start + k :]
        return " ".join(visible), " ".join(hidden)


class TokenMasker:
    """Hide RANDOM scattered tokens (BERT/data2vec flavor). More diffuse pressure than a single span.

    `protect` — an optional predicate `(token) -> bool`; a protected token is NEVER masked and never
    counts toward the maskable pool. Use it to keep scaffolding visible while hiding only content, e.g.
    for chess movetext `protect=lambda t: t.endswith(".")` masks moves but keeps the `1.`/`2.` numbering
    so positions stay anchored (mask a random % of the actual MOVES, both players)."""

    def __init__(
        self,
        ratio: float = 0.15,
        sentinel: str = "[MASK]",
        protect: Callable[[str], bool] | None = None,
    ) -> None:
        self.ratio, self.sentinel, self.protect = ratio, sentinel, protect

    def __call__(self, text: str, rng: random.Random) -> tuple[str, str]:
        toks = text.split()
        maskable = [i for i, t in enumerate(toks) if self.protect is None or not self.protect(t)]
        if len(maskable) < 1 or len(toks) < 2:
            return text, ""
        k = max(1, min(len(maskable), round(len(maskable) * self.ratio)))
        idx = set(rng.sample(maskable, k))
        hidden = [toks[i] for i in sorted(idx)]
        visible = [self.sentinel if i in idx else t for i, t in enumerate(toks)]
        return " ".join(visible), " ".join(hidden)


class FieldMasker:
    """Hide whole FIELDS of a delimited record (structured text / tabular / board renders). Splits on `sep`;
    hides a fraction of fields. With `kv_sep` set, keeps each field's KEY and hides only its VALUE
    (e.g. cells `e1:K` -> `e1:[MASK]`), so the model must infer WHAT is at a KNOWN slot — the sharp form of
    "represent the fine detail." This is the masker for a chess board render (cells) or any key:value record."""

    def __init__(
        self,
        sep: str = " ",
        ratio: float = 0.2,
        sentinel: str = "[MASK]",
        kv_sep: str | None = None,
    ) -> None:
        self.sep, self.ratio, self.sentinel, self.kv_sep = sep, ratio, sentinel, kv_sep

    def __call__(self, text: str, rng: random.Random) -> tuple[str, str]:
        items = text.split(self.sep)
        if len(items) < 2:
            return text, ""
        k = max(1, min(len(items) - 1, round(len(items) * self.ratio)))
        idx = set(rng.sample(range(len(items)), k))
        hidden: list[str] = []
        visible: list[str] = []
        for i, it in enumerate(items):
            if i not in idx:
                visible.append(it)
            elif self.kv_sep is not None and self.kv_sep in it:
                key, _val = it.split(self.kv_sep, 1)
                visible.append(f"{key}{self.kv_sep}{self.sentinel}")
                hidden.append(it)
            else:
                visible.append(self.sentinel)
                hidden.append(it)
        return self.sep.join(visible), " ".join(hidden)


def resolve_masker(spec: Masker | str | None, ratio: float = 0.15) -> Masker:
    """Turn a masker SPEC into a Masker. A callable passes through; a string picks a default algorithm; None
    is the reasonable default ('word' = scattered-word masking). Used by the Trainer so callers can write
    `masker="span"` / `masker=None` and never build a masker by hand."""
    if callable(spec):
        return spec
    if spec in (None, "word", "token"):
        return TokenMasker(ratio)
    if spec == "span":
        return SpanMasker(ratio)
    if spec == "field":
        return FieldMasker(ratio=ratio)
    raise ValueError(
        f"unknown masker spec {spec!r}; use 'word', 'span', 'field', a callable, or None"
    )


def mask_view(texts: list[str], masker: Masker, rng: random.Random) -> list[str]:
    """One FRESH masked view per text — the VISIBLE side; the target is the full text itself. This is what the
    Trainer calls at the start of EVERY epoch so masks are never reused (unlimited diversity from raw text).
    A text where the masker hides nothing (too short) passes through unmasked (that row just carries no signal
    this epoch)."""
    out: list[str] = []
    for t in texts:
        visible, hidden = masker(t, rng)
        out.append(visible if hidden.strip() else t)
    return out


def build_masked(
    texts: list[str], masker: Masker, views: int = 1, seed: int = 0, target_mode: str = "full"
) -> list[dict[str, str]]:
    """Whole texts -> langset JEPA rows. Each text yields `views` masked pairs with different random masks
    (mask diversity, like JEPA re-masking). target_mode: 'full' = predict the WHOLE text's latent from the
    masked view (masked-view -> full-view alignment); 'hidden' = predict only the withheld content's latent."""
    if target_mode not in ("full", "hidden"):
        raise ValueError("target_mode must be 'full' or 'hidden'")
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for t in texts:
        for _ in range(views):
            visible, hidden = masker(t, rng)
            if not visible.strip() or not hidden.strip() or visible.strip() == t.strip():
                continue  # skip degenerate (nothing hidden)
            out.append(
                {"input_text": visible, "target_text": t if target_mode == "full" else hidden}
            )
    return out


def build_masked_pairs(
    pairs: list[tuple[str, str]],
    masker: Masker,
    views: int = 1,
    seed: int = 0,
    sep: str = " ",
    mask_region: str = "target",
    target_mode: str = "full",
) -> list[dict[str, str]]:
    """FUSE (input, target) then mask — the "state + facets you want to predict" JEPA.

    Each pair is `(input, target)`: `input` is the state of the world you ALWAYS have; `target` is the
    ADDITIONAL facets you want the latent to be able to recover. We MERGE them (`input + sep + target`),
    run the masker on the merged text, and predict the masked content — so one encoder is forced to hold
    BOTH the state and the extra facets, with no external label.

    mask_region:
      * 'target' (default) — mask ONLY inside the target span; the whole input stays visible. This is the
        pointed form: "given the full state, reconstruct the withheld facets." (The encoder learns to
        predict the facets FROM the state.)
      * 'all' — mask anywhere across the merged text (symmetric masked self-prediction over state+facets).
    target_mode: 'full' = predict the whole merged (input+sep+target) latent; 'hidden' = predict only the
    withheld content's latent.
    """
    if mask_region not in ("target", "all"):
        raise ValueError("mask_region must be 'target' or 'all'")
    if target_mode not in ("full", "hidden"):
        raise ValueError("target_mode must be 'full' or 'hidden'")
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for inp, tgt in pairs:
        merged = f"{inp}{sep}{tgt}"
        for _ in range(views):
            if mask_region == "all":
                visible, hidden = masker(merged, rng)
            else:  # mask only the target span; keep input whole
                vis_t, hidden = masker(tgt, rng)
                visible = f"{inp}{sep}{vis_t}"
            if not visible.strip() or not hidden.strip() or visible.strip() == merged.strip():
                continue  # skip degenerate (nothing hidden)
            out.append(
                {"input_text": visible, "target_text": merged if target_mode == "full" else hidden}
            )
    return out
