"""Utilities for masked self-prediction datasets and training views.

A masker is a callable with the following interface:

    (text: str, rng: random.Random) -> (visible_text, hidden_text)

The built-in maskers hide contiguous spans, scattered tokens, or fields in a delimited record. `Trainer` can
apply a masker to a raw `text` column at the start of each epoch while using the original text as the target:

    rows = [{"text": text} for text in texts]
    args = TrainingArguments(masker="span", mask_ratio=0.2)
    Trainer(model, args, rows).train()

When no masker is specified for raw-text training, scattered-token masking is used with the configured mask
ratio.

`build_masked()` and `build_masked_pairs()` materialize deterministic masked datasets for callers that need fixed
views instead of per-epoch masking.
"""

from __future__ import annotations

import random
from typing import Callable, Protocol, cast

Masker = Callable[[str, random.Random], tuple[str, str]]


class _MaskerBase(Protocol):
    def __call__(self, text: str, rng: random.Random) -> tuple[str, str]: ...


class SpanMasker:
    """Replace one contiguous span of whitespace-delimited tokens with a sentinel.

    At least one token is hidden and at least one original token remains visible. Texts containing fewer than two
    tokens are returned unchanged with an empty hidden string.
    """

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
    """Replace randomly selected whitespace-delimited tokens with a sentinel.

    `ratio` is applied to the tokens eligible for masking. When `protect` is provided, tokens for which
    `protect(token)` returns true remain visible and are excluded from the maskable population.

    The hidden text contains the selected original tokens in source order. Texts with no maskable tokens, or fewer
    than two total tokens, are returned unchanged with an empty hidden string.
    """

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
    """Mask randomly selected fields in a delimited record.

    Fields are split with `sep`, and `ratio` determines how many are selected. Without `kv_sep`, each selected field
    is replaced by `sentinel`. With `kv_sep`, a selected key/value field keeps its key and replaces its value; for
    example, `e1:K` becomes `e1:[MASK]`.

    The hidden text contains each selected field in its original form. Inputs with fewer than two fields are returned
    unchanged with an empty hidden string.
    """

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
    """Resolve a masker specification.

    Args:
        spec: A masker callable, `"word"`, `"token"`, `"span"`, `"field"`, or `None`. `None`, `"word"`, and
            `"token"` select `TokenMasker`.
        ratio: Masking ratio passed to a built-in masker.

    Returns:
        The supplied callable or a newly constructed built-in masker.

    Raises:
        ValueError: If `spec` is an unrecognized string.
    """
    if callable(spec):
        return cast("Masker", spec)
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
    """Generate one masked visible view for each text.

    If a masker returns an empty or whitespace-only hidden string, the original text is preserved in the result.

    Args:
        texts: Source texts to mask.
        masker: Masking callable.
        rng: Random generator controlling mask selection.

    Returns:
        One visible text for each source text, in the same order.
    """
    out: list[str] = []
    for t in texts:
        visible, hidden = masker(t, rng)
        out.append(visible if hidden.strip() else t)
    return out


def build_masked(
    texts: list[str], masker: Masker, views: int = 1, seed: int = 0, target_mode: str = "full"
) -> list[dict[str, str]]:
    """Build fixed masked input/target rows from whole texts.

    Each source text is masked `views` times using a deterministic random generator initialized from `seed`.
    Degenerate views that hide no content are omitted.

    Args:
        texts: Source texts.
        masker: Masking callable.
        views: Number of masking attempts per source text.
        seed: Seed for deterministic mask selection.
        target_mode: `"full"` uses the original text as `target_text`; `"hidden"` uses only the text returned as
            hidden by the masker.

    Returns:
        Dictionaries containing `input_text` and `target_text`.

    Raises:
        ValueError: If `target_mode` is not `"full"` or `"hidden"`.
    """
    if target_mode not in ("full", "hidden"):
        raise ValueError("target_mode must be 'full' or 'hidden'")
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for t in texts:
        for _ in range(views):
            visible, hidden = masker(t, rng)
            if not visible.strip() or not hidden.strip() or visible.strip() == t.strip():
                continue  # Skip views that hide no usable content.
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
    """Build fixed masked rows from `(input, target)` text pairs.

    The two parts are joined as `input + sep + target`. With `mask_region="target"`, only the target part is passed
    to the masker and the input part remains visible. With `mask_region="all"`, the masker is applied to the joined
    text.

    Args:
        pairs: Source `(input, target)` pairs.
        masker: Masking callable.
        views: Number of masking attempts per pair.
        seed: Seed for deterministic mask selection.
        sep: Separator inserted between the two parts.
        mask_region: `"target"` to mask only the target part, or `"all"` to mask the joined text.
        target_mode: `"full"` uses the unmasked joined text as `target_text`; `"hidden"` uses only the content
            returned as hidden by the masker.

    Returns:
        Dictionaries containing `input_text` and `target_text`. Degenerate views that hide no usable content are
        omitted.

    Raises:
        ValueError: If `mask_region` or `target_mode` is invalid.
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
            else:  # Mask only the target part.
                vis_t, hidden = masker(tgt, rng)
                visible = f"{inp}{sep}{vis_t}"
            if not visible.strip() or not hidden.strip() or visible.strip() == merged.strip():
                continue  # Skip views that hide no usable content.
            out.append(
                {"input_text": visible, "target_text": merged if target_mode == "full" else hidden}
            )
    return out
