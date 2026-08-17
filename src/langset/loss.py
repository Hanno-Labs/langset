"""Loss functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch
import torch.nn.functional as F
from torch.nn import Linear

if TYPE_CHECKING:
    from langset.modeling import LangSetModel
    from langset.training_args import TrainingArguments


class Loss(Protocol):
    """A loss result that exposes a scalar optimization objective."""

    def to_tensor(self) -> torch.Tensor:
        """Return the scalar objective to weight, combine, and backpropagate."""
        ...


@dataclass
class ScalarLoss:
    """A leaf loss containing one scalar optimization objective."""

    value: torch.Tensor
    """Scalar objective tensor used directly for optimization."""

    def to_tensor(self) -> torch.Tensor:
        """Return this loss's scalar objective."""
        return self.value


ContextT = TypeVar("ContextT", contravariant=True)

LossFn = Callable[[ContextT], Loss]


@dataclass
class TrainingContext:
    """Dependencies shared by training losses."""

    model: LangSetModel
    """The model being trained."""

    args: TrainingArguments
    """Training configuration, including objective weights and temperatures."""


@dataclass
class ReconstructionLossContext(TrainingContext):
    """Inputs to :func:`recon_loss`.

    ``latent`` contains one latent vector per selected row. ``tr_ids`` and
    ``tr_mask`` contain tokenized target text for every dataset row; ``rows``
    selects the targets corresponding to ``latent``.
    """

    latent: torch.Tensor
    """Latents to decode, shaped ``[batch_size, latent_dim]``."""

    rows: torch.Tensor
    """Dataset-row indices selecting the target tokens for this batch."""

    tr_ids: torch.Tensor
    """Token IDs for every row's reconstruction target text."""

    tr_mask: torch.Tensor
    """Attention mask aligned with ``tr_ids``; padding tokens are ignored."""

    connector: Linear
    """Projects each latent into a sequence of synthetic prefix-token embeddings."""


def recon_loss(ctx: ReconstructionLossContext) -> Loss:
    """Decode each latent into its target text using next-token cross-entropy.

    The connector turns a latent into synthetic prefix tokens. The model then
    predicts each following target token. Cross-entropy penalizes the model when
    it assigns little probability to the actual next token, while ignoring
    padding. For example, a latent for ``"The capital of France is Paris"`` is
    trained to help predict the tokens in that target text.

    This auxiliary objective grounds the latent in the target's textual content.
    """

    target_ids, target_mask = ctx.tr_ids[ctx.rows], ctx.tr_mask[ctx.rows]
    """training targets to predict as tokens and their attention mask"""
    recon_k = ctx.connector.out_features // ctx.model.h
    """number of synthetic tokens to generate before target text"""
    target_embeddings = ctx.model.embed(target_ids)
    """Teacher-forced embeddings of the real target tokens appended after the prefix."""
    synthetic_tokens = (
        ctx.connector(ctx.latent)
        .view(ctx.latent.size(0), recon_k, ctx.model.h)
        .to(target_embeddings.dtype)
    )
    """Synthetic prefix embeddings generated from the latent. Prepending them lets target
    token predictions depend on the latent, so reconstruction gradients update it."""
    sequence = torch.cat([synthetic_tokens, target_embeddings], dim=1)
    """Combined sequence of synthetic prefix and target text embeddings"""
    attention_mask = torch.cat(
        [
            torch.ones(
                ctx.latent.size(0), recon_k, device=target_mask.device, dtype=target_mask.dtype
            ),
            target_mask,
        ],
        dim=1,
    )
    """Attention mask combining synthetic prefix and target text attention"""
    backbone_output = ctx.model._run_backbone(sequence, attention_mask, target_ids, recon_k)
    # Shift by one: the final prefix position predicts target_ids[:, 0].
    target_slice = slice(recon_k - 1, recon_k - 1 + target_ids.size(1))
    prediction_logits: torch.Tensor
    """Vocabulary logits for target tokens, supplied by the LM head or derived from hidden states."""

    if (logits := getattr(backbone_output, "logits", None)) is not None:
        prediction_logits = logits[:, target_slice, :].float()
    else:
        target_hidden = ctx.model._last_hidden(backbone_output)[:, target_slice, :]
        prediction_logits = F.linear(target_hidden.float(), ctx.model.embed.weight.float())
    return ScalarLoss(
        F.cross_entropy(
            prediction_logits.reshape(-1, ctx.model.vocab_size),
            target_ids.masked_fill(target_mask == 0, -100).reshape(-1),
            ignore_index=-100,
        )
    )


recon_loss_fn: LossFn[ReconstructionLossContext] = recon_loss
"""The reconstruction loss function."""


@dataclass
class LearnLossContext(TrainingContext):
    """Inputs to :func:`learn_loss` for a batch from the replay pool."""

    pos: torch.Tensor
    """Indices selecting replay examples from the tokenized replay pool."""

    ln_doc_ids: torch.Tensor
    """Left-padded document token IDs for every replay-pool row."""

    ln_doc_mask: torch.Tensor
    """Attention mask aligned with ``ln_doc_ids``."""

    ln_tgt_ids: torch.Tensor
    """Right-padded target token IDs for every replay-pool row."""

    ln_tgt_mask: torch.Tensor
    """Attention mask aligned with ``ln_tgt_ids``; padding tokens are ignored."""


def learn_loss(ctx: LearnLossContext) -> Loss:
    """Rehearse document-to-target knowledge with teacher-forced cross-entropy.

    This is standard next-token training: after reading a document, the model
    predicts its paired target text and is penalized when it assigns low
    probability to the actual next target token. For example, a replay pair of
    document ``"France's capital is"`` and target ``"Paris"`` trains that
    association even when the row is not in the embedding objective.
    """

    document_ids, document_mask = ctx.ln_doc_ids[ctx.pos], ctx.ln_doc_mask[ctx.pos]
    """Left-padded replay documents for this batch and their attention mask."""
    target_ids, target_mask = ctx.ln_tgt_ids[ctx.pos], ctx.ln_tgt_mask[ctx.pos]
    """Right-padded replay targets for this batch and their attention mask."""
    sequence = torch.cat([document_ids, target_ids], dim=1)
    """Teacher-forced document followed by its target tokens."""
    attention_mask = torch.cat([document_mask, target_mask], dim=1)
    """Attention mask aligned with ``sequence``; target padding is excluded."""
    hidden_states = ctx.model._last_hidden(
        ctx.model._run_backbone(ctx.model.embed(sequence), attention_mask, sequence, 0)
    )  # all real tokens -> real_start=0
    """Final hidden state for every document and target token in ``sequence``."""
    document_length = document_ids.size(1)
    """Fixed left-padded document width, which is the target's starting offset."""
    target_hidden = hidden_states[
        :, document_length - 1 : document_length - 1 + target_ids.size(1), :
    ]
    """Shifted hidden states whose outputs predict the corresponding target tokens."""
    prediction_logits = F.linear(target_hidden.float(), ctx.model.embed.weight.float())
    """Vocabulary logits derived from the target-prediction hidden states."""
    return ScalarLoss(
        F.cross_entropy(
            prediction_logits.reshape(-1, ctx.model.vocab_size),
            target_ids.masked_fill(target_mask == 0, -100).reshape(-1),
            ignore_index=-100,
        )
    )


learn_loss_fn: LossFn[LearnLossContext] = learn_loss
"""The learn loss function."""


@dataclass
class SLContext(TrainingContext):
    """Inputs to the single-latent contrastive objective.

    Each row in ``pred`` should be close to the corresponding row in ``target``
    and far from the other target embeddings in the batch. This is not a
    supervised-classification loss: no field here is a class label or a token
    vocabulary logit vector.
    """

    pred: torch.Tensor
    """Input-text embeddings being trained, shaped ``[batch_size, embedding_dim]``."""

    target: torch.Tensor
    """Matching target-text embeddings, shaped ``[batch_size, embedding_dim]``."""

    hn: torch.Tensor | None
    """Optional mined hard-negative target embeddings, one per batch row."""

    idx: torch.Tensor
    """Dataset-row indices corresponding to the batch embeddings."""

    mask_keys: list[frozenset[str]] | None
    """Per-row dataset metadata used to exclude same-key false negatives."""

    hard_neg_text: list[str] | None
    """Per-row mined hard-negative text; an empty string marks no valid negative."""


def sl_loss(
    ctx: SLContext,
) -> Loss:
    """Align paired input and target embeddings with an InfoNCE-style objective.

    Each row of ``pred`` is an input embedding (the anchor). Its target at the
    same row of ``target`` is the positive; other batch targets are negatives.
    Their temperature-scaled dot products are similarity logits—not vocabulary
    logits—and cross-entropy selects the same-row target as the correct
    candidate. This trains each input to be more similar to its paired target
    than to the permitted alternatives.

    Mined hard negatives can add one extra candidate per anchor. Matching
    ``mask_keys`` exclude known false negatives from that anchor's denominator.
    This function prepares those single-latent candidates, positive indices, and
    exclusion mask, then delegates the primary contrastive calculation to
    :func:`info_nce_loss`.

    The contrastive term has an implicit weight of ``1.0``. When
    ``args.lam_uniform > 0`` and the batch contains multiple rows, the returned
    loss also includes ``lam_uniform * uniformity_term``. This auxiliary term
    spreads normalized input embeddings across the unit sphere.
    """

    target_matrix = ctx.target if ctx.hn is None else torch.cat([ctx.target, ctx.hn], dim=0)
    """Candidate target embeddings: batch positives plus optional mined hard negatives."""
    batch_size = len(ctx.idx)
    """Number of input/positive-target pairs in the current batch."""
    neg_mask = torch.zeros(
        batch_size, target_matrix.size(0), dtype=torch.bool, device=ctx.pred.device
    )
    """Mask marking candidate columns to exclude from each contrastive denominator."""
    if ctx.mask_keys is not None:  # in-batch block: drop same-issue false negatives
        batch_mask_keys = [ctx.mask_keys[j] for j in ctx.idx.tolist()]
        """False-negative exclusion keys corresponding to the rows in this batch."""
        for row in range(batch_size):
            if not batch_mask_keys[row]:
                continue
            for column in range(batch_size):
                if row != column and (batch_mask_keys[row] & batch_mask_keys[column]):
                    neg_mask[row, column] = True
    if (
        ctx.hn is not None
    ):  # PER-ANCHOR-ONLY hard neg: anchor i sees ONLY its own mined hard neg (col B+i)
        assert ctx.hard_neg_text is not None
        valid_hard_negatives = [bool(ctx.hard_neg_text[j]) for j in ctx.idx.tolist()]
        """Whether each batch row has a non-empty mined hard-negative text value."""
        for row in range(batch_size):
            for column in range(batch_size):
                if not (row == column and valid_hard_negatives[column]):
                    neg_mask[row, batch_size + column] = True
    positive_indices = torch.arange(batch_size, device=ctx.pred.device)
    """Diagonal candidate indices: each input's paired target is its positive."""
    contrastive = info_nce_loss(
        InfoNCELossContext(
            anchors=ctx.pred,
            candidates=target_matrix,
            positive_indices=positive_indices,
            temperature=ctx.args.tau,
            logit_mask=neg_mask if bool(neg_mask.any()) else None,
        )
    )
    """Primary contrastive loss delegated to the reusable InfoNCE kernel."""
    objective = contrastive.to_tensor()
    """Scalar contrastive objective before the optional uniformity term."""
    if ctx.args.lam_uniform > 0 and batch_size > 1:  # aux: uniformity
        squared_distances = torch.pdist(F.normalize(ctx.pred, p=2, dim=-1), p=2).pow(2)
        """Pairwise squared distances between normalized prediction embeddings."""
        objective = (
            objective + ctx.args.lam_uniform * squared_distances.mul(-2.0).exp().mean().log()
        )
    return ScalarLoss(objective)


sl_loss_fn: LossFn[SLContext] = sl_loss
"""The single-latent contrastive loss function."""


# ---- multi-latent loss kernels ------------------------------------------------------------------
@dataclass
class InfoNCELossContext:
    """Inputs to :func:`info_nce_loss`.

    The candidate at ``positive_indices[row]`` is the positive for
    ``anchors[row]``. All other unmasked candidates are negatives.
    """

    anchors: torch.Tensor
    """Anchor embeddings, shaped ``[anchor_count, embedding_dim]``."""

    candidates: torch.Tensor
    """Positive and negative candidate embeddings, shaped ``[candidate_count, embedding_dim]``."""

    positive_indices: torch.Tensor
    """Candidate-column index of each anchor's positive, shaped ``[anchor_count]``."""

    temperature: float
    """Positive divisor applied to similarity logits before cross-entropy."""

    logit_mask: torch.Tensor | None = None
    """Optional boolean ``[anchor_count, candidate_count]`` mask; ``True`` excludes a candidate."""

    normalize_embeddings: bool = False
    """Whether to L2-normalize anchors and candidates before calculating similarities."""


def info_nce_loss(ctx: InfoNCELossContext) -> Loss:
    """Compute an InfoNCE-style contrastive cross-entropy loss.

    Every anchor competes against the same candidate matrix. The positive index
    selects its matching candidate, while an optional mask removes false
    negatives or candidates that are not valid for that anchor. Set
    ``normalize_embeddings`` for cosine-similarity InfoNCE; leave it off when
    callers already provide normalized vectors or intentionally use dot product.
    """

    anchors = F.normalize(ctx.anchors, dim=-1) if ctx.normalize_embeddings else ctx.anchors
    """Anchor vectors, optionally normalized to unit length."""
    candidates = F.normalize(ctx.candidates, dim=-1) if ctx.normalize_embeddings else ctx.candidates
    """Candidate vectors, optionally normalized to unit length."""
    logits = anchors @ candidates.t() / ctx.temperature
    """Temperature-scaled embedding similarity logits, one row per anchor."""
    if ctx.logit_mask is not None:
        logits = logits.masked_fill(ctx.logit_mask, float("-inf"))
    return ScalarLoss(F.cross_entropy(logits, ctx.positive_indices))


info_nce_loss_fn: LossFn[InfoNCELossContext] = info_nce_loss
"""The general InfoNCE-style contrastive loss function."""


@dataclass
class SoftTargetCrossEntropyContext:
    """Inputs to :func:`soft_target_cross_entropy_loss`."""

    logits: torch.Tensor
    """Unnormalized class scores, shaped ``[..., class_count]``."""

    target_probabilities: torch.Tensor
    """Target probability distribution aligned with ``logits``."""

    selection: torch.Tensor
    """Boolean mask over the leading dimensions selecting supervised positions."""


def soft_target_cross_entropy_loss(ctx: SoftTargetCrossEntropyContext) -> Loss:
    """Compute mean cross-entropy against probability targets at selected positions.

    This is used by multi-latent code, concept, and state objectives. Unlike
    ordinary classification CE, each target can spread probability mass across
    several classes. It returns scalar zero when no positions are selected.
    """

    log_probabilities = F.log_softmax(ctx.logits, dim=-1)
    """Log-probability for every class at every candidate position."""
    per_position_loss = -(ctx.target_probabilities * log_probabilities).sum(dim=-1)
    """Soft-target cross-entropy for each position before masking and reduction."""
    objective = (
        per_position_loss[ctx.selection].mean()
        if bool(ctx.selection.any())
        else log_probabilities.new_zeros(())
    )
    """Mean selected soft-target CE, or scalar zero when nothing is supervised."""
    return ScalarLoss(objective)


soft_target_cross_entropy_loss_fn: LossFn[SoftTargetCrossEntropyContext] = (
    soft_target_cross_entropy_loss
)
"""The masked soft-target cross-entropy loss function."""


@dataclass
class StopLossContext:
    """Inputs to :func:`stop_loss` for an autoregressive multi-latent emission."""

    logits: torch.Tensor
    """One stop logit per row and emission position, shaped ``[batch_size, max_items + 1]``."""

    lengths: list[int]
    """Number of real emitted items in each batch row; the next position is the stop target."""


def stop_loss(ctx: StopLossContext) -> Loss:
    """Train an independent sigmoid stop decision after each possible emission.

    Each row has label ``0`` through its final real item and label ``1`` at the
    following position. Positions after that stop decision are excluded, so
    padding never contributes to BCE.
    """

    stop_logits = ctx.logits.squeeze(-1).float()
    """Stop logits normalized to ``[batch_size, max_items + 1]``."""
    labels = torch.zeros_like(stop_logits)
    """Binary targets with a single stop position in each batch row."""
    selection = torch.zeros_like(stop_logits, dtype=torch.bool)
    """Mask selecting real continuation decisions and the one stop decision per row."""
    for row, length in enumerate(ctx.lengths):
        labels[row, length] = 1.0
        selection[row, : length + 1] = True
    return ScalarLoss(F.binary_cross_entropy_with_logits(stop_logits[selection], labels[selection]))


stop_loss_fn: LossFn[StopLossContext] = stop_loss
"""The autoregressive emission stop loss function."""


@dataclass
class SupervisedContrastiveLossContext:
    """Inputs to :func:`supervised_contrastive_loss`."""

    embeddings: torch.Tensor
    """Emitted embeddings, shaped ``[item_count, embedding_dim]``."""

    labels: list[str]
    """Group labels aligned with embeddings; empty and unknown labels are excluded."""

    temperature: float
    """Positive divisor applied to pairwise cosine-similarity logits."""


def supervised_contrastive_loss(ctx: SupervisedContrastiveLossContext) -> Loss:
    """Pull same-label embeddings together while pushing differently labeled ones apart.

    This is supervised contrastive learning (SupCon), not classification: labels
    define which other embeddings are positives rather than indexing a classifier
    output. Unlabeled items are excluded. The loss is zero when fewer than two
    labeled examples exist or no anchor has a same-label positive.
    """

    keep = [
        index
        for index, label in enumerate(ctx.labels)
        if str(label).strip().lower() not in ("", "unknown", "none", "nan")
    ]
    """Embedding rows with usable supervision labels."""
    if len(keep) < 2:
        return ScalarLoss(ctx.embeddings.new_zeros(()))
    embeddings = F.normalize(ctx.embeddings[keep], p=2, dim=-1)
    """Unit-length embeddings participating in pairwise cosine similarity."""
    labels = [ctx.labels[index] for index in keep]
    """Usable labels aligned with the selected embeddings."""
    item_count = len(keep)
    """Number of labeled embeddings participating in the objective."""
    similarities = (embeddings @ embeddings.t() / ctx.temperature).masked_fill(
        torch.eye(item_count, device=ctx.embeddings.device, dtype=torch.bool), -1e9
    )
    """Pairwise similarity logits with self-comparisons excluded from every denominator."""
    log_probabilities = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    """Log probability assigned to each other embedding by every anchor."""
    positives = torch.tensor(
        [
            [
                1.0 if row != column and labels[row] == labels[column] else 0.0
                for column in range(item_count)
            ]
            for row in range(item_count)
        ],
        device=ctx.embeddings.device,
    )
    """Indicator matrix identifying same-label, non-self positive pairs."""
    positive_counts = positives.sum(dim=1)
    """Number of positive embeddings available to each anchor."""
    valid_anchors = positive_counts > 0
    """Anchors with at least one same-label positive."""
    if not bool(valid_anchors.any()):
        return ScalarLoss(ctx.embeddings.new_zeros(()))
    objective = (
        -(positives * log_probabilities).sum(dim=1)[valid_anchors] / positive_counts[valid_anchors]
    ).mean()
    """Mean SupCon objective across anchors that have at least one positive."""
    return ScalarLoss(objective)


supervised_contrastive_loss_fn: LossFn[SupervisedContrastiveLossContext] = (
    supervised_contrastive_loss
)
"""The supervised contrastive loss function."""


@dataclass
class CoTLossContext(TrainingContext):
    """Inputs to :func:`cot_loss` after the strategy tokenizes seed and reasoning text."""

    seed_ids: torch.Tensor
    """Left-padded seed token IDs, shaped ``[batch_size, seed_length]``."""

    seed_mask: torch.Tensor
    """Attention mask aligned with ``seed_ids``."""

    cot_ids: torch.Tensor
    """Right-padded chain-of-thought target token IDs."""

    cot_mask: torch.Tensor
    """Attention mask aligned with ``cot_ids``; padding is ignored by cross-entropy."""


def cot_loss(ctx: CoTLossContext) -> Loss:
    """Generate chain-of-thought text from a teacher-forced seed using token CE.

    The left-padded seed's final real token predicts the first CoT token. CoT
    tokens are right-padded, so no padding appears between seed and target text
    and padded targets are ignored by cross-entropy.
    """

    sequence = torch.cat([ctx.seed_ids, ctx.cot_ids], dim=1)
    """Teacher-forced seed followed by chain-of-thought target tokens."""
    attention_mask = torch.cat([ctx.seed_mask, ctx.cot_mask], dim=1)
    """Attention mask aligned with the combined seed and CoT sequence."""
    hidden_states = ctx.model._last_hidden(
        ctx.model._run_backbone(ctx.model.embed(sequence), attention_mask, sequence, 0)
    )
    """Final backbone hidden states for every token in the combined sequence."""
    seed_length = ctx.seed_ids.size(1)
    """Fixed left-padded seed width; its final position predicts the first CoT token."""
    cot_hidden = hidden_states[:, seed_length - 1 : seed_length - 1 + ctx.cot_ids.size(1), :]
    """Shifted hidden states whose outputs predict the aligned CoT tokens."""
    prediction_logits = F.linear(cot_hidden, ctx.model.embed.weight)
    """Vocabulary logits for each CoT target token, kept in the model's native precision."""
    return ScalarLoss(
        F.cross_entropy(
            prediction_logits.reshape(-1, ctx.model.vocab_size),
            ctx.cot_ids.masked_fill(ctx.cot_mask == 0, -100).reshape(-1),
            ignore_index=-100,
        )
    )


cot_loss_fn: LossFn[CoTLossContext] = cot_loss
"""The chain-of-thought generation loss function."""


@dataclass
class BridgeLossContext:
    """Inputs to :func:`bridge_loss` after Hungarian matching prepares bridge targets."""

    matched_predictions: torch.Tensor
    """Hungarian-matched query embeddings, shaped ``[match_count, embedding_dim]``."""

    target_bank: torch.Tensor
    """Normalized in-batch targets followed by optional hard-negative target embeddings."""

    positive_indices: torch.Tensor
    """Index into ``target_bank`` of each matched prediction's assigned target."""

    validity_logits: torch.Tensor
    """One validity logit per bridge query slot."""

    validity_labels: torch.Tensor
    """Binary labels marking the query slots selected by Hungarian matching."""

    temperature: float
    """Positive divisor applied to matched-query versus target-bank similarities."""

    validity_weight: float
    """Weight applied to the validity BCE term in the returned total."""

    positive_weight: float
    """BCE positive-class weight used to compensate for sparse valid query slots."""


@dataclass
class BridgeLoss:
    """A composed bridge objective with inspectable component losses."""

    info_nce: Loss
    """Matched query-to-target retrieval loss."""

    validity: Loss
    """Binary query-validity loss."""

    validity_weight: float
    """Weight applied to ``validity`` when forming the scalar optimization objective."""

    def to_tensor(self) -> torch.Tensor:
        """Return ``info_nce + validity_weight * validity`` for optimization."""
        return self.info_nce.to_tensor() + self.validity_weight * self.validity.to_tensor()


def bridge_loss(ctx: BridgeLossContext) -> BridgeLoss:
    """Compute QueryBridge's matched-target InfoNCE and query-validity BCE.

    Hungarian matching is deliberately outside this function: it creates the
    positive indices and validity labels but is non-differentiable and depends on
    SciPy. This loss consumes those prepared tensors, uses all target-bank rows
    as contrastive candidates, and combines retrieval with weighted validity BCE.
    """

    if ctx.matched_predictions.numel():
        info_nce = info_nce_loss(
            InfoNCELossContext(
                anchors=ctx.matched_predictions,
                candidates=ctx.target_bank,
                positive_indices=ctx.positive_indices,
                temperature=ctx.temperature,
            )
        )
    else:
        info_nce = ScalarLoss(ctx.target_bank.new_zeros(()))
    """InfoNCE over Hungarian-matched query embeddings, or zero with no matches."""
    validity = ScalarLoss(
        F.binary_cross_entropy_with_logits(
            ctx.validity_logits,
            ctx.validity_labels,
            pos_weight=torch.tensor(ctx.positive_weight, device=ctx.validity_logits.device),
        )
    )
    """BCE supervising which bridge query slots should emit a latent."""
    return BridgeLoss(
        info_nce=info_nce,
        validity=validity,
        validity_weight=ctx.validity_weight,
    )


bridge_loss_fn: LossFn[BridgeLossContext] = bridge_loss
"""The composed QueryBridge loss function."""
