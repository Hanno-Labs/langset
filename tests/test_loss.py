"""Unit tests for reusable loss kernels."""

import torch
import torch.nn.functional as F

from langset.loss import InfoNCELossContext, info_nce_loss


def test_info_nce_applies_logit_mask() -> None:
    """Masked candidates are excluded from each anchor's contrastive denominator."""
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positive_indices = torch.tensor([0, 1])
    logit_mask = torch.tensor([[False, True], [False, False]])

    loss = info_nce_loss(
        InfoNCELossContext(
            anchors=anchors,
            candidates=candidates,
            positive_indices=positive_indices,
            temperature=1.0,
            logit_mask=logit_mask,
        )
    ).to_tensor()

    expected = F.cross_entropy(
        (anchors @ candidates.t()).masked_fill(logit_mask, float("-inf")), positive_indices
    )
    assert torch.allclose(loss, expected)


def test_info_nce_normalization_toggle_changes_similarity_scale() -> None:
    """Normalization switches InfoNCE from dot-product to cosine similarities."""
    anchors = torch.tensor([[2.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positive_indices = torch.tensor([0])

    dot_product_loss = info_nce_loss(
        InfoNCELossContext(
            anchors=anchors,
            candidates=candidates,
            positive_indices=positive_indices,
            temperature=1.0,
        )
    ).to_tensor()
    cosine_loss = info_nce_loss(
        InfoNCELossContext(
            anchors=anchors,
            candidates=candidates,
            positive_indices=positive_indices,
            temperature=1.0,
            normalize_embeddings=True,
        )
    ).to_tensor()

    assert torch.allclose(
        dot_product_loss, F.cross_entropy(torch.tensor([[2.0, 0.0]]), positive_indices)
    )
    assert torch.allclose(
        cosine_loss, F.cross_entropy(torch.tensor([[1.0, 0.0]]), positive_indices)
    )
    assert not torch.allclose(dot_product_loss, cosine_loss)


def test_info_nce_backpropagates_to_anchors_and_candidates() -> None:
    """The scalar loss preserves autograd paths through both embedding inputs."""
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    candidates = torch.tensor([[0.8, 0.2], [0.2, 0.8]], requires_grad=True)

    loss = info_nce_loss(
        InfoNCELossContext(
            anchors=anchors,
            candidates=candidates,
            positive_indices=torch.tensor([0, 1]),
            temperature=0.5,
        )
    ).to_tensor()
    loss.backward()

    assert anchors.grad is not None
    assert candidates.grad is not None
    assert bool(anchors.grad.abs().any())
    assert bool(candidates.grad.abs().any())
