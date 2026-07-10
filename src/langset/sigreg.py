"""SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun, arXiv:2511.08544).

An EMA-free anti-collapse mechanism. Instead of a stop-gradient EMA teacher preventing representation
collapse, SIGReg directly constrains the embedding distribution toward an isotropic Gaussian via an
Epps-Pulley goodness-of-fit test: project embeddings onto random 1-D slices and compare each slice's
empirical characteristic function against the standard normal's, exp(-t^2/2), by Gaussian-windowed
quadrature on [0, 3]. If every random 1-D projection looks standard-normal, the joint is isotropic
Gaussian (Cramer-Wold).

Used by trainer.py behind `TrainingArguments.use_sigreg` as the alternative to the EMA twin. TRAINING-ONLY
(never persisted, never used at eval). For a token-native FSQ head, hook this on the PRE-QUANTIZATION
z = down_proj(latent) (the raw FSQ input), NOT the emitted reconstruction: regularizing the input to the
quantizer spreads the encoder's codes across the whole grid, which is what stops the (twin-free) live
encoder folding every input into one cell.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Epps-Pulley isotropic-Gaussian test. forward(x[..., N, D]) -> scalar statistic (0 == perfectly Gaussian).

    knots  = quadrature points on [0, 3] for the CF integral (trapezoid, Gaussian-windowed).
    slices = number of random 1-D projection directions, resampled every call.
    """

    def __init__(self, knots: int = 17, slices: int = 256) -> None:
        super().__init__()
        self.slices = slices
        t = torch.linspace(0, 3, knots)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt                                  # trapezoid endpoints
        window = torch.exp(-t.square() / 2.0)                 # standard-normal characteristic function
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)     # window the quadrature so the tail is down-weighted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., N, D]. CENTER ONLY — do NOT divide by std. The target is the STANDARD normal N(0,1), whose unit
        # VARIANCE is the anti-collapse mechanism: a collapsed batch (variance -> 0) must score badly so the gradient
        # spreads it back out to variance 1. Standardizing first divides by that near-zero std and rescales collapse
        # to unit variance, so it passes the Gaussianity test with ZERO spreading gradient (a fully collapsed batch
        # scores ~0.6 standardized vs ~100 centered). Center-only keeps the mean-0 target while making variance-
        # collapse the dominant penalty.
        x = x - x.mean(-2, keepdim=True)
        A = torch.randn(x.size(-1), self.slices, device=x.device, dtype=x.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)               # unit-norm slice directions, resampled each call
        x_t = (x @ A).unsqueeze(-1) * self.t                   # [..., N, slices, knots]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()   # emp. CF vs target CF
        statistic = (err @ self.weights) * x.size(-2)          # Epps-Pulley statistic per slice, scaled by N
        return statistic.mean()
