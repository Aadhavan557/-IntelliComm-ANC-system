"""
loss.py
=======
IntelliComm - AI-Based Speech Enhancement for Military/Field Communication
Loss functions and evaluation metrics for Conv-TasNet training.

Exports
-------
sisnr_loss(pred, target)          -> scalar loss  (lower is better)
combined_loss(pred, target, ...)  -> scalar loss  (SI-SNR + 0.1 * L1)
si_snr_metric(pred, target)       -> scalar dB    (higher is better)
si_sdr_metric(pred, target)       -> scalar dB    (higher is better)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Numerical stability floor
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Core: Scale-Invariant SNR
# ---------------------------------------------------------------------------

def _zero_mean(x: torch.Tensor) -> torch.Tensor:
    """Remove DC bias along the time axis (last dimension)."""
    return x - x.mean(dim=-1, keepdim=True)


def _si_snr_components(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (s_target, e_noise) tensors for SI-SNR / SI-SDR computation.

    Parameters
    ----------
    pred   : [B, 1, T] or [B, T]   model output (enhanced waveform)
    target : [B, 1, T] or [B, T]   clean reference waveform

    Returns
    -------
    s_target : torch.Tensor   projection of target onto pred direction
    e_noise  : torch.Tensor   noise residual = pred - s_target
    """
    # Flatten to [B, T]
    pred   = pred.squeeze(1)
    target = target.squeeze(1)

    # Zero-mean
    pred   = _zero_mean(pred)
    target = _zero_mean(target)

    # Projection: s_target = <pred, target> / ||target||^2 * target
    dot     = (pred * target).sum(dim=-1, keepdim=True)         # [B, 1]
    t_power = (target ** 2).sum(dim=-1, keepdim=True) + _EPS    # [B, 1]
    s_target = dot / t_power * target                           # [B, T]

    e_noise = pred - s_target                                   # [B, T]
    return s_target, e_noise


def sisnr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Batch-averaged Scale-Invariant SNR loss (negated, lower = better).

    SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    Loss   = -mean(SI-SNR)   so gradient descent maximises SI-SNR.

    Parameters
    ----------
    pred   : torch.Tensor  [B, 1, T]  model output
    target : torch.Tensor  [B, 1, T]  clean reference

    Returns
    -------
    torch.Tensor  scalar
    """
    s_target, e_noise = _si_snr_components(pred, target)
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + _EPS) + _EPS
    )
    return -si_snr.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.1,
) -> torch.Tensor:
    """SI-SNR loss + weighted L1 waveform loss.

    total = sisnr_loss(pred, target) + l1_weight * L1(pred, target)

    Parameters
    ----------
    pred      : torch.Tensor  [B, 1, T]
    target    : torch.Tensor  [B, 1, T]
    l1_weight : float         weight for the L1 term (default 0.1)

    Returns
    -------
    torch.Tensor  scalar
    """
    loss_sisnr = sisnr_loss(pred, target)
    loss_l1    = F.l1_loss(pred, target)
    return loss_sisnr + l1_weight * loss_l1


# ---------------------------------------------------------------------------
# Metrics  (no gradients needed; use torch.no_grad() when calling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def si_snr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Scale-Invariant SNR in dB (higher = better).

    Returns the batch-averaged SI-SNR value as a plain Python float.
    """
    s_target, e_noise = _si_snr_components(pred, target)
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + _EPS) + _EPS
    )
    return si_snr.mean().item()


@torch.no_grad()
def si_sdr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Scale-Invariant SDR in dB (higher = better).

    SI-SDR uses the same formula as SI-SNR when applied to the estimated
    source vs. the ground-truth. For single-source speech enhancement
    SI-SDR = SI-SNR; included as a separate named metric for logging clarity.

    Returns the batch-averaged value as a plain Python float.
    """
    return si_snr_metric(pred, target)


@torch.no_grad()
def snr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Standard (non-scale-invariant) SNR in dB.

    SNR = 10 * log10(||target||^2 / ||pred - target||^2)
    """
    pred   = pred.squeeze(1)
    target = target.squeeze(1)
    noise  = pred - target
    snr    = 10 * torch.log10(
        (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + _EPS) + _EPS
    )
    return snr.mean().item()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 64000
    clean = torch.randn(B, 1, T)
    noisy = clean + 0.3 * torch.randn(B, 1, T)

    print("--- Loss functions ---")
    print(f"sisnr_loss  (noisy vs clean) : {sisnr_loss(noisy, clean):.4f}")
    print(f"sisnr_loss  (clean vs clean) : {sisnr_loss(clean, clean):.4f}")
    print(f"combined_loss               : {combined_loss(noisy, clean):.4f}")

    print("\n--- Metrics ---")
    print(f"SI-SNR  (noisy vs clean) : {si_snr_metric(noisy, clean):.2f} dB")
    print(f"SI-SNR  (clean vs clean) : {si_snr_metric(clean, clean):.2f} dB")
    print(f"SI-SDR  (noisy vs clean) : {si_sdr_metric(noisy, clean):.2f} dB")
    print(f"SNR     (noisy vs clean) : {snr_metric(noisy, clean):.2f} dB")
