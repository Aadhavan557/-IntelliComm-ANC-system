"""
model.py
========
IntelliComm - AI-Based Speech Enhancement for Military/Field Communication
Conv-TasNet model for waveform-domain speech enhancement.

Architecture
------------
  Input  : [B, 1, T]        waveform (T = 64000 at 16 kHz -> 4 s)
  Output : [B, 1, T]        enhanced waveform

  Encoder   -> Conv1d  (1->N, kernel=L, stride=L//2) + ReLU
  Separator -> TCN stacks producing N-channel masks via sigmoid
  Decoder   -> ConvTranspose1d (N->1, kernel=L, stride=L//2)

Default hyper-parameters (Conv-TasNet paper, Table 1):
  N = 256   encoder channels
  L = 20    encoder kernel size  (1.25 ms at 16 kHz)
  B = 256   bottleneck channels in TCN
  H = 512   hidden channels in TCN depthwise blocks
  P = 3     depthwise conv kernel size
  X = 8     conv blocks per repeat
  R = 4     number of repeats

References
----------
  Luo & Mesgarani (2019) "Conv-TasNet: Surpassing ideal time-frequency
  magnitude masking for speech separation." IEEE/ACM TASLP 27(8):1256-1266.
  https://arxiv.org/abs/1809.07454
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _GlobalLayerNorm(nn.Module):
    """Global layer norm (gLN) operating on [B, C, T] tensors."""

    def __init__(self, num_features: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_features, 1))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[1, 2], keepdim=True)
        std  = ((x - mean).pow(2).mean(dim=[1, 2], keepdim=True) + self.eps).sqrt()
        return self.gamma * (x - mean) / std + self.beta


class _DepthwiseSeparableConv(nn.Module):
    """One TCN building block: 1x1 conv -> D-conv -> 1x1 conv + residual skip."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int = 1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.PReLU(),
            _GlobalLayerNorm(hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                groups=hidden_channels,
            ),
            nn.PReLU(),
            _GlobalLayerNorm(hidden_channels),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
        )
        self.skip_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        if self.causal:
            out = out[..., : x.shape[-1]]
        residual = out + x
        skip     = self.skip_conv(out)
        return residual, skip


class _TCN(nn.Module):
    """Full Temporal Convolutional Network separator."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_blocks: int,
        num_repeats: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_repeats):
            for block_idx in range(num_blocks):
                dilation = 2 ** block_idx
                self.blocks.append(
                    _DepthwiseSeparableConv(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        causal=causal,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return skip_sum


# ---------------------------------------------------------------------------
# Conv-TasNet
# ---------------------------------------------------------------------------

class ConvTasNet(nn.Module):
    """Conv-TasNet for single-channel speech enhancement.

    Input : [B, 1, T]  ->  Output : [B, 1, T]

    Parameters
    ----------
    N : int   Encoder channels (256)
    L : int   Encoder kernel size (20)
    B : int   TCN bottleneck channels (256)
    H : int   TCN hidden channels (512)
    P : int   TCN depthwise kernel size (3)
    X : int   TCN blocks per repeat (8)
    R : int   TCN repeats (4)
    causal : bool  Causal convolutions for streaming
    """

    def __init__(
        self,
        N: int = 256,
        L: int = 20,
        B: int = 256,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 4,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.N = N; self.L = L; self.B = B; self.H = H
        self.P = P; self.X = X; self.R = R; self.causal = causal

        stride = L // 2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, N, kernel_size=L, stride=stride, padding=0, bias=False),
            nn.ReLU(),
        )
        # Separator
        self.bottleneck_norm = _GlobalLayerNorm(N)
        self.bottleneck      = nn.Conv1d(N, B, kernel_size=1)
        self.tcn = _TCN(
            in_channels=B, hidden_channels=H, kernel_size=P,
            num_blocks=X, num_repeats=R, causal=causal,
        )
        self.mask_conv = nn.Sequential(
            nn.Conv1d(B, N, kernel_size=1),
            nn.Sigmoid(),
        )
        # Decoder
        self.decoder = nn.ConvTranspose1d(
            N, 1, kernel_size=L, stride=stride, padding=0, bias=False
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_in = x.shape[-1]
        enc  = self.encoder(x)                               # [B, N, T_enc]
        sep  = self.bottleneck(self.bottleneck_norm(enc))    # [B, B, T_enc]
        skip = self.tcn(sep)                                 # [B, B, T_enc]
        mask = self.mask_conv(skip)                          # [B, N, T_enc]
        out  = self.decoder(enc * mask)                      # [B, 1, T_out]
        return self._match_length(out, T_in)

    @staticmethod
    def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur > target_len:
            return x[..., :target_len]
        if cur < target_len:
            return F.pad(x, (0, target_len - cur))
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            "ConvTasNet",
            f"  Encoder  : Conv1d(1->{self.N}, kernel={self.L}, stride={self.L//2})",
            f"  TCN      : B={self.B}, H={self.H}, P={self.P}, X={self.X}, R={self.R}",
            f"  Decoder  : ConvTranspose1d({self.N}->1, kernel={self.L}, stride={self.L//2})",
            f"  Causal   : {self.causal}",
            f"  Params   : {self.count_parameters():,}",
        ]
        return "\n".join(lines)


def build_model(
    N: int = 256, L: int = 20, B: int = 256, H: int = 512,
    P: int = 3, X: int = 8, R: int = 4,
    causal: bool = False,
    device: Optional[torch.device] = None,
) -> ConvTasNet:
    """Construct ConvTasNet and move to *device* (default: CUDA if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ConvTasNet(N=N, L=L, B=B, H=H, P=P, X=X, R=R, causal=causal).to(device)


if __name__ == "__main__":
    import time
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device=dev)
    print(model.summary())
    print(f"\nDevice : {dev}")
    x     = torch.randn(2, 1, 64000, device=dev)
    t0    = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")
    print(f"Time   : {(time.perf_counter()-t0)*1000:.1f} ms (batch=2, no_grad)")
    if dev.type == "cuda":
        print(f"GPU mem: {torch.cuda.max_memory_allocated(dev)/1e6:.1f} MB")
