"""
preprocessing.py
================
SentinelANC Transmitter-Side Speech Enhancement
Core audio preprocessing primitives for Conv-TasNet training.

Design principles
-----------------
* Pure functions / stateless — no global mutable state.
* Lazy I/O — soundfile reads only the requested slice; never loads 15 GB into RAM.
* Non-destructive — source files are NEVER modified or deleted.
* PyTorch-native tensors as the primary data type throughout.
* All public functions are fully type-annotated.

Dependencies
------------
    pip install torch torchaudio soundfile numpy
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Target sample rate for all processed audio (Hz).
SAMPLE_RATE: int = 16_000

#: Segment length in samples at 16 kHz  → 4 seconds.
SEGMENT_SAMPLES: int = 4 * SAMPLE_RATE  # 64 000

#: SNR levels (dB) used for dynamic noise mixing.
SNR_LEVELS_DB: Tuple[int, ...] = (10, 5, 0, -5)

#: Supported audio file extensions (lowercase).
SUPPORTED_EXTENSIONS: frozenset = frozenset({".wav", ".flac"})

#: Numerical epsilon used to guard against division-by-zero.
_EPS: float = 1e-8

#: Peak amplitude ceiling; mixtures exceeding this are rescaled.
_CLIP_CEILING: float = 0.99

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioInfo:
    """Lightweight metadata returned by :func:`probe_audio_file`.

    Reading metadata via ``soundfile.info()`` does **not** decode audio
    samples, making it O(1) regardless of file size.
    """
    path: Path
    sample_rate: int
    channels: int
    frames: int          # total samples (per channel)
    duration_s: float    # in seconds
    format: str          # e.g. "WAV", "FLAC"
    subtype: str         # e.g. "PCM_16", "PCM_24"


@dataclass
class ValidationResult:
    """Result of :func:`validate_audio_file`."""
    path: Path
    ok: bool
    info: Optional[AudioInfo] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def scan_audio_files(
    root: Union[str, Path],
    recursive: bool = True,
) -> List[Path]:
    """Return a sorted list of all WAV/FLAC files under *root*.

    Parameters
    ----------
    root:
        Directory to scan.
    recursive:
        If ``True`` (default), descend into sub-directories.

    Returns
    -------
    List[Path]
        Absolute :class:`~pathlib.Path` objects, sorted for reproducibility.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Scan root does not exist: {root}")

    pattern = "**/*" if recursive else "*"
    paths = [
        p.resolve()
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def scan_audio_files_iter(
    root: Union[str, Path],
    recursive: bool = True,
) -> Iterator[Path]:
    """Generator variant of :func:`scan_audio_files` — yields paths one-by-one.

    Useful when scanning millions of files to avoid building a large list.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Scan root does not exist: {root}")

    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p.resolve()


# ---------------------------------------------------------------------------
# Metadata probing
# ---------------------------------------------------------------------------

def probe_audio_file(path: Union[str, Path]) -> AudioInfo:
    """Read file metadata **without** decoding audio samples.

    Uses ``soundfile.info()`` which is O(1) regardless of file size.

    Raises
    ------
    RuntimeError
        If the file cannot be opened or is not a recognised audio format.
    """
    path = Path(path)
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise RuntimeError(f"Cannot probe '{path}': {exc}") from exc

    return AudioInfo(
        path=path,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration_s=info.duration,
        format=info.format,
        subtype=info.subtype,
    )


def validate_audio_file(path: Union[str, Path]) -> ValidationResult:
    """Attempt to open and decode a small slice of *path* to confirm integrity.

    Checks performed
    ~~~~~~~~~~~~~~~~
    * Can the file be opened at all? (metadata probe)
    * Is the file non-empty (> 0 frames)?
    * Can the first 1-second slice be decoded without errors?
    * Is the RMS of that slice above silence threshold?
    * Is the peak amplitude below the clipping ceiling?

    Returns
    -------
    ValidationResult
        ``ok=True`` if all checks pass; ``ok=False`` with ``error`` populated
        otherwise.  Non-fatal issues appear in ``warnings``.
    """
    path = Path(path)
    result = ValidationResult(path=path, ok=False)

    # --- 1. Metadata probe ---------------------------------------------------
    try:
        info = probe_audio_file(path)
        result.info = info
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    if info.frames == 0:
        result.error = "File has 0 frames (empty audio)."
        return result

    # --- 2. Decode first 1-second slice -------------------------------------
    probe_frames = min(info.frames, info.sample_rate)  # up to 1 second
    try:
        data, _ = sf.read(
            str(path),
            frames=probe_frames,
            always_2d=True,
            dtype="float32",
        )
    except Exception as exc:
        result.error = f"Decode error on first {probe_frames} frames: {exc}"
        return result

    # --- 3. Amplitude checks -------------------------------------------------
    rms = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))

    if rms < 1e-5:
        result.warnings.append(
            f"Very low RMS ({rms:.2e}) — possible silence or near-silence."
        )
    if peak > _CLIP_CEILING:
        result.warnings.append(
            f"Peak amplitude {peak:.4f} exceeds {_CLIP_CEILING} — possible clipping."
        )

    result.ok = True
    return result


# ---------------------------------------------------------------------------
# Audio loading  (lazy / memory-efficient)
# ---------------------------------------------------------------------------

def load_audio(
    path: Union[str, Path],
    start_frame: int = 0,
    num_frames: int = -1,
) -> Tuple[torch.Tensor, int]:
    """Load audio from *path* into a float32 tensor.

    This function is **memory-efficient**: soundfile reads only the requested
    frame range.  The full file is never loaded unless *num_frames* == -1.

    Parameters
    ----------
    path:
        Path to a WAV or FLAC file.
    start_frame:
        First sample to read (0-indexed, per-channel).
    num_frames:
        Number of samples to read.  ``-1`` reads to the end of file.

    Returns
    -------
    waveform : torch.Tensor
        Shape ``[C, T]`` where *C* = channel count and *T* = sample count.
    sample_rate : int
        Native sample rate of the file (before any resampling).

    Raises
    ------
    RuntimeError
        If the file cannot be decoded.
    """
    path = Path(path)
    kwargs: Dict = dict(always_2d=True, dtype="float32")
    if start_frame > 0:
        kwargs["start"] = start_frame
    if num_frames >= 0:
        kwargs["frames"] = num_frames

    try:
        data, sr = sf.read(str(path), **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load '{path}': {exc}") from exc

    # soundfile → (frames, channels); PyTorch expects (channels, frames)
    waveform = torch.from_numpy(data.T.copy())  # [C, T], float32
    return waveform, sr


# ---------------------------------------------------------------------------
# Channel processing
# ---------------------------------------------------------------------------

def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a multi-channel waveform to mono by averaging channels.

    Parameters
    ----------
    waveform:
        Shape ``[C, T]`` — any number of channels.

    Returns
    -------
    torch.Tensor
        Shape ``[1, T]``.
    """
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected 2-D tensor [C, T], got shape {tuple(waveform.shape)}"
        )
    if waveform.shape[0] == 1:
        return waveform  # already mono — no copy
    return waveform.mean(dim=0, keepdim=True)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Resample *waveform* to *target_sr* using a high-quality sinc filter.

    Parameters
    ----------
    waveform:
        Shape ``[C, T]``.
    orig_sr:
        Original sample rate (Hz).
    target_sr:
        Desired sample rate (Hz).  Defaults to :data:`SAMPLE_RATE` (16 kHz).

    Returns
    -------
    torch.Tensor
        Resampled waveform, shape ``[C, T']``.
    """
    if orig_sr == target_sr:
        return waveform
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive. Got {orig_sr} -> {target_sr}.")

    return F.resample(
        waveform,
        orig_freq=orig_sr,
        new_freq=target_sr,
        resampling_method="sinc_interp_hann",
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Peak-normalize *waveform* to ``[-1, 1]``.

    For near-silent signals (peak < :data:`_EPS`) the waveform is returned
    unchanged to avoid amplifying digital silence or noise floor artifacts.

    Parameters
    ----------
    waveform:
        Shape ``[C, T]``.

    Returns
    -------
    torch.Tensor
        Normalized waveform, same shape.
    """
    peak = waveform.abs().max()
    if peak < _EPS:
        warnings.warn(
            "normalize_audio: waveform peak is near zero — skipping normalization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return waveform
    return waveform / peak


# ---------------------------------------------------------------------------
# Segmentation / padding
# ---------------------------------------------------------------------------

def segment_or_pad(
    waveform: torch.Tensor,
    segment_samples: int = SEGMENT_SAMPLES,
    hop_samples: Optional[int] = None,
) -> List[torch.Tensor]:
    """Split a waveform into fixed-length segments or zero-pad if shorter.

    * If ``T >= segment_samples``: the waveform is sliced into consecutive
      non-overlapping windows of exactly *segment_samples*.  Any trailing
      remainder shorter than one full window is **zero-padded** and included.
    * If ``T < segment_samples``: the waveform is zero-padded on the right
      to *segment_samples* and returned as a single-element list.

    Parameters
    ----------
    waveform:
        Shape ``[C, T]``.
    segment_samples:
        Target window length in samples.  Default: :data:`SEGMENT_SAMPLES`
        (64 000 samples = 4 s at 16 kHz).
    hop_samples:
        Stride between successive windows.  Defaults to *segment_samples*
        (non-overlapping).  Set to a smaller value for overlapping windows.

    Returns
    -------
    List[torch.Tensor]
        Each element has shape ``[C, segment_samples]``.
    """
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected [C, T] tensor, got shape {tuple(waveform.shape)}"
        )
    C, T = waveform.shape
    if hop_samples is None:
        hop_samples = segment_samples

    segments: List[torch.Tensor] = []

    if T < segment_samples:
        # Single segment — zero-pad on the right
        pad = torch.zeros(C, segment_samples - T, dtype=waveform.dtype)
        segments.append(torch.cat([waveform, pad], dim=1))
        return segments

    # Multiple segments
    num_full = (T - segment_samples) // hop_samples + 1
    for i in range(num_full):
        start = i * hop_samples
        end = start + segment_samples
        segments.append(waveform[:, start:end].clone())

    # Remainder
    remainder_start = num_full * hop_samples
    if remainder_start < T:
        tail = waveform[:, remainder_start:]
        pad = torch.zeros(C, segment_samples - tail.shape[1], dtype=waveform.dtype)
        segments.append(torch.cat([tail, pad], dim=1))

    return segments


def get_segment(
    waveform: torch.Tensor,
    segment_idx: int,
    segment_samples: int = SEGMENT_SAMPLES,
    hop_samples: Optional[int] = None,
) -> torch.Tensor:
    """Return a single segment by index without materialising all segments.

    Useful for :meth:`~torch.utils.data.Dataset.__getitem__` where you know
    the segment index ahead of time (e.g. from a pre-built manifest).

    Parameters
    ----------
    waveform:
        Shape ``[C, T]``.
    segment_idx:
        0-based index into the list that :func:`segment_or_pad` would return.
    segment_samples:
        Window length in samples.
    hop_samples:
        Stride between successive windows.

    Returns
    -------
    torch.Tensor
        Shape ``[C, segment_samples]``.
    """
    if hop_samples is None:
        hop_samples = segment_samples

    C, T = waveform.shape
    start = segment_idx * hop_samples
    end = start + segment_samples

    if start >= T:
        raise IndexError(
            f"segment_idx {segment_idx} is out of range for waveform of "
            f"length {T} with hop {hop_samples}."
        )

    chunk = waveform[:, start:min(end, T)]
    if chunk.shape[1] < segment_samples:
        pad = torch.zeros(C, segment_samples - chunk.shape[1], dtype=waveform.dtype)
        chunk = torch.cat([chunk, pad], dim=1)
    return chunk


def count_segments(
    num_frames: int,
    sample_rate: int,
    target_sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
    hop_samples: Optional[int] = None,
) -> int:
    """Return the number of segments the pipeline would produce for a file.

    Accounts for the fact that the file may have a native sample rate different
    from *target_sr*.  All arithmetic is done in the **target** sample-rate
    domain so the result matches what :func:`load_segment` will produce.

    Parameters
    ----------
    num_frames:
        Total frames in the file at *sample_rate*.
    sample_rate:
        Native sample rate of the file.
    target_sr:
        Target sample rate after resampling.
    segment_samples:
        Segment length in **target** samples.
    hop_samples:
        Stride between windows in **target** samples.

    Returns
    -------
    int
    """
    if hop_samples is None:
        hop_samples = segment_samples
    if num_frames <= 0:
        return 0

    # Resampled length (approximate — same as what torchaudio would produce)
    resampled_frames = math.ceil(num_frames * target_sr / sample_rate)

    if resampled_frames < segment_samples:
        return 1
    num_full = (resampled_frames - segment_samples) // hop_samples + 1
    remainder_start = num_full * hop_samples
    extra = 1 if remainder_start < resampled_frames else 0
    return num_full + extra


# ---------------------------------------------------------------------------
# Noise mixing  (SNR-controlled, clipping-safe)
# ---------------------------------------------------------------------------

def compute_rms(waveform: torch.Tensor) -> float:
    """Return root-mean-square energy of *waveform* as a Python float.

    Parameters
    ----------
    waveform:
        Arbitrary shape.

    Returns
    -------
    float
    """
    return float(waveform.pow(2).mean().sqrt())


def mix_with_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    prevent_clipping: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mix *clean* and *noise* at the requested SNR level.

    Mixing algorithm
    ~~~~~~~~~~~~~~~~
    1. Compute RMS of *clean* and *noise* over the full segment.
    2. Scale *noise* so that ``RMS(clean) / RMS(scaled_noise) == 10^(snr_db/20)``.
    3. Produce ``mixture = clean + scaled_noise``.
    4. If ``|mixture|_max > _CLIP_CEILING``, rescale **both** clean and
       mixture by the same scalar (their ratio is preserved — the Conv-TasNet
       objective is unchanged because only the scale changes, not the
       clean-to-noisy relationship).

    Parameters
    ----------
    clean:
        Clean reference waveform, shape ``[C, T]``.
    noise:
        Noise waveform, shape ``[C, T]``.  Must have the same shape as *clean*.
    snr_db:
        Target signal-to-noise ratio in dB.
    prevent_clipping:
        Whether to apply the anti-clipping rescaling step.  Default: ``True``.

    Returns
    -------
    mixture : torch.Tensor
        Noisy speech, shape ``[C, T]``.
    clean_aligned : torch.Tensor
        Clean reference (possibly rescaled for clipping prevention),
        shape ``[C, T]``.

    Raises
    ------
    ValueError
        If *clean* and *noise* do not have the same shape.
    """
    if clean.shape != noise.shape:
        raise ValueError(
            f"clean {tuple(clean.shape)} and noise {tuple(noise.shape)} must "
            f"have the same shape."
        )

    clean_rms = compute_rms(clean)
    noise_rms = compute_rms(noise)

    # Target noise RMS from SNR formula: SNR = 20*log10(rms_s / rms_n)
    #   => rms_n_target = rms_s / 10^(snr_db/20)
    linear_ratio = 10.0 ** (snr_db / 20.0)

    if noise_rms < _EPS:
        warnings.warn(
            "mix_with_snr: noise RMS is near zero — cannot scale noise to "
            "target SNR.  Returning clean signal as mixture.",
            RuntimeWarning,
            stacklevel=2,
        )
        return clean.clone(), clean.clone()

    if clean_rms < _EPS:
        warnings.warn(
            "mix_with_snr: clean RMS is near zero — resulting mixture will "
            "be dominated by noise.",
            RuntimeWarning,
            stacklevel=2,
        )

    target_noise_rms = clean_rms / (linear_ratio + _EPS)
    scale_factor = target_noise_rms / (noise_rms + _EPS)
    scaled_noise = noise * scale_factor

    mixture = clean + scaled_noise
    clean_out = clean.clone()

    # --- Anti-clipping rescaling -------------------------------------------
    if prevent_clipping:
        peak = mixture.abs().max()
        if peak > _CLIP_CEILING:
            rescale = _CLIP_CEILING / (peak + _EPS)
            mixture = mixture * rescale
            clean_out = clean_out * rescale

    return mixture, clean_out


# ---------------------------------------------------------------------------
# Convenience: full pipeline for a single file
# ---------------------------------------------------------------------------

def preprocess_file(
    path: Union[str, Path],
    target_sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
    normalize: bool = True,
) -> List[torch.Tensor]:
    """Load -> mono -> resample -> normalize -> segment a single audio file.

    This is a convenience wrapper combining the individual steps.  It is
    **not** called during training (which uses lazy loading via
    :func:`load_segment`), but is useful for offline inspection and unit tests.

    Parameters
    ----------
    path:
        Path to a WAV or FLAC file.
    target_sr:
        Target sample rate (Hz).  Default: 16 000.
    segment_samples:
        Segment length in samples.  Default: 64 000 (4 s).
    normalize:
        Whether to peak-normalize before segmentation.

    Returns
    -------
    List[torch.Tensor]
        Each element has shape ``[1, segment_samples]``.
    """
    waveform, sr = load_audio(path)
    waveform = to_mono(waveform)
    waveform = resample_audio(waveform, sr, target_sr)
    if normalize:
        waveform = normalize_audio(waveform)
    return segment_or_pad(waveform, segment_samples=segment_samples)


# ---------------------------------------------------------------------------
# Lazy segment reader  (called inside Dataset.__getitem__)
# ---------------------------------------------------------------------------

def load_segment(
    path: Union[str, Path],
    segment_idx: int,
    native_sr: int,
    native_frames: int,
    target_sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
    normalize: bool = True,
) -> torch.Tensor:
    """Load exactly one segment from disk without reading the whole file.

    This is the hot path called inside :meth:`~torch.utils.data.Dataset.__getitem__`.
    It computes the byte-level frame range to ask soundfile for, avoiding
    any wasted I/O.

    Strategy
    ~~~~~~~~
    Work in **native sample-rate space** when issuing the read request, then
    resample the chunk.  This avoids loading the entire file just to
    resample it before slicing.

    Parameters
    ----------
    path:
        Path to a WAV or FLAC file.
    segment_idx:
        0-based index of the desired segment.
    native_sr:
        Native sample rate of the file (from the manifest / probe).
    native_frames:
        Total frames in the file (from the manifest / probe).
    target_sr:
        Target sample rate after resampling.
    segment_samples:
        Segment length **in target sample-rate samples**.
    normalize:
        Whether to peak-normalize the segment.

    Returns
    -------
    torch.Tensor
        Shape ``[1, segment_samples]``.
    """
    # --- Compute slice in native sample-rate space --------------------------
    # Each target segment_samples maps to this many native frames:
    native_segment = math.ceil(segment_samples * native_sr / target_sr)
    native_start = segment_idx * native_segment

    if native_start >= native_frames:
        raise IndexError(
            f"segment_idx {segment_idx} starts at native frame {native_start}, "
            f"but file only has {native_frames} frames: {path}"
        )

    native_end = min(native_start + native_segment, native_frames)
    native_read = native_end - native_start

    waveform, _ = load_audio(path, start_frame=native_start, num_frames=native_read)
    waveform = to_mono(waveform)

    if native_sr != target_sr:
        waveform = resample_audio(waveform, native_sr, target_sr)

    # Exact-length guarantee — resampling may introduce ±1 sample rounding
    if waveform.shape[1] < segment_samples:
        pad = torch.zeros(1, segment_samples - waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :segment_samples]

    if normalize:
        waveform = normalize_audio(waveform)

    return waveform  # [1, segment_samples]
