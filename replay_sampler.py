"""
replay_sampler.py
=================
IntelliComm - AI-Based Speech Enhancement for Military/Field Communication

Experience-replay dataset components for staged continual training.

Design
------
Stage 1 : Train on Chunk 1 (no replay)
Stage 2 : 80% Chunk 2 + 20% replay from Chunk 1
Stage 3 : 80% Chunk 3 + 20% replay from Chunks 1 & 2
Stage 4 : 80% Chunk 4 + 20% replay from Chunks 1, 2 & 3

NO audio files are duplicated. The replay buffer is a list of file paths;
audio is loaded lazily on-the-fly inside __getitem__.

Classes
-------
ChunkDataset     - PyTorch Dataset for a single chunk of clean files.
ReplayDataset    - Random subset of past chunks for experience replay.
build_stage_dataset - Factory: returns (Dataset, Optional[WeightedRandomSampler])
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from preprocessing import (
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SNR_LEVELS_DB,
    probe_audio_file,
    load_segment,
    mix_with_snr,
    normalize_audio,
    count_segments,
)
from chunk_builder import get_chunk_paths, get_noise_paths

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Helper: flatten chunk paths into (AudioInfo, segment_idx) index
# ---------------------------------------------------------------------------

def _build_segment_index(
    clean_paths: List[Path],
    target_sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
) -> List[Tuple]:
    """Return list of (path, native_sr, native_frames, segment_idx) per segment."""
    index = []
    n_failed = 0
    for p in clean_paths:
        try:
            info = probe_audio_file(p)
            # AudioInfo fields: sample_rate, frames, channels, duration_s
            native_sr     = info.sample_rate
            native_frames = info.frames
            n_segs = count_segments(
                num_frames=native_frames,
                sample_rate=native_sr,
                target_sr=target_sr,
                segment_samples=segment_samples,
            )
            # Always yield at least one segment (handles short/padded files)
            n_segs = max(n_segs, 1)
            for seg_idx in range(n_segs):
                index.append((p, native_sr, native_frames, seg_idx))
        except Exception as exc:
            n_failed += 1
            if n_failed <= 5:
                logger.warning("Skipping %s: %s", p.name, exc)
    if n_failed > 0:
        logger.warning("Skipped %d/%d files (unreadable)", n_failed, len(clean_paths))
    return index


# ---------------------------------------------------------------------------
# ChunkDataset
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    """Lazy PyTorch Dataset for one chunk of clean speech files.

    Each __getitem__:
      1. Loads one 4-second clean segment from disk (lazy).
      2. Picks a random noise file and segment.
      3. Mixes at a randomly chosen SNR level (+10/+5/0/-5 dB).
      4. Returns {"noisy": [1, 64000], "clean": [1, 64000]}.
    """

    def __init__(
        self,
        clean_paths: List[Path],
        noise_paths: List[Path],
        snr_levels: Tuple[int, ...] = SNR_LEVELS_DB,
        seed: int = 42,
        target_sr: int = SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.noise_paths     = noise_paths
        self.snr_levels      = snr_levels
        self.seed            = seed
        self.target_sr       = target_sr
        self.segment_samples = segment_samples
        self.normalize       = normalize

        logger.info("Building segment index for %d clean files ...", len(clean_paths))
        self._index = _build_segment_index(clean_paths, target_sr, segment_samples)
        logger.info("  -> %d segments total", len(self._index))

        # Build noise index similarly
        self._noise_index = _build_segment_index(noise_paths, target_sr, segment_samples) \
            if noise_paths else []

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean_path, native_sr, native_frames, seg_idx = self._index[idx]

        # Deterministic per-sample RNG (safe with multiprocessing workers)
        rng = random.Random(self.seed + idx)

        # --- Load clean segment ---
        try:
            clean = load_segment(
                path=clean_path,
                segment_idx=seg_idx,
                native_sr=native_sr,
                native_frames=native_frames,
                target_sr=self.target_sr,
                segment_samples=self.segment_samples,
                normalize=self.normalize,
            )                                      # [1, segment_samples]
        except Exception as exc:
            logger.debug("load_segment failed for %s seg %d: %s", clean_path.name, seg_idx, exc)
            clean = torch.zeros(1, self.segment_samples)

        # --- Load noise segment ---
        if self._noise_index:
            n_path, n_native_sr, n_native_frames, n_seg_idx = rng.choice(self._noise_index)
            try:
                noise = load_segment(
                    path=n_path,
                    segment_idx=n_seg_idx,
                    native_sr=n_native_sr,
                    native_frames=n_native_frames,
                    target_sr=self.target_sr,
                    segment_samples=self.segment_samples,
                    normalize=self.normalize,
                )
            except Exception:
                noise = torch.randn_like(clean) * 0.01
        else:
            noise = torch.randn_like(clean) * 0.01

        # --- Mix at random SNR ---
        # mix_with_snr returns (noisy, clean_scaled) tuple
        snr_db = rng.choice(self.snr_levels)
        try:
            noisy, clean = mix_with_snr(clean, noise, snr_db=snr_db)
        except Exception:
            noisy = clean + 0.1 * noise

        return {"noisy": noisy, "clean": clean}


# ---------------------------------------------------------------------------
# ReplayDataset
# ---------------------------------------------------------------------------

class ReplayDataset(Dataset):
    """Random subset of past-chunk segments for experience replay."""

    def __init__(
        self,
        past_chunk_ids: List[int],
        dataset_root: str | Path,
        replay_fraction: float = 0.20,
        seed: int = 42,
        target_sr: int = SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES,
        snr_levels: Tuple[int, ...] = SNR_LEVELS_DB,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        dataset_root = Path(dataset_root)
        rng = random.Random(seed + 9999)

        all_clean: List[Path] = []
        all_noise:  List[Path] = []
        for cid in past_chunk_ids:
            all_clean.extend(get_chunk_paths(dataset_root, cid))
            all_noise.extend(get_noise_paths(dataset_root, cid))

        # Sample at the file level, not segment level
        n_replay = max(1, int(len(all_clean) * replay_fraction))
        rng.shuffle(all_clean)
        replay_clean = all_clean[:n_replay]

        logger.info(
            "ReplayDataset: %d/%d clean files from chunks %s (%.0f%% replay)",
            len(replay_clean), len(all_clean), past_chunk_ids, replay_fraction * 100,
        )

        self._inner = ChunkDataset(
            clean_paths=replay_clean,
            noise_paths=all_noise,
            snr_levels=snr_levels,
            seed=seed + 1234,
            target_sr=target_sr,
            segment_samples=segment_samples,
            normalize=normalize,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._inner[idx]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_stage_dataset(
    stage: int,
    dataset_root: str | Path,
    replay_fraction: float = 0.20,
    seed: int = 42,
    target_sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
    snr_levels: Tuple[int, ...] = SNR_LEVELS_DB,
    normalize: bool = True,
    debug_max_files: Optional[int] = None,
) -> Tuple[Dataset, Optional[WeightedRandomSampler]]:
    """Build training dataset (+ sampler) for a given stage.

    Returns (dataset, sampler) where sampler enforces 80/20 split for
    stages 2-4, and None for stage 1.
    """
    dataset_root  = Path(dataset_root)
    current_chunk = stage

    current_clean = get_chunk_paths(dataset_root, current_chunk)
    current_noise = get_noise_paths(dataset_root, current_chunk)

    if debug_max_files is not None:
        rng = random.Random(seed)
        rng.shuffle(current_clean)
        current_clean = current_clean[:debug_max_files]
        current_noise = current_noise[:max(1, debug_max_files // 4)]
        logger.info("[DEBUG] Limiting to %d clean files", len(current_clean))

    current_ds = ChunkDataset(
        clean_paths=current_clean,
        noise_paths=current_noise,
        snr_levels=snr_levels,
        seed=seed,
        target_sr=target_sr,
        segment_samples=segment_samples,
        normalize=normalize,
    )

    if stage == 1:
        logger.info("Stage 1: %d segments (no replay)", len(current_ds))
        return current_ds, None

    past_ids = list(range(1, stage))
    replay_ds = ReplayDataset(
        past_chunk_ids=past_ids,
        dataset_root=dataset_root,
        replay_fraction=replay_fraction,
        seed=seed,
        target_sr=target_sr,
        segment_samples=segment_samples,
        snr_levels=snr_levels,
        normalize=normalize,
    )

    combined = ConcatDataset([current_ds, replay_ds])

    n_current = len(current_ds)
    n_replay  = len(replay_ds)
    n_total   = n_current + n_replay

    w_current = (1.0 - replay_fraction) / n_current if n_current > 0 else 0.0
    w_replay  = replay_fraction         / n_replay  if n_replay  > 0 else 0.0

    weights = [w_current] * n_current + [w_replay] * n_replay
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n_total,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(
        "Stage %d: %d current + %d replay segments (%.0f%% / %.0f%%)",
        stage, n_current, n_replay,
        (1 - replay_fraction) * 100, replay_fraction * 100,
    )
    return combined, sampler


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="Dataset")
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    ds, sampler = build_stage_dataset(args.stage, args.dataset_root, debug_max_files=20)
    print(f"\nDataset length : {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        sample_noisy = sample["noisy"]
        sample_clean = sample["clean"]
        print("noisy shape:", sample_noisy.shape, " clean shape:", sample_clean.shape)
        sample_noisy = sample["noisy"]
        sample_clean = sample["clean"]
        print("noisy shape:", sample_noisy.shape, " clean shape:", sample_clean.shape)
    print("replay_sampler smoke test PASSED")
