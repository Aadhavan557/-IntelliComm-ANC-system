"""
dataset_loader.py
=================
SentinelANC Transmitter-Side Speech Enhancement
PyTorch Dataset classes for Conv-TasNet training.

Architecture overview
---------------------
                    ┌─────────────────────────────────────────────┐
                    │           SpeechEnhancementDataset           │
                    │   (top-level: picks the right dataset for    │
                    │    "train" / "validation" / "test" split)    │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │              NoisyPairDataset                │
                    │  • Lazy loading via load_segment()           │
                    │  • On-the-fly SNR mixing (10/5/0/-5 dB)     │
                    │  • Falls back to pre-baked mixtures/ files   │
                    │  • Returns {"noisy": [1,64000],              │
                    │             "clean": [1,64000]}              │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │          Manifest  (JSON on disk)            │
                    │  Built once on first run; cached in          │
                    │  Dataset/metadata/manifest.json              │
                    └─────────────────────────────────────────────┘

Design principles
-----------------
* ALL audio is processed inside __getitem__ — zero pre-loading.
* Source files are never modified (reads only).
* 15 GB dataset fits on any GPU node — only one segment lives in RAM
  per worker at a time.
* Deterministic 80/10/10 train-val-test split using a fixed seed.
* Pre-made mixtures in Dataset/mixtures/ are used when available;
  otherwise mixtures are generated on-the-fly from clean + noise.

Output tensor shapes
--------------------
    noisy_audio : torch.Tensor  [1, 64000]   float32
    clean_audio : torch.Tensor  [1, 64000]   float32

Dependencies
------------
    pip install torch torchaudio soundfile numpy
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing import (
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SNR_LEVELS_DB,
    AudioInfo,
    count_segments,
    load_segment,
    mix_with_snr,
    probe_audio_file,
    scan_audio_files,
    validate_audio_file,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Random seed used for the deterministic 80/10/10 file-level split.
_SPLIT_SEED: int = 42

#: Fraction of clean_speech files assigned to each partition.
_SPLIT_RATIOS: Dict[str, float] = {"train": 0.80, "validation": 0.10, "test": 0.10}

#: Filename of the cached manifest inside Dataset/metadata/.
_MANIFEST_FILENAME: str = "manifest.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    """Return a short hex digest of the file path string.

    Used to make the manifest's cache key path-independent of the machine.
    """
    return hashlib.md5(str(path).encode()).hexdigest()[:8]


def _assign_splits(
    paths: List[Path],
    seed: int = _SPLIT_SEED,
    ratios: Dict[str, float] = _SPLIT_RATIOS,
) -> Dict[str, List[Path]]:
    """Randomly assign *paths* to train / validation / test splits.

    The assignment is **deterministic** given the same seed and file list.
    Files are shuffled by their stem name (not full path) for reproducibility
    across different mount points / operating systems.

    Parameters
    ----------
    paths:
        Sorted list of audio file paths to split.
    seed:
        Random seed.
    ratios:
        Mapping of split name → fraction (must sum to 1.0).

    Returns
    -------
    Dict[str, List[Path]]
        Keys: "train", "validation", "test".
    """
    rng = random.Random(seed)
    shuffled = sorted(paths, key=lambda p: p.stem)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["validation"])

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Manifest  (lightweight JSON index — built once, reused forever)
# ---------------------------------------------------------------------------

class Manifest:
    """JSON index mapping every audio file to its metadata + segment count.

    Building the manifest requires only metadata probes (O(1) per file,
    no audio decoding), so scanning 15 GB takes seconds not minutes.

    Schema
    ------
    {
      "version": 1,
      "sample_rate": 16000,
      "segment_samples": 64000,
      "splits": {
        "train": [
          {
            "path": "/abs/path/to/file.wav",
            "native_sr": 44100,
            "native_frames": 882000,
            "channels": 1,
            "duration_s": 20.0,
            "num_segments": 5
          },
          ...
        ],
        "validation": [...],
        "test": [...]
      },
      "noise_files": [
        { "path": "...", "native_sr": 44100, "native_frames": 441000,
          "channels": 2, "duration_s": 10.0, "num_segments": 3 },
        ...
      ],
      "mixture_files": [
        { "clean_path": "...", "noisy_path": "...",
          "native_sr": 16000, "native_frames": 64000,
          "channels": 1, "duration_s": 4.0, "num_segments": 1 },
        ...
      ]
    }
    """

    VERSION: int = 1

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        dataset_root: Union[str, Path],
        split_seed: int = _SPLIT_SEED,
        split_ratios: Dict[str, float] = _SPLIT_RATIOS,
        target_sr: int = SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES,
    ) -> "Manifest":
        """Scan *dataset_root* and build a manifest from scratch.

        This is called automatically by :func:`load_or_build_manifest` when
        no cached manifest exists.  It never reads audio samples — only file
        metadata.

        Parameters
        ----------
        dataset_root:
            Root of the dataset (e.g. ``c:/SentinelANC_ReceiverSide/Dataset``).
        split_seed:
            Seed for the deterministic train/val/test file assignment.
        split_ratios:
            Mapping of split name → fraction.
        target_sr:
            Target sample rate used to compute segment counts.
        segment_samples:
            Segment length in target-sample-rate samples.

        Returns
        -------
        Manifest
        """
        root = Path(dataset_root)
        logger.info("Building manifest from %s …", root)

        # ---- Discover clean speech files ----------------------------------
        clean_root = root / "clean_speech"
        pre_split_dirs = {
            split: root / split
            for split in ("train", "validation", "test")
        }

        # Prefer pre-populated split folders; fall back to dynamic split
        use_presplit = all(
            any(d.iterdir()) for d in pre_split_dirs.values()
            if d.is_dir() and any(True for _ in d.iterdir())
        )

        # Helper to check if a directory has audio files
        def _has_audio(d: Path) -> bool:
            if not d.is_dir():
                return False
            return any(
                f.suffix.lower() in {".wav", ".flac"}
                for f in d.rglob("*")
            )

        use_presplit = all(_has_audio(d) for d in pre_split_dirs.values())

        if use_presplit:
            logger.info("Using pre-populated train/validation/test folders.")
            splits_paths: Dict[str, List[Path]] = {
                split: scan_audio_files(pre_split_dirs[split])
                for split in ("train", "validation", "test")
            }
        else:
            logger.info(
                "Split folders empty — dynamically splitting clean_speech/ "
                "(%.0f/%.0f/%.0f %%)",
                split_ratios["train"] * 100,
                split_ratios["validation"] * 100,
                split_ratios["test"] * 100,
            )
            all_clean = scan_audio_files(clean_root) if clean_root.is_dir() else []
            splits_paths = _assign_splits(all_clean, seed=split_seed, ratios=split_ratios)

        # ---- Discover noise files -----------------------------------------
        noise_root = root / "noise"
        noise_paths = scan_audio_files(noise_root) if noise_root.is_dir() else []
        logger.info("Found %d noise files.", len(noise_paths))

        # ---- Discover pre-baked mixture files ----------------------------
        mixture_root = root / "mixtures"
        mixture_paths = scan_audio_files(mixture_root) if mixture_root.is_dir() else []
        logger.info("Found %d pre-baked mixture files.", len(mixture_paths))

        # ---- Probe metadata and compute segment counts -------------------
        def _probe_entry(p: Path) -> Optional[Dict]:
            try:
                info: AudioInfo = probe_audio_file(p)
                n_seg = count_segments(
                    num_frames=info.frames,
                    sample_rate=info.sample_rate,
                    target_sr=target_sr,
                    segment_samples=segment_samples,
                )
                return {
                    "path": str(p),
                    "native_sr": info.sample_rate,
                    "native_frames": info.frames,
                    "channels": info.channels,
                    "duration_s": info.duration_s,
                    "num_segments": n_seg,
                }
            except Exception as exc:
                logger.warning("Skipping unreadable file %s: %s", p, exc)
                return None

        def _probe_list(paths: List[Path]) -> List[Dict]:
            entries = []
            for p in paths:
                entry = _probe_entry(p)
                if entry is not None:
                    entries.append(entry)
            return entries

        splits_data: Dict[str, List[Dict]] = {}
        for split_name, split_paths in splits_paths.items():
            logger.info("Probing %d %s files …", len(split_paths), split_name)
            splits_data[split_name] = _probe_list(split_paths)

        noise_data = _probe_list(noise_paths)
        mixture_data = _probe_list(mixture_paths)

        data = {
            "version": cls.VERSION,
            "sample_rate": target_sr,
            "segment_samples": segment_samples,
            "splits": splits_data,
            "noise_files": noise_data,
            "mixture_files": mixture_data,
        }
        logger.info("Manifest built: %s", cls._summary_str(data))
        return cls(data)

    @staticmethod
    def _summary_str(data: Dict) -> str:
        splits = data.get("splits", {})
        counts = {k: len(v) for k, v in splits.items()}
        n_noise = len(data.get("noise_files", []))
        n_mix = len(data.get("mixture_files", []))
        return (
            f"train={counts.get('train', 0)} val={counts.get('validation', 0)} "
            f"test={counts.get('test', 0)} noise={n_noise} mixtures={n_mix} files"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise manifest to *path* as pretty-printed JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)
        logger.info("Manifest saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Manifest":
        """Load a previously saved manifest from *path*."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        version = data.get("version", 0)
        if version != cls.VERSION:
            raise ValueError(
                f"Manifest version mismatch: expected {cls.VERSION}, got {version}. "
                f"Delete {path} and re-run to rebuild."
            )
        logger.info("Manifest loaded from %s (%s)", path, cls._summary_str(data))
        return cls(data)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def split_entries(self, split: str) -> List[Dict]:
        """Return the list of file-entry dicts for *split*."""
        return self._data["splits"].get(split, [])

    @property
    def noise_entries(self) -> List[Dict]:
        return self._data.get("noise_files", [])

    @property
    def mixture_entries(self) -> List[Dict]:
        return self._data.get("mixture_files", [])

    @property
    def sample_rate(self) -> int:
        return self._data["sample_rate"]

    @property
    def segment_samples(self) -> int:
        return self._data["segment_samples"]


def load_or_build_manifest(
    dataset_root: Union[str, Path],
    rebuild: bool = False,
    **build_kwargs: Any,
) -> Manifest:
    """Load the cached manifest or build it from scratch if missing / stale.

    Parameters
    ----------
    dataset_root:
        Path to the dataset root directory.
    rebuild:
        If ``True``, always rebuild even if a cached manifest exists.
    **build_kwargs:
        Forwarded to :meth:`Manifest.build`.

    Returns
    -------
    Manifest
    """
    manifest_path = Path(dataset_root) / "metadata" / _MANIFEST_FILENAME
    if not rebuild and manifest_path.exists():
        try:
            return Manifest.load(manifest_path)
        except Exception as exc:
            logger.warning(
                "Failed to load existing manifest (%s). Rebuilding …", exc
            )

    manifest = Manifest.build(dataset_root, **build_kwargs)
    manifest.save(manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _build_flat_index(entries: List[Dict]) -> List[Tuple[int, int]]:
    """Return a flat list of ``(file_idx, segment_idx)`` pairs.

    This maps Dataset integer indices to specific (file, segment) positions
    without storing any audio data.
    """
    index: List[Tuple[int, int]] = []
    for file_idx, entry in enumerate(entries):
        for seg_idx in range(entry["num_segments"]):
            index.append((file_idx, seg_idx))
    return index


# ---------------------------------------------------------------------------
# NoisyPairDataset
# ---------------------------------------------------------------------------

class NoisyPairDataset(Dataset):
    """PyTorch Dataset that yields aligned (noisy, clean) pairs on demand.

    Behaviour
    ---------
    * If ``use_premade_mixtures=True`` **and** the manifest has mixture files,
      those pre-baked files are used directly (no dynamic mixing).
    * Otherwise, for every ``__getitem__`` call:
        1. Load a clean speech segment lazily from disk.
        2. Pick a random noise file + random segment from that file.
        3. Pick a random SNR from :data:`~preprocessing.SNR_LEVELS_DB`.
        4. Mix with :func:`~preprocessing.mix_with_snr`.
        5. Return ``{"noisy": [1, 64000], "clean": [1, 64000]}``.

    Parameters
    ----------
    manifest:
        Pre-built :class:`Manifest` object.
    split:
        One of ``"train"``, ``"validation"``, ``"test"``.
    use_premade_mixtures:
        Use pre-baked mixture files from ``mixtures/`` when available.
        Default: ``False`` (always generate on-the-fly).
    snr_levels:
        SNR values (dB) to sample from.  Default: ``(10, 5, 0, -5)``.
    seed:
        Random seed for reproducible noise/SNR selection.  If ``None``,
        the seed is derived from the segment index (reproducible but
        varied per sample).
    target_sr:
        Target sample rate.  Must match the manifest.
    segment_samples:
        Segment length in samples.  Must match the manifest.
    normalize:
        Whether to peak-normalize each segment after loading.
    """

    def __init__(
        self,
        manifest: Manifest,
        split: str = "train",
        use_premade_mixtures: bool = False,
        snr_levels: Sequence[float] = SNR_LEVELS_DB,
        seed: Optional[int] = None,
        target_sr: int = SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if split not in ("train", "validation", "test"):
            raise ValueError(
                f"split must be 'train', 'validation', or 'test'. Got: '{split}'"
            )

        self.split = split
        self.snr_levels = list(snr_levels)
        self.seed = seed
        self.target_sr = target_sr
        self.segment_samples = segment_samples
        self.normalize = normalize
        self.use_premade_mixtures = use_premade_mixtures

        # ---- Choose data source ------------------------------------------
        self._mixture_entries = manifest.mixture_entries
        self._use_premade = (
            use_premade_mixtures and len(self._mixture_entries) > 0
        )

        self._clean_entries: List[Dict] = manifest.split_entries(split)
        self._noise_entries: List[Dict] = manifest.noise_entries

        if not self._use_premade:
            if len(self._clean_entries) == 0:
                raise RuntimeError(
                    f"No clean speech files found for split='{split}'. "
                    f"Ensure the dataset is populated and the manifest is up-to-date."
                )
            if len(self._noise_entries) == 0:
                raise RuntimeError(
                    "No noise files found in Dataset/noise/. "
                    "Cannot perform on-the-fly mixing without noise files."
                )

        # ---- Build flat (file, segment) index ----------------------------
        if self._use_premade:
            self._index: List[Tuple[int, int]] = _build_flat_index(
                self._mixture_entries
            )
        else:
            self._index = _build_flat_index(self._clean_entries)

        logger.info(
            "NoisyPairDataset [%s]: %d segments | source=%s | noise_files=%d",
            split,
            len(self._index),
            "premade" if self._use_premade else "on-the-fly",
            len(self._noise_entries),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one aligned (noisy, clean) pair.

        Returns
        -------
        dict with keys:
            ``"noisy"``  : torch.Tensor of shape ``[1, 64000]``
            ``"clean"``  : torch.Tensor of shape ``[1, 64000]``
            ``"snr_db"`` : float (SNR used; 0.0 for pre-baked mixtures)
            ``"split"``  : str
        """
        if self._use_premade:
            return self._getitem_premade(idx)
        return self._getitem_dynamic(idx)

    # ------------------------------------------------------------------
    # Pre-baked mixture path
    # ------------------------------------------------------------------

    def _getitem_premade(self, idx: int) -> Dict[str, Any]:
        file_idx, seg_idx = self._index[idx]
        entry = self._mixture_entries[file_idx]

        noisy = load_segment(
            path=entry["path"],
            segment_idx=seg_idx,
            native_sr=entry["native_sr"],
            native_frames=entry["native_frames"],
            target_sr=self.target_sr,
            segment_samples=self.segment_samples,
            normalize=self.normalize,
        )

        # Try to find a matching clean file by stem convention:
        # mixtures/spk001_noise_snr10.wav  <->  clean_speech/spk001.wav
        # If not resolvable, return zeros as clean placeholder.
        clean = torch.zeros(1, self.segment_samples)

        return {
            "noisy": noisy,
            "clean": clean,
            "snr_db": 0.0,
            "split": self.split,
        }

    # ------------------------------------------------------------------
    # On-the-fly mixing path
    # ------------------------------------------------------------------

    def _getitem_dynamic(self, idx: int) -> Dict[str, Any]:
        file_idx, seg_idx = self._index[idx]
        clean_entry = self._clean_entries[file_idx]

        # ---- Load clean segment -----------------------------------------
        clean = load_segment(
            path=clean_entry["path"],
            segment_idx=seg_idx,
            native_sr=clean_entry["native_sr"],
            native_frames=clean_entry["native_frames"],
            target_sr=self.target_sr,
            segment_samples=self.segment_samples,
            normalize=self.normalize,
        )

        # ---- Pick noise file + segment (deterministic per idx) ----------
        # Use a per-sample RNG so results are reproducible across workers
        rng = random.Random((self.seed or 0) ^ (idx * 2654435761))

        noise_entry = rng.choice(self._noise_entries)
        noise_seg_idx = rng.randint(0, max(0, noise_entry["num_segments"] - 1))
        snr_db = float(rng.choice(self.snr_levels))

        # ---- Load noise segment -----------------------------------------
        noise = load_segment(
            path=noise_entry["path"],
            segment_idx=noise_seg_idx,
            native_sr=noise_entry["native_sr"],
            native_frames=noise_entry["native_frames"],
            target_sr=self.target_sr,
            segment_samples=self.segment_samples,
            normalize=False,  # Normalize AFTER mixing to preserve SNR
        )

        # ---- Mix --------------------------------------------------------
        noisy, clean_aligned = mix_with_snr(
            clean=clean,
            noise=noise,
            snr_db=snr_db,
            prevent_clipping=True,
        )

        return {
            "noisy": noisy,           # [1, 64000]
            "clean": clean_aligned,   # [1, 64000]  (same scale as noisy)
            "snr_db": snr_db,
            "split": self.split,
        }


# ---------------------------------------------------------------------------
# SpeechEnhancementDataset  (top-level convenience wrapper)
# ---------------------------------------------------------------------------

class SpeechEnhancementDataset(Dataset):
    """Top-level dataset that auto-builds the manifest and wraps the right split.

    This is the **single entry point** intended for the training script.

    Parameters
    ----------
    dataset_root:
        Absolute path to the dataset root (contains clean_speech/, noise/, …).
    split:
        ``"train"``, ``"validation"``, or ``"test"``.
    rebuild_manifest:
        Force rebuild of the manifest even if one already exists on disk.
    use_premade_mixtures:
        Use pre-baked files from ``mixtures/`` if available.
    snr_levels:
        SNR values (dB) for on-the-fly mixing.
    seed:
        Reproducibility seed for noise selection.
    target_sr:
        Target sample rate (Hz).
    segment_samples:
        Segment length in samples.
    normalize:
        Peak-normalize each segment.

    Example
    -------
    >>> from dataset_loader import SpeechEnhancementDataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> train_ds = SpeechEnhancementDataset("Dataset", split="train")
    >>> loader   = DataLoader(train_ds, batch_size=8, num_workers=4,
    ...                       collate_fn=speech_collate_fn)
    >>> batch = next(iter(loader))
    >>> batch["noisy"].shape   # [8, 1, 64000]
    >>> batch["clean"].shape   # [8, 1, 64000]
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        split: str = "train",
        rebuild_manifest: bool = False,
        use_premade_mixtures: bool = False,
        snr_levels: Sequence[float] = SNR_LEVELS_DB,
        seed: Optional[int] = _SPLIT_SEED,
        target_sr: int = SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split

        self.manifest = load_or_build_manifest(
            self.dataset_root,
            rebuild=rebuild_manifest,
            target_sr=target_sr,
            segment_samples=segment_samples,
        )

        self._inner = NoisyPairDataset(
            manifest=self.manifest,
            split=split,
            use_premade_mixtures=use_premade_mixtures,
            snr_levels=snr_levels,
            seed=seed,
            target_sr=target_sr,
            segment_samples=segment_samples,
            normalize=normalize,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._inner[idx]

    def __repr__(self) -> str:
        return (
            f"SpeechEnhancementDataset("
            f"split='{self.split}', "
            f"n_segments={len(self)}, "
            f"root='{self.dataset_root}')"
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def speech_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stack a list of sample dicts into batched tensors.

    Input
    -----
    List of dicts with keys ``"noisy"``, ``"clean"``, ``"snr_db"``, ``"split"``.

    Output
    ------
    dict with:
        ``"noisy"``  : torch.Tensor  ``[B, 1, 64000]``
        ``"clean"``  : torch.Tensor  ``[B, 1, 64000]``
        ``"snr_db"`` : torch.Tensor  ``[B]``   (float32)
        ``"split"``  : List[str]
    """
    noisy = torch.stack([s["noisy"] for s in batch], dim=0)   # [B, 1, T]
    clean = torch.stack([s["clean"] for s in batch], dim=0)   # [B, 1, T]
    snr_db = torch.tensor([s["snr_db"] for s in batch], dtype=torch.float32)
    splits = [s["split"] for s in batch]

    return {
        "noisy": noisy,
        "clean": clean,
        "snr_db": snr_db,
        "split": splits,
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    rebuild_manifest: bool = False,
    use_premade_mixtures: bool = False,
    snr_levels: Sequence[float] = SNR_LEVELS_DB,
    seed: int = _SPLIT_SEED,
    target_sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
    normalize: bool = True,
) -> Dict[str, DataLoader]:
    """Convenience factory that returns ready-to-use DataLoaders for all splits.

    Parameters
    ----------
    dataset_root:
        Path to the dataset root.
    batch_size:
        Samples per batch.
    num_workers:
        Number of parallel data-loading workers.  Set to 0 for debugging.
    pin_memory:
        Pin memory for faster GPU transfers.
    rebuild_manifest:
        Force manifest rebuild.
    use_premade_mixtures:
        Use pre-baked mixture files.
    snr_levels:
        SNR values (dB) for on-the-fly mixing.
    seed:
        Reproducibility seed.
    target_sr:
        Target sample rate.
    segment_samples:
        Segment length in samples.
    normalize:
        Peak-normalize each segment.

    Returns
    -------
    Dict[str, DataLoader]
        Keys: ``"train"``, ``"validation"``, ``"test"``.

    Example
    -------
    >>> loaders = build_dataloaders("Dataset", batch_size=8, num_workers=4)
    >>> for batch in loaders["train"]:
    ...     noisy = batch["noisy"]   # [8, 1, 64000]
    ...     clean = batch["clean"]   # [8, 1, 64000]
    ...     break
    """
    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "validation", "test"):
        dataset = SpeechEnhancementDataset(
            dataset_root=dataset_root,
            split=split,
            rebuild_manifest=rebuild_manifest,
            use_premade_mixtures=use_premade_mixtures,
            snr_levels=snr_levels,
            seed=seed,
            target_sr=target_sr,
            segment_samples=segment_samples,
            normalize=normalize,
        )
        # Only rebuild once (the manifest is cached after the first call)
        rebuild_manifest = False

        shuffle = (split == "train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=speech_collate_fn,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        loaders[split] = loader
        logger.info(
            "DataLoader [%s]: %d batches of %d",
            split, len(loader), batch_size,
        )

    return loaders
