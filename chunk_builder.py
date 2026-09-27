"""
chunk_builder.py
================
IntelliComm - AI-Based Speech Enhancement for Military/Field Communication

Splits the TRAINING portion of the dataset into 4 balanced chunks.
NO audio files are copied or moved. The chunks are stored as a JSON
index at Dataset/metadata/chunks.json.

Usage
-----
    python chunk_builder.py --dataset_root Dataset
    python chunk_builder.py --dataset_root Dataset --num_chunks 4 --seed 42 --rebuild

Chunk guarantee
---------------
Each chunk contains a diverse mix of:
  - Clean speech from multiple speakers
  - All SNR conditions (+10 / +5 / 0 / -5 dB) applied at training time
  - Noise from every noise category (stationary / nonstationary / impulsive)

This is achieved via stratified assignment: clean_speech files are first
grouped by speaker/source directory, then distributed round-robin across
chunks so that no chunk is dominated by a single speaker or noise type.

Output (chunks.json schema)
---------------------------
{
  "version": 1,
  "num_chunks": 4,
  "seed": 42,
  "dataset_root": "Dataset",
  "chunks": {
    "1": ["abs/path/to/file1.wav", "abs/path/to/file2.wav", ...],
    "2": [...],
    "3": [...],
    "4": [...]
  },
  "noise_chunks": {
    "1": ["abs/path/to/noise1.wav", ...],
    "2": [...],
    ...
  },
  "stats": {
    "1": {"num_clean": 250, "num_noise": 80, "approx_hours": 16.7},
    ...
  }
}

Important: validation and test files are NEVER placed in any chunk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise category detection
# ---------------------------------------------------------------------------

# Known top-level noise category folder names in Dataset/noise/
_NOISE_CATEGORIES = [
    "impulsive",
    "nonstationary",
    "stationary",
    "unclassified",
]

_AUDIO_EXTS = {".wav", ".flac"}


def _scan_audio(directory: Path) -> List[Path]:
    """Recursively find all WAV/FLAC files under *directory*."""
    files = []
    for ext in _AUDIO_EXTS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def _noise_category(path: Path) -> str:
    """Return the top-level noise category of a noise file path."""
    parts = [p.lower() for p in path.parts]
    for cat in _NOISE_CATEGORIES:
        if cat in parts:
            return cat
    return "unclassified"


def _speaker_id(clean_path: Path) -> str:
    """Return a speaker/source identifier from a clean speech file path.

    For LibriSpeech: the speaker ID is the first numeric directory component.
    For VCTK: the speaker ID is the 'pXXX' directory.
    Falls back to the parent directory name.
    """
    parts = clean_path.parts
    # Walk up until we find a numeric or VCTK-style component
    for part in reversed(parts[:-1]):
        if part.isdigit():
            return part
        if part.lower().startswith("p") and part[1:].isdigit():
            return part
    return clean_path.parent.name


# ---------------------------------------------------------------------------
# Stratified chunk assignment
# ---------------------------------------------------------------------------

def _stratified_split(
    files: List[Path],
    num_chunks: int,
    seed: int,
    key_fn,
) -> Dict[int, List[Path]]:
    """Distribute *files* across *num_chunks* chunks via stratified round-robin.

    Files are first grouped by ``key_fn(path)``, then within each group
    they are shuffled and assigned to chunks in round-robin order.  This
    ensures every chunk receives files from every group.

    Parameters
    ----------
    files     : list of Path objects to distribute
    num_chunks: target number of chunks
    seed      : RNG seed for reproducibility
    key_fn    : callable(Path) -> str  grouping key

    Returns
    -------
    Dict[int, List[Path]]
        1-indexed mapping of chunk_id -> list of file paths.
    """
    rng = random.Random(seed)

    # Group files by key
    groups: Dict[str, List[Path]] = {}
    for f in files:
        k = key_fn(f)
        groups.setdefault(k, []).append(f)

    # Shuffle within each group for reproducibility
    for k in groups:
        rng.shuffle(groups[k])

    chunks: Dict[int, List[Path]] = {i + 1: [] for i in range(num_chunks)}
    chunk_idx = 0

    # Round-robin across groups
    for key in sorted(groups.keys()):
        for f in groups[key]:
            chunks[(chunk_idx % num_chunks) + 1].append(f)
            chunk_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Duration estimation (from filename if available; else default 4 s)
# ---------------------------------------------------------------------------

def _approx_hours(paths: List[Path], default_seg_s: float = 4.0) -> float:
    """Estimate total duration in hours (no audio decoded)."""
    # Each preprocessed segment file is exactly 4 s (or close to it)
    total_s = len(paths) * default_seg_s
    return total_s / 3600


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_chunks(
    dataset_root: str | Path,
    num_chunks: int = 4,
    seed: int = 42,
    rebuild: bool = False,
) -> Dict:
    """Build and save the chunk index.

    Parameters
    ----------
    dataset_root : path to the Dataset directory
    num_chunks   : number of chunks to create (default 4)
    seed         : RNG seed (default 42)
    rebuild      : overwrite existing chunks.json if True

    Returns
    -------
    dict  The full chunk index (same structure as chunks.json).
    """
    root        = Path(dataset_root).resolve()
    meta_dir    = root / "metadata"
    chunks_path = meta_dir / "chunks.json"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if chunks_path.exists() and not rebuild:
        logger.info("chunks.json already exists. Use --rebuild to overwrite.")
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Building chunk index from %s ...", root)

    # ---- Discover validation & test files to EXCLUDE ----------------------
    # Read manifest to identify val/test files
    manifest_path = meta_dir / "manifest.json"
    excluded_paths: set[str] = set()

    if manifest_path.exists():
        logger.info("Reading manifest to exclude val/test files ...")
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        for split in ("validation", "test"):
            for entry in manifest.get("splits", {}).get(split, []):
                excluded_paths.add(str(Path(entry["path"]).resolve()))
        logger.info("  Excluding %d val/test files", len(excluded_paths))
    else:
        logger.warning(
            "manifest.json not found. Run dataset_loader.py first to build it, "
            "or validation/test files may leak into training chunks."
        )

    # ---- Collect clean speech train files ---------------------------------
    clean_root = root / "clean_speech"
    if not clean_root.is_dir():
        raise FileNotFoundError(f"clean_speech directory not found: {clean_root}")

    all_clean = _scan_audio(clean_root)
    train_clean = [
        f for f in all_clean
        if str(f.resolve()) not in excluded_paths
    ]
    logger.info("  Found %d clean train files (excluded %d val/test)",
                len(train_clean), len(all_clean) - len(train_clean))

    if len(train_clean) < num_chunks:
        raise ValueError(
            f"Only {len(train_clean)} training files found, but {num_chunks} chunks requested."
        )

    # ---- Collect noise files ----------------------------------------------
    noise_root = root / "noise"
    all_noise: List[Path] = []
    if noise_root.is_dir():
        all_noise = _scan_audio(noise_root)
    logger.info("  Found %d noise files", len(all_noise))

    # ---- Stratified split: clean speech -----------------------------------
    logger.info("Splitting %d clean files into %d chunks (seed=%d) ...",
                len(train_clean), num_chunks, seed)
    clean_chunks = _stratified_split(
        train_clean, num_chunks, seed, key_fn=_speaker_id
    )

    # ---- Stratified split: noise ------------------------------------------
    noise_chunks: Dict[int, List[Path]] = {i + 1: [] for i in range(num_chunks)}
    if all_noise:
        noise_chunks = _stratified_split(
            all_noise, num_chunks, seed, key_fn=_noise_category
        )

    # ---- Compute stats ----------------------------------------------------
    stats: Dict[str, Dict] = {}
    for i in range(1, num_chunks + 1):
        stats[str(i)] = {
            "num_clean":    len(clean_chunks[i]),
            "num_noise":    len(noise_chunks[i]),
            "approx_hours": round(_approx_hours(clean_chunks[i]), 2),
        }

    # ---- Build JSON -------------------------------------------------------
    index = {
        "version":      1,
        "num_chunks":   num_chunks,
        "seed":         seed,
        "dataset_root": str(root),
        "chunks": {
            str(k): [str(p) for p in v]
            for k, v in clean_chunks.items()
        },
        "noise_chunks": {
            str(k): [str(p) for p in v]
            for k, v in noise_chunks.items()
        },
        "stats": stats,
    }

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    logger.info("chunks.json saved to %s", chunks_path)

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Chunk Summary")
    print("=" * 60)
    for i in range(1, num_chunks + 1):
        s = stats[str(i)]
        print(
            f"  Chunk {i}: {s['num_clean']:>5} clean files | "
            f"{s['num_noise']:>4} noise files | "
            f"~{s['approx_hours']:.1f} h"
        )
    print("=" * 60)

    return index


def load_chunks(dataset_root: str | Path) -> Dict:
    """Load the existing chunks.json index.

    Raises
    ------
    FileNotFoundError if chunks.json has not been built yet.
    """
    path = Path(dataset_root) / "metadata" / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {path}. "
            "Run: python chunk_builder.py --dataset_root <path>"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chunk_paths(
    dataset_root: str | Path,
    chunk_id: int,
) -> List[Path]:
    """Return the list of clean speech file Paths for a given chunk ID (1-4)."""
    index = load_chunks(dataset_root)
    key   = str(chunk_id)
    if key not in index["chunks"]:
        raise ValueError(
            f"Chunk {chunk_id} not found. Available: {list(index['chunks'].keys())}"
        )
    return [Path(p) for p in index["chunks"][key]]


def get_noise_paths(
    dataset_root: str | Path,
    chunk_id: int,
) -> List[Path]:
    """Return the list of noise file Paths assigned to *chunk_id*."""
    index = load_chunks(dataset_root)
    key   = str(chunk_id)
    return [Path(p) for p in index.get("noise_chunks", {}).get(key, [])]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build chunk index for IntelliComm staged training."
    )
    parser.add_argument(
        "--dataset_root", type=str, default="Dataset",
        help="Path to the Dataset directory (default: Dataset)"
    )
    parser.add_argument(
        "--num_chunks", type=int, default=4,
        help="Number of chunks to create (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild chunks.json even if it already exists"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        build_chunks(
            dataset_root=args.dataset_root,
            num_chunks=args.num_chunks,
            seed=args.seed,
            rebuild=args.rebuild,
        )
    except Exception as exc:
        logger.error("chunk_builder failed: %s", exc)
        sys.exit(1)
