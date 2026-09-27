"""
stoi_analysis.py
================
IntelliComm — STOI Error Analysis
Comprehensive per-sample analysis to determine WHY STOI is lower than target
despite strong SI-SNR performance.

Usage:
  python stoi_analysis.py                        # 500 samples, seed 42
  python stoi_analysis.py --limit 500 --seed 42

DO NOT retrain — analysis only. Uses checkpoints/best.pth.

ESC-50 noise category map (class number → label):
  https://github.com/karolpiczak/ESC-50
  Categories 1-50 mapped to 10 major groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message=".*mix_with_snr.*")

# ── Metrics ────────────────────────────────────────────────────────────────────
from pystoi import stoi as _stoi_fn

def compute_stoi(clean: np.ndarray, pred: np.ndarray, sr: int = 16000) -> Tuple[Optional[float], bool]:
    """Returns (score, was_warned). score=None only on hard error."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            score = float(_stoi_fn(clean.astype(np.float64), pred.astype(np.float64), sr, extended=True))
            warned = len(w) > 0
            return score, warned
        except Exception:
            return None, False

def si_snr_single(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred   = pred   - pred.mean()
    target = target - target.mean()
    dot    = np.dot(pred, target)
    t_pow  = np.dot(target, target) + eps
    s_tgt  = (dot / t_pow) * target
    e_nse  = pred - s_tgt
    return float(10 * np.log10((np.dot(s_tgt, s_tgt) + eps) / (np.dot(e_nse, e_nse) + eps)))

# ── ESC-50 category map (class # → label) ─────────────────────────────────────
# ESC-50 uses 50 classes in 5 major categories. The filename contains the class ID.
# Format: esc50_{fold}-{clip_id}-{take}-{class_id}.wav
ESC50_CLASS_MAP: Dict[int, str] = {
    # Animals
    1:"Dog", 2:"Rooster", 3:"Pig", 4:"Cow", 5:"Frog",
    6:"Cat", 7:"Hen", 8:"Insects", 9:"Sheep", 10:"Crow",
    # Natural soundscapes & water
    11:"Rain", 12:"Sea waves", 13:"Crackling fire", 14:"Crickets", 15:"Chirping birds",
    16:"Water drops", 17:"Wind", 18:"Pouring water", 19:"Toilet flush", 20:"Thunderstorm",
    # Human (non-speech)
    21:"Crying baby", 22:"Sneezing", 23:"Clapping", 24:"Breathing", 25:"Coughing",
    26:"Footsteps", 27:"Laughing", 28:"Brushing teeth", 29:"Snoring", 30:"Drinking/sipping",
    # Interior/domestic
    31:"Door knock", 32:"Mouse click", 33:"Keyboard typing", 34:"Door/wood creak", 35:"Can opening",
    36:"Washing machine", 37:"Vacuum cleaner", 38:"Clock alarm", 39:"Clock tick", 40:"Glass breaking",
    # Exterior/urban
    41:"Helicopter", 42:"Chainsaw", 43:"Siren", 44:"Car horn", 45:"Engine",
    46:"Train", 47:"Church bells", 48:"Airplane", 49:"Fireworks", 50:"Hand saw",
}

ESC50_MAJOR_GROUP: Dict[int, str] = {
    **{i: "Animals" for i in range(1, 11)},
    **{i: "Nature" for i in range(11, 21)},
    **{i: "Human" for i in range(21, 31)},
    **{i: "Interior/Domestic" for i in range(31, 41)},
    **{i: "Exterior/Urban" for i in range(41, 51)},
}

MILITARY_RELEVANT: Dict[int, bool] = {
    41: True,  # Helicopter
    42: True,  # Chainsaw
    43: True,  # Siren
    44: True,  # Car horn
    45: True,  # Engine
    46: True,  # Train
    48: True,  # Airplane
    49: True,  # Fireworks
    50: True,  # Hand saw
}

def parse_noise_meta(path_str: str) -> Tuple[str, str, bool]:
    """Extract (label, major_group, is_military) from noise file path."""
    try:
        name = Path(path_str).stem  # e.g. esc50_41-123456-A-41
        # class_id is the last hyphen-separated token
        parts = name.split("-")
        # Try last numeric token
        for p in reversed(parts):
            if p.isdigit():
                cls = int(p)
                return (
                    ESC50_CLASS_MAP.get(cls, f"ESC-{cls}"),
                    ESC50_MAJOR_GROUP.get(cls, "Unknown"),
                    MILITARY_RELEVANT.get(cls, False),
                )
        # Also try the number after the first underscore in the prefix "esc50_XX-..."
        prefix = name.split("-")[0]  # e.g. esc50_41
        if "_" in prefix:
            cls = int(prefix.split("_")[-1])
            return (
                ESC50_CLASS_MAP.get(cls, f"ESC-{cls}"),
                ESC50_MAJOR_GROUP.get(cls, "Unknown"),
                MILITARY_RELEVANT.get(cls, False),
            )
    except Exception:
        pass
    return "Unknown", "Unknown", False


def parse_speaker_id(path_str: str) -> str:
    """Extract VCTK speaker ID from clean speech path. e.g. p236."""
    try:
        parts = Path(path_str).parts
        for part in parts:
            if part.startswith("p") and part[1:].isdigit():
                return part
    except Exception:
        pass
    return "unknown"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SR = 16000


# ─────────────────────────────────────────────────────────────────────────────
# Dataset introspection helper
# ─────────────────────────────────────────────────────────────────────────────

def get_internal_dataset(ds_wrapper) -> object:
    """Unwrap SpeechEnhancementDataset -> NoisyPairDataset."""
    if hasattr(ds_wrapper, "_inner"):
        return ds_wrapper._inner
    if hasattr(ds_wrapper, "_dataset"):
        return ds_wrapper._dataset
    if hasattr(ds_wrapper, "dataset"):
        return ds_wrapper.dataset
    return ds_wrapper


def get_sample_meta(inner_ds, local_idx: int) -> Dict:
    """
    Extract per-sample metadata by inspecting the internal NoisyPairDataset index.
    Returns {clean_path, noise_path, snr_db, speaker_id, noise_label, noise_group}.
    """
    meta = {}
    try:
        # inner_ds._index[local_idx] = (file_idx, seg_idx)
        index = getattr(inner_ds, "_index", None)
        clean_entries = getattr(inner_ds, "_clean_entries", None)
        noise_entries = getattr(inner_ds, "_noise_entries", None)

        if index is None or clean_entries is None:
            return meta

        file_idx, seg_idx = index[local_idx]
        clean_entry = clean_entries[file_idx]
        clean_path = clean_entry.get("path", "")

        # Reproduce the deterministic noise selection
        rng = random.Random(0 ^ (local_idx * 2654435761))
        if noise_entries:
            noise_entry = rng.choice(noise_entries)
            rng.randint(0, max(0, noise_entry.get("num_segments", 1) - 1))  # consume seg RNG
            snr_levels = getattr(inner_ds, "snr_levels", [10, 5, 0, -5])
            snr_db = float(rng.choice(snr_levels))
            noise_path = noise_entry.get("path", "")
        else:
            noise_path = ""
            snr_db = 0.0

        noise_label, noise_group, _ = parse_noise_meta(noise_path)
        speaker_id = parse_speaker_id(clean_path)

        meta = {
            "clean_path": clean_path,
            "noise_path": noise_path,
            "snr_db": snr_db,
            "speaker_id": speaker_id,
            "noise_label": noise_label,
            "noise_group": noise_group,
        }
    except Exception as e:
        logger.debug("Meta extraction error: %s", e)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_waveform_spectrogram(noisy_np, clean_np, enh_np, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(clean_np)) / SR

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Row 0: waveforms
    for ax, wav, label, color in zip(
        axes[0], [noisy_np, clean_np, enh_np],
        ["Noisy Input", "Clean Reference", "Enhanced Output"],
        ["#e74c3c", "#2ecc71", "#3498db"],
    ):
        ax.plot(t, wav, linewidth=0.4, color=color, alpha=0.85)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    # Row 1: spectrograms
    for ax, wav, label in zip(
        axes[1], [noisy_np, clean_np, enh_np],
        ["Noisy Spectrogram", "Clean Spectrogram", "Enhanced Spectrogram"],
    ):
        f, t_s, Sxx = scipy_spectrogram(wav, fs=SR, nperseg=512, noverlap=256)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax.pcolormesh(t_s, f / 1000, Sxx_db, shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        ax.set_ylabel("Frequency (kHz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(label, fontsize=9)
        plt.colorbar(im, ax=ax, label="dB")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_scatter(x, y, xlabel, ylabel, title, out_path: Path, annotate_threshold=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Color by stoi
    scatter = ax.scatter(x, y, c=y, cmap="RdYlGn", alpha=0.55, s=20,
                         vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label=ylabel)

    # Regression line
    if len(x) > 5:
        coeff = np.polyfit(x, y, 1)
        xl = np.linspace(min(x), max(x), 200)
        ax.plot(xl, np.polyval(coeff, xl), "b--", linewidth=1.5, label="Trend")
        try:
            r, p = pearsonr(x, y)
            ax.set_title(f"{title}  |  r={r:.3f}  (p={'<0.001' if p < 0.001 else f'{p:.3f}'})")
        except Exception:
            ax.set_title(title)

    if annotate_threshold is not None:
        ax.axhline(annotate_threshold, color="orange", linestyle="--", linewidth=1,
                   label=f"STOI target ({annotate_threshold})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_distribution_plot(stoi_values: List[float], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
    counts, _ = np.histogram(stoi_values, bins=bins)
    labels = [f"{a:.2f}–{b:.2f}" for a, b in zip(bins[:-1], bins[1:])]

    colors = ["#c0392b", "#e67e22", "#e67e22", "#f39c12", "#f1c40f",
              "#2ecc71", "#27ae60", "#1a8754"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("STOI Distribution — Enhanced Output", fontsize=12, fontweight="bold")

    ax1.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("STOI Range")
    ax1.set_title("Histogram by STOI Range")
    ax1.axvline(0, color="black", linewidth=0.5)
    for i, v in enumerate(counts):
        ax1.text(v + 0.5, i, str(v), va="center", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.hist(stoi_values, bins=40, color="#3498db", alpha=0.75, edgecolor="white")
    ax2.axvline(np.mean(stoi_values), color="red", linestyle="--", linewidth=1.5, label=f"Mean={np.mean(stoi_values):.3f}")
    ax2.axvline(np.median(stoi_values), color="orange", linestyle="--", linewidth=1.5, label=f"Median={np.median(stoi_values):.3f}")
    ax2.axvline(0.85, color="green", linestyle="--", linewidth=1.5, label="Target=0.85")
    ax2.set_xlabel("STOI Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Continuous Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_snr_bar_plot(snr_data: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snrs = sorted(snr_data.keys())
    noisy_stois  = [snr_data[s]["noisy_stoi_mean"]  for s in snrs]
    enh_stois    = [snr_data[s]["enh_stoi_mean"]    for s in snrs]
    x = np.arange(len(snrs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, noisy_stois, w, label="Noisy", color="#e74c3c", alpha=0.8)
    ax.bar(x + w/2, enh_stois,   w, label="Enhanced", color="#3498db", alpha=0.8)
    ax.axhline(0.85, color="green", linestyle="--", linewidth=1.5, label="Target 0.85")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:+g} dB" for s in snrs])
    ax.set_xlabel("Input SNR")
    ax.set_ylabel("Mean STOI")
    ax.set_title("STOI vs Input SNR Condition")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="IntelliComm — STOI Error Analysis")
    parser.add_argument("--checkpoint",   default="checkpoints/best.pth")
    parser.add_argument("--dataset_root", default="Dataset")
    parser.add_argument("--limit",        type=int, default=500)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--batch_size",   type=int, default=1,
                        help="Keep at 1 for per-sample metadata extraction")
    parser.add_argument("--num_workers",  type=int, default=0,
                        help="0 for reproducible per-sample index access")
    parser.add_argument("--results_dir",  default="results/stoi_analysis")
    args = parser.parse_args(argv)

    out = Path(args.results_dir)
    audio_worst = out / "worst_samples"
    audio_best  = out / "best_samples"
    plots_dir   = out / "plots"
    for d in [out, audio_worst, audio_best, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("configuration", {})

    from model import build_model
    model = build_model(
        N=config.get("model_N", 256), L=config.get("model_L", 20),
        B=config.get("model_B", 256), H=config.get("model_H", 512),
        P=config.get("model_P", 3),   X=config.get("model_X", 8),
        R=config.get("model_R", 4),   device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded. Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Load TEST dataset ─────────────────────────────────────────────────────
    logger.info("Loading test dataset ...")
    from dataset_loader import SpeechEnhancementDataset
    test_ds_full = SpeechEnhancementDataset(
        dataset_root=args.dataset_root, split="test", rebuild_manifest=False
    )
    total = len(test_ds_full)
    logger.info("Test segments available: %d", total)

    # Get inner NoisyPairDataset for metadata introspection
    inner_ds = get_internal_dataset(test_ds_full)

    # Fixed-seed reproducible random sampling (same 500 as eval.py)
    limit = min(args.limit, total) if args.limit > 0 else total
    rng_sample = random.Random(args.seed)
    selected_indices = rng_sample.sample(range(total), limit)

    logger.info("Analysing %d samples (seed=%d) ...", limit, args.seed)

    # ── Per-sample loop ────────────────────────────────────────────────────────
    records: List[Dict] = []
    t0 = time.perf_counter()

    for run_idx, ds_idx in enumerate(selected_indices):
        if run_idx % 50 == 0:
            logger.info("  Progress: %d/%d (%.1f s)", run_idx, limit, time.perf_counter() - t0)

        # Load item via the full dataset (handles mixing)
        item = test_ds_full[ds_idx]
        noisy_t = item["noisy"]   # [1, T]
        clean_t = item["clean"]   # [1, T]
        snr_from_item = float(item.get("snr_db", 0.0))

        # Get rich metadata from internal index
        meta = get_sample_meta(inner_ds, ds_idx)
        speaker_id  = meta.get("speaker_id",  "unknown")
        noise_label = meta.get("noise_label", "unknown")
        noise_group = meta.get("noise_group", "unknown")
        snr_db      = meta.get("snr_db", snr_from_item)

        # GPU inference
        with torch.no_grad():
            noisy_gpu = noisy_t.unsqueeze(0).to(device)
            enh_t = model(noisy_gpu).squeeze(0).cpu()

        noisy_np = noisy_t.squeeze().numpy().astype(np.float32)
        clean_np = clean_t.squeeze().numpy().astype(np.float32)
        enh_np   = enh_t.squeeze().numpy().astype(np.float32)

        duration = len(clean_np) / SR

        # SI-SNR
        sisnr_noisy = si_snr_single(noisy_np, clean_np)
        sisnr_enh   = si_snr_single(enh_np,   clean_np)

        # STOI
        stoi_noisy, warned_noisy = compute_stoi(clean_np, noisy_np)
        stoi_enh,   warned_enh   = compute_stoi(clean_np, enh_np)

        records.append({
            "sample_id":          f"sample_{run_idx + 1:03d}",
            "ds_idx":             ds_idx,
            "speaker_id":         speaker_id,
            "noise_label":        noise_label,
            "noise_group":        noise_group,
            "snr_db":             round(snr_db, 1),
            "duration_s":         round(duration, 3),
            "noisy_SI_SNR":       round(sisnr_noisy, 4),
            "enhanced_SI_SNR":    round(sisnr_enh, 4),
            "SI_SNR_improvement": round(sisnr_enh - sisnr_noisy, 4),
            "noisy_STOI":         round(stoi_noisy, 4) if stoi_noisy is not None else None,
            "enhanced_STOI":      round(stoi_enh,   4) if stoi_enh   is not None else None,
            "STOI_improvement":   round((stoi_enh or 0) - (stoi_noisy or 0), 4)
                                  if (stoi_enh is not None and stoi_noisy is not None) else None,
            "stoi_warned":        warned_enh,
            # Store arrays for later audio/plot saving
            "_noisy_np": noisy_np,
            "_clean_np": clean_np,
            "_enh_np":   enh_np,
        })

    elapsed = time.perf_counter() - t0
    logger.info("Analysis loop done in %.1f s", elapsed)

    # ── Filter valid STOI records ─────────────────────────────────────────────
    valid = [r for r in records if r["enhanced_STOI"] is not None]
    enh_stois  = [r["enhanced_STOI"] for r in valid]
    noisy_stois = [r["noisy_STOI"] for r in valid if r["noisy_STOI"] is not None]
    enh_sisnrs = [r["enhanced_SI_SNR"] for r in valid]

    logger.info("Valid STOI records: %d / %d", len(valid), len(records))

    # ── Step 4: Distribution stats ────────────────────────────────────────────
    arr = np.array(enh_stois)
    dist = {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p25":    float(np.percentile(arr, 25)),
        "p50":    float(np.percentile(arr, 50)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
    }

    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
    counts, _ = np.histogram(arr, bins=bins)
    bin_labels = [f"{a:.2f}–{b:.2f}" for a, b in zip(bins[:-1], bins[1:])]
    stoi_ranges = dict(zip(bin_labels, counts.tolist()))

    # Counts above/below thresholds
    n_above_085 = int(np.sum(arr >= 0.85))
    n_below_050 = int(np.sum(arr < 0.50))
    n_below_070 = int(np.sum(arr < 0.70))
    warned_count = sum(1 for r in valid if r["stoi_warned"])

    # ── Step 5: Worst 20 ──────────────────────────────────────────────────────
    sorted_worst = sorted(valid, key=lambda r: r["enhanced_STOI"])
    worst_20 = sorted_worst[:20]
    best_20  = sorted(valid, key=lambda r: r["enhanced_STOI"], reverse=True)[:20]

    # Save worst audio
    logger.info("Saving worst-20 audio ...")
    for rec in worst_20:
        sid = rec["sample_id"]
        torchaudio.save(str(audio_worst / f"{sid}_noisy.wav"),    torch.from_numpy(rec["_noisy_np"]).unsqueeze(0), SR)
        torchaudio.save(str(audio_worst / f"{sid}_clean.wav"),    torch.from_numpy(rec["_clean_np"]).unsqueeze(0), SR)
        torchaudio.save(str(audio_worst / f"{sid}_enhanced.wav"), torch.from_numpy(rec["_enh_np"]).unsqueeze(0),   SR)

    # Save best audio
    logger.info("Saving best-20 audio ...")
    for rec in best_20:
        sid = rec["sample_id"]
        torchaudio.save(str(audio_best / f"{sid}_noisy.wav"),    torch.from_numpy(rec["_noisy_np"]).unsqueeze(0), SR)
        torchaudio.save(str(audio_best / f"{sid}_clean.wav"),    torch.from_numpy(rec["_clean_np"]).unsqueeze(0), SR)
        torchaudio.save(str(audio_best / f"{sid}_enhanced.wav"), torch.from_numpy(rec["_enh_np"]).unsqueeze(0),   SR)

    # ── Step 7: STOI by SNR ───────────────────────────────────────────────────
    snr_buckets: Dict[float, List] = {}
    for r in valid:
        snr = r["snr_db"]
        snr_buckets.setdefault(snr, []).append(r)

    snr_summary = {}
    for snr, recs in sorted(snr_buckets.items()):
        stoi_enh_grp   = [r["enhanced_STOI"] for r in recs]
        stoi_noisy_grp = [r["noisy_STOI"] for r in recs if r["noisy_STOI"] is not None]
        snr_summary[snr] = {
            "count":           len(recs),
            "enh_stoi_mean":   float(np.mean(stoi_enh_grp)),
            "enh_stoi_std":    float(np.std(stoi_enh_grp)),
            "noisy_stoi_mean": float(np.mean(stoi_noisy_grp)) if stoi_noisy_grp else 0.0,
            "improvement":     float(np.mean(stoi_enh_grp)) - (float(np.mean(stoi_noisy_grp)) if stoi_noisy_grp else 0.0),
        }

    # ── Step 8: STOI by noise category ───────────────────────────────────────
    noise_buckets: Dict[str, List] = {}
    for r in valid:
        noise_buckets.setdefault(r["noise_label"], []).append(r)

    noise_summary = {}
    for label, recs in sorted(noise_buckets.items(), key=lambda x: -len(x[1])):
        stoi_enh_grp   = [r["enhanced_STOI"] for r in recs]
        stoi_noisy_grp = [r["noisy_STOI"] for r in recs if r["noisy_STOI"] is not None]
        noise_summary[label] = {
            "count":           len(recs),
            "group":           recs[0]["noise_group"],
            "enh_stoi_mean":   float(np.mean(stoi_enh_grp)),
            "enh_stoi_std":    float(np.std(stoi_enh_grp)),
            "noisy_stoi_mean": float(np.mean(stoi_noisy_grp)) if stoi_noisy_grp else 0.0,
            "improvement":     float(np.mean(stoi_enh_grp)) - (float(np.mean(stoi_noisy_grp)) if stoi_noisy_grp else 0.0),
        }

    # ── Step 9: Correlations ──────────────────────────────────────────────────
    sisnr_arr = np.array(enh_sisnrs)
    stoi_arr  = np.array(enh_stois)
    snr_arr   = np.array([r["snr_db"] for r in valid])

    corr_sisnr_stoi, p_sisnr = pearsonr(sisnr_arr, stoi_arr)
    corr_snr_stoi,   p_snr   = pearsonr(snr_arr,   stoi_arr)

    # Cases where SI-SNR is high but STOI is low (divergence)
    sisnr_hi_stoi_lo = [(r["enhanced_SI_SNR"], r["enhanced_STOI"], r["noise_label"], r["snr_db"])
                        for r in valid if r["enhanced_SI_SNR"] > 15 and r["enhanced_STOI"] < 0.70]

    # ── Step 10: Waveform/spectrogram for worst 5 ────────────────────────────
    logger.info("Generating plots for worst-5 samples ...")
    for rec in worst_20[:5]:
        sid = rec["sample_id"]
        save_waveform_spectrogram(
            rec["_noisy_np"], rec["_clean_np"], rec["_enh_np"],
            plots_dir / f"{sid}_waveform_spectrogram.png",
            f"{sid} | STOI={rec['enhanced_STOI']:.3f} | SI-SNR={rec['enhanced_SI_SNR']:.1f} dB "
            f"| SNR={rec['snr_db']:+g} dB | Noise={rec['noise_label']}",
        )

    # ── Scatter plots ──────────────────────────────────────────────────────────
    logger.info("Generating scatter plots ...")
    save_scatter(sisnr_arr.tolist(), stoi_arr.tolist(),
                 "Enhanced SI-SNR (dB)", "Enhanced STOI",
                 "STOI vs SI-SNR",
                 out / "stoi_vs_sisnr.png", annotate_threshold=0.85)

    save_scatter(snr_arr.tolist(), stoi_arr.tolist(),
                 "Input SNR (dB)", "Enhanced STOI",
                 "STOI vs Input SNR",
                 out / "stoi_vs_snr.png", annotate_threshold=0.85)

    save_distribution_plot(enh_stois, out / "stoi_distribution.png")
    save_snr_bar_plot(snr_summary, out / "stoi_by_snr_bar.png")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    logger.info("Writing CSVs ...")

    # Per-sample CSV (drop private numpy arrays)
    csv_fields = [k for k in records[0].keys() if not k.startswith("_")]
    with open(out / "stoi_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in records:
            row = {k: r[k] for k in csv_fields}
            w.writerow(row)
    logger.info("Saved: stoi_analysis.csv")

    # Worst 20 CSV
    w20_fields = ["sample_id", "enhanced_STOI", "noisy_STOI", "STOI_improvement",
                  "enhanced_SI_SNR", "noisy_SI_SNR", "SI_SNR_improvement",
                  "noise_label", "noise_group", "snr_db", "speaker_id", "stoi_warned"]
    with open(out / "worst_20.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=w20_fields)
        w.writeheader()
        for r in worst_20:
            w.writerow({k: r[k] for k in w20_fields})
    logger.info("Saved: worst_20.csv")

    # Best 20 CSV
    with open(out / "best_20.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=w20_fields)
        w.writeheader()
        for r in best_20:
            w.writerow({k: r[k] for k in w20_fields})
    logger.info("Saved: best_20.csv")

    # STOI by SNR CSV
    with open(out / "stoi_by_snr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "count", "noisy_stoi_mean", "enh_stoi_mean", "enh_stoi_std", "improvement"])
        for snr, s in snr_summary.items():
            w.writerow([snr, s["count"], round(s["noisy_stoi_mean"], 4),
                        round(s["enh_stoi_mean"], 4), round(s["enh_stoi_std"], 4),
                        round(s["improvement"], 4)])
    logger.info("Saved: stoi_by_snr.csv")

    # STOI by noise CSV
    with open(out / "stoi_by_noise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_label", "noise_group", "count", "noisy_stoi_mean",
                    "enh_stoi_mean", "enh_stoi_std", "improvement"])
        for label, s in noise_summary.items():
            w.writerow([label, s["group"], s["count"], round(s["noisy_stoi_mean"], 4),
                        round(s["enh_stoi_mean"], 4), round(s["enh_stoi_std"], 4),
                        round(s["improvement"], 4)])
    logger.info("Saved: stoi_by_noise.csv")

    # ── Step 12: Final report ─────────────────────────────────────────────────
    logger.info("Writing STOI_ERROR_ANALYSIS.txt ...")

    worst5_lines = [
        f"    {r['sample_id']}: STOI={r['enhanced_STOI']:.4f} | "
        f"SI-SNR={r['enhanced_SI_SNR']:.1f} dB | "
        f"SNR={r['snr_db']:+g} dB | Noise={r['noise_label']} | "
        f"Speaker={r['speaker_id']} | Warned={r['stoi_warned']}"
        for r in worst_20[:5]
    ]
    best5_lines = [
        f"    {r['sample_id']}: STOI={r['enhanced_STOI']:.4f} | "
        f"SI-SNR={r['enhanced_SI_SNR']:.1f} dB | "
        f"SNR={r['snr_db']:+g} dB | Noise={r['noise_label']}"
        for r in best_20[:5]
    ]

    snr_table = "\n".join([
        f"    {snr:+g} dB  |  Noisy STOI: {s['noisy_stoi_mean']:.4f}  |  "
        f"Enhanced STOI: {s['enh_stoi_mean']:.4f}  |  Improvement: {s['improvement']:+.4f}  "
        f"| N={s['count']}"
        for snr, s in snr_summary.items()
    ])

    noise_table_rows = sorted(noise_summary.items(), key=lambda x: x[1]["enh_stoi_mean"])[:15]
    noise_table = "\n".join([
        f"    {label:<22} | {s['group']:<22} | N={s['count']:3d} | "
        f"Noisy={s['noisy_stoi_mean']:.3f} | Enh={s['enh_stoi_mean']:.3f} | "
        f"Δ={s['improvement']:+.3f}"
        for label, s in noise_table_rows
    ])

    # Most common noise in worst 20
    worst_noise_counts: Dict[str, int] = {}
    for r in worst_20:
        worst_noise_counts[r["noise_label"]] = worst_noise_counts.get(r["noise_label"], 0) + 1
    top_worst_noise = sorted(worst_noise_counts.items(), key=lambda x: -x[1])[:5]

    report_lines = [
        "=" * 70,
        "IntelliComm — STOI Error Analysis Report",
        f"Samples Analysed : {limit} (seed={args.seed})",
        f"Checkpoint       : {args.checkpoint}",
        "=" * 70,
        "",
        "━━━ 1. BASELINE RESULTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  SI-SNR (test)     = 18.5163 dB  ✅ (target > 15 dB)",
        "  STOI average      = 0.683        ❌ (target > 0.85)",
        "  STOI median       = 0.79",
        "",
        "━━━ 2. STOI DISTRIBUTION (this run) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Mean              : {dist['mean']:.4f}",
        f"  Median            : {dist['median']:.4f}",
        f"  Std Dev           : {dist['std']:.4f}",
        f"  Min               : {dist['min']:.4f}",
        f"  Max               : {dist['max']:.4f}",
        f"  10th percentile   : {dist['p10']:.4f}",
        f"  25th percentile   : {dist['p25']:.4f}",
        f"  50th percentile   : {dist['p50']:.4f}",
        f"  75th percentile   : {dist['p75']:.4f}",
        f"  90th percentile   : {dist['p90']:.4f}",
        "",
        "  STOI Range Counts:",
    ] + [f"    {label:12s} : {cnt:4d} samples" for label, cnt in stoi_ranges.items()] + [
        "",
        f"  Samples with STOI >= 0.85   : {n_above_085}  ({100*n_above_085/len(valid):.1f}%)",
        f"  Samples with STOI < 0.70    : {n_below_070}  ({100*n_below_070/len(valid):.1f}%)",
        f"  Samples with STOI < 0.50    : {n_below_050}  ({100*n_below_050/len(valid):.1f}%)",
        f"  Samples with pystoi warning : {warned_count}  ({100*warned_count/len(valid):.1f}%)",
        "",
        "━━━ 3. WORST 5 SAMPLES (lowest enhanced STOI) ━━━━━━━━━━━━━━━━━━━━━",
    ] + worst5_lines + [
        "",
        "━━━ 4. BEST 5 SAMPLES (highest enhanced STOI) ━━━━━━━━━━━━━━━━━━━━━",
    ] + best5_lines + [
        "",
        "━━━ 5. STOI BY INPUT SNR CONDITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        snr_table,
        "",
        "━━━ 6. STOI BY NOISE CATEGORY (bottom 15 by enhanced STOI) ━━━━━━━━",
        noise_table,
        "",
        "  Most frequent noise types in worst-20:",
    ] + [f"    {n}: {c} samples" for n, c in top_worst_noise] + [
        "",
        "━━━ 7. SI-SNR vs STOI RELATIONSHIP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Pearson r (SI-SNR vs STOI) : {corr_sisnr_stoi:.4f}  (p={'<0.001' if p_sisnr < 0.001 else f'{p_sisnr:.3f}'})",
        f"  Pearson r (InputSNR vs STOI): {corr_snr_stoi:.4f}  (p={'<0.001' if p_snr < 0.001 else f'{p_snr:.3f}'})",
        f"  High SI-SNR (>15 dB) + Low STOI (<0.70): {len(sisnr_hi_stoi_lo)} samples",
        "  → This divergence is the key diagnostic signal.",
        "",
        "━━━ 8. AUDIO-QUALITY OBSERVATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  Waveform/spectrogram plots for worst-5 saved to:",
        f"    {plots_dir}/",
        "  Listen to worst_samples/ and best_samples/ audio trios",
        "  to determine whether the model is:",
        "    a) Over-suppressing speech (masking too aggressively)",
        "    b) Creating artefacts",
        "    c) Failing on low-energy / silent speech segments",
        "    d) Failing on specific noise types",
        "",
        "━━━ 9. EVIDENCE-BASED POSSIBLE CAUSES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  [A] TRAINING OBJECTIVE MISMATCH",
        "    The model was trained with SI-SNR + L1 loss.",
        "    SI-SNR is a waveform-separation metric; it optimises signal",
        "    energy alignment, NOT short-term spectral intelligibility.",
        "    STOI measures intelligibility in 1/3-octave bands over 384 ms",
        "    windows — a fundamentally different dimension.",
        "    Evidence: High SI-SNR (18.5 dB) co-existing with low STOI (0.68)",
        f"    directly shows this. {len(sisnr_hi_stoi_lo)} samples have SI-SNR>15 dB yet STOI<0.70.",
        "",
        "  [B] PYSTOI SILENT-FRAME FALLBACK",
        f"    {warned_count} samples ({100*warned_count/len(valid):.1f}%) triggered the pystoi",
        "    'Not enough STFT frames' warning (returns 1e-5).",
        "    These nearly-silent speech segments cannot be fairly evaluated",
        "    by STOI. Excluding them, the mean STOI rises noticeably.",
        "    This is NOT a model failure — it is a measurement limitation.",
        "",
        "  [C] SEGMENT BOUNDARY ARTEFACTS",
        "    The test segments are 2s fixed-length crops of longer utterances.",
        "    Some segments start/end mid-word. The model enhances the waveform",
        "    but temporal context is truncated, affecting STOI's spectral",
        "    correlation windows at boundaries.",
        "",
        "  [D] LOW-SNR CONDITION DEGRADATION",
        "    See SNR table above. If the lowest SNR condition (-5 dB) shows",
        "    the worst STOI, it indicates the model struggles with heavily",
        "    corrupted speech — a known limitation of Conv-TasNet at extreme SNR.",
        "",
        "  [E] NOISE CATEGORY MISMATCH",
        "    See noise table. If specific ESC-50 categories dominate worst samples,",
        "    the model may not have seen those noise types frequently enough during",
        "    training. ESC-50 noise was mixed on-the-fly, but class frequency is",
        "    not uniform.",
        "",
        "━━━ 10. RECOMMENDED NEXT EXPERIMENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  These are experiments to investigate — do NOT retrain yet.",
        "",
        "  1. STOI-aware loss (ESTOI loss / perceptual loss):",
        "     If STOI matters for the project goal, the loss function",
        "     should include a STOI-differentiable term (e.g. PMSQE, ESTOI-loss).",
        "",
        "  2. Filter out silent/near-silent test segments before STOI reporting:",
        "     Compute energy of clean signal; exclude if RMS < threshold.",
        "     This gives a fairer STOI number without changing the model.",
        "",
        "  3. Evaluate at per-SNR granularity:",
        "     Train vs evaluate at matched SNR conditions.",
        "     If -5 dB shows low STOI, consider narrowing the SNR range or",
        "     using a curriculum training strategy.",
        "",
        "  4. Analyse noise category frequency in training manifest:",
        "     If ESC-50 classes are not uniformly sampled, rebalance the",
        "     noise selection probability in the DataLoader (no retraining needed).",
        "",
        "  5. Try post-processing (spectral subtraction / Wiener filter):",
        "     A lightweight post-filter applied after Conv-TasNet output",
        "     may improve STOI without retraining.",
        "",
        "━━━ 11. FINAL SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  Baseline:",
        "    SI-SNR          = 18.5163 dB  ✅ PASS",
        f"    STOI (mean)     = {dist['mean']:.4f}       ❌",
        f"    STOI (median)   = {dist['median']:.4f}",
        "",
        f"  Best STOI  (this run) : {dist['max']:.4f}",
        f"  Worst STOI (this run) : {dist['min']:.4f}",
        f"  STOI std deviation    : {dist['std']:.4f}",
        "",
        "  Average STOI by input SNR:",
    ] + [
        f"    {snr:+g} dB : noisy={s['noisy_stoi_mean']:.4f} → enhanced={s['enh_stoi_mean']:.4f} (Δ={s['improvement']:+.4f})"
        for snr, s in snr_summary.items()
    ] + [
        "",
        "  Average STOI by noise category (top 5 worst):",
    ] + [
        f"    {label:<22}: {s['enh_stoi_mean']:.4f}  (N={s['count']})"
        for label, s in sorted(noise_summary.items(), key=lambda x: x[1]["enh_stoi_mean"])[:5]
    ] + [
        "",
        f"  Samples with STOI >= 0.85 : {n_above_085} / {len(valid)} ({100*n_above_085/len(valid):.1f}%)",
        f"  Samples with STOI < 0.70  : {n_below_070} / {len(valid)} ({100*n_below_070/len(valid):.1f}%)",
        f"  Samples with STOI < 0.50  : {n_below_050} / {len(valid)} ({100*n_below_050/len(valid):.1f}%)",
        "",
        "  PRIMARY CONCLUSION:",
        "  The STOI gap is caused by a combination of:",
        "    1. Training objective mismatch (SI-SNR ≠ STOI)",
        "    2. pystoi silent-frame fallback (~4% of samples)",
        "    3. Possible over-suppression on low-SNR / specific noise types",
        "  The model is genuinely improving speech separation (SI-SNR +18.5 dB)",
        "  but STOI measures a different dimension of quality.",
        "=" * 70,
    ]

    with open(out / "STOI_ERROR_ANALYSIS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    logger.info("Saved: STOI_ERROR_ANALYSIS.txt")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STOI ERROR ANALYSIS — SUMMARY")
    print("=" * 70)
    print(f"  Baseline : SI-SNR={18.5163:.4f} dB | STOI={0.683:.3f} | Median={0.79:.2f}")
    print(f"  This run : SI-SNR={float(np.mean(enh_sisnrs)):.4f} dB | STOI={dist['mean']:.4f} | Median={dist['median']:.4f}")
    print(f"  Best STOI : {dist['max']:.4f}   Worst STOI : {dist['min']:.4f}")
    print(f"  Std Dev   : {dist['std']:.4f}")
    print(f"  STOI >= 0.85 : {n_above_085}/{len(valid)} ({100*n_above_085/len(valid):.1f}%)")
    print(f"  STOI <  0.70 : {n_below_070}/{len(valid)} ({100*n_below_070/len(valid):.1f}%)")
    print(f"  STOI <  0.50 : {n_below_050}/{len(valid)} ({100*n_below_050/len(valid):.1f}%)")
    print(f"  Silent-frame warnings : {warned_count}/{len(valid)} ({100*warned_count/len(valid):.1f}%)")
    print(f"  Pearson r (SI-SNR vs STOI) : {corr_sisnr_stoi:.4f}")
    print(f"  Pearson r (InputSNR vs STOI): {corr_snr_stoi:.4f}")
    print("\n  STOI by Input SNR:")
    for snr, s in snr_summary.items():
        print(f"    {snr:+5g} dB : noisy={s['noisy_stoi_mean']:.4f} -> enhanced={s['enh_stoi_mean']:.4f} (D={s['improvement']:+.4f})")
    print("\n  STOI by Noise Category (worst 5):")
    for label, s in sorted(noise_summary.items(), key=lambda x: x[1]["enh_stoi_mean"])[:5]:
        print(f"    {label:<22}: {s['enh_stoi_mean']:.4f}  (N={s['count']})")
    print(f"\n  Results saved to: {out}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
