"""
eval.py
=======
IntelliComm – AI-Based Speech Enhancement for Military/Field Communication
Complete Evaluation Script

Tasks:
  1. Evaluate checkpoints/best.pth on the unseen TEST split
  2. Compute SI-SNR, SI-SDR, STOI, PESQ
  3. Save results/final_evaluation.csv and results/final_evaluation.txt
  4. Save before/after audio samples  (results/audio/)
  5. Save waveform + spectrogram plots (results/plots/)
  6. Write results/final_report.txt with PASS / NOT PASS targets

Usage:
  python eval.py                              # defaults: 500 samples, seed 42
  python eval.py --limit 500 --seed 42        # reproducible 500-sample subset
  python eval.py --limit 0                    # full test set (~11k samples)
  python eval.py --audio_samples 5            # save 5 before/after audio trios
  python eval.py --plot_samples 3             # save 3 waveform+spectrogram plots

STOI:
  Requires: pip install pystoi                (✅ already installed)
PESQ:
  Requires: pip install pesq
  NOTE: pesq must be compiled from source on Windows.
  It needs Microsoft C++ Build Tools (free):
    https://visualstudio.microsoft.com/visual-cpp-build-tools/
  After installing Build Tools, run: pip install pesq
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset

from model import build_model
from loss import si_snr_metric, si_sdr_metric
from dataset_loader import SpeechEnhancementDataset

# ── Optional perceptual metrics ───────────────────────────────────────────────
try:
    from pystoi import stoi as _stoi_fn
    HAVE_STOI = True
except ImportError:
    HAVE_STOI = False

try:
    from pesq import pesq as _pesq_fn
    HAVE_PESQ = True
except ImportError:
    HAVE_PESQ = False

# ── Matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram as _scipy_spectrogram
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

warnings.filterwarnings("ignore", message=".*mix_with_snr.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project targets ───────────────────────────────────────────────────────────
TARGETS = {"SI-SNR_dB": 15.0, "STOI": 0.85, "PESQ": 2.5}
SR = 16000  # model sample rate


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """[1, T] or [T] tensor → float32 numpy."""
    return tensor.squeeze().cpu().float().numpy()


def compute_stoi(clean_np: np.ndarray, enhanced_np: np.ndarray) -> float | None:
    """Extended STOI via pystoi. Returns None if unavailable or error."""
    if not HAVE_STOI:
        return None
    try:
        return float(_stoi_fn(
            clean_np.astype(np.float64),
            enhanced_np.astype(np.float64),
            SR, extended=True,
        ))
    except Exception as e:
        logger.debug("STOI error: %s", e)
        return None


def compute_pesq(clean_np: np.ndarray, enhanced_np: np.ndarray) -> float | None:
    """Wideband PESQ via pesq package. Returns None if unavailable or error."""
    if not HAVE_PESQ:
        return None
    try:
        return float(_pesq_fn(SR, clean_np.astype(np.float32), enhanced_np.astype(np.float32), "wb"))
    except Exception as e:
        logger.debug("PESQ error: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_wav(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = tensor.squeeze().unsqueeze(0).cpu().float()  # [1, T]
    torchaudio.save(str(path), wav, SR)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_waveform_plot(noisy_np, clean_np, enhanced_np, out_path: Path, sample_id: str) -> None:
    if not HAVE_MPL:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(clean_np)) / SR

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Waveform Comparison — {sample_id}", fontsize=13, fontweight="bold")

    for ax, wav, label, color in zip(
        axes,
        [noisy_np, clean_np, enhanced_np],
        ["Noisy Input", "Clean Reference", "Enhanced Output"],
        ["#e74c3c", "#2ecc71", "#3498db"],
    ):
        ax.plot(t, wav, linewidth=0.4, color=color, alpha=0.85)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved waveform plot: %s", out_path)


def save_spectrogram_plot(noisy_np, clean_np, enhanced_np, out_path: Path, sample_id: str) -> None:
    if not HAVE_MPL:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Spectrogram Comparison — {sample_id}", fontsize=13, fontweight="bold")

    for ax, wav, label in zip(
        axes,
        [noisy_np, clean_np, enhanced_np],
        ["Noisy Input", "Clean Reference", "Enhanced Output"],
    ):
        f, t, Sxx = _scipy_spectrogram(wav, fs=SR, nperseg=512, noverlap=256)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax.pcolormesh(t, f / 1000, Sxx_db, shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        ax.set_ylabel("Frequency (kHz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, label="dB")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spectrogram plot: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f", fallback="N/A"):
    return format(val, fmt) if isinstance(val, float) else fallback


def save_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)
    logger.info("CSV results saved: %s", path)


def save_txt_summary(path: Path, m: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "IntelliComm — Evaluation Summary",
        "=" * 60,
        f"  Timestamp       : {m.get('timestamp', 'N/A')}",
        f"  Checkpoint      : {m.get('checkpoint', 'N/A')}",
        f"  Test samples    : {m.get('num_samples', 'N/A')}",
        f"  Random seed     : {m.get('seed', 'N/A')}",
        f"  Evaluation time : {_fmt(m.get('eval_time_s'), '.2f')} s",
        "",
        "  Metrics:",
        f"    SI-SNR  : {_fmt(m.get('si_snr_dB'))} dB",
        f"    SI-SDR  : {_fmt(m.get('si_sdr_dB'))} dB",
        f"    STOI    : {_fmt(m.get('stoi'))}",
        f"    PESQ    : {_fmt(m.get('pesq'))}",
        "=" * 60,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Text summary saved: %s", path)


def save_final_report(path: Path, m: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def pf(value, target: float) -> str:
        if not isinstance(value, float):
            return "NOT MEASURED"
        return "✅ PASS" if value >= target else "❌ NOT PASS"

    si_snr = m.get("si_snr_dB")
    stoi   = m.get("stoi")
    pesq   = m.get("pesq")

    pesq_note = (
        "  PESQ requires the 'pesq' package (needs C++ Build Tools on Windows).\n"
        "  To install:\n"
        "    1. Download & install Microsoft C++ Build Tools:\n"
        "       https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "    2. Run: pip install pesq\n"
        "    3. Re-run: python eval.py"
    ) if not isinstance(pesq, float) else ""

    lines = [
        "=" * 65,
        "IntelliComm — Final Evaluation Report",
        f"Generated  : {m.get('timestamp', 'N/A')}",
        "=" * 65,
        "",
        "[ MODEL ]",
        "  Architecture  : Conv-TasNet",
        "  Training data : 10,000 balanced samples",
        "  Epochs        : 30",
        "  Checkpoint    : checkpoints/best.pth",
        "",
        "[ EVALUATION SETUP ]",
        f"  Test split    : SpeechEnhancementDataset (split='test')",
        f"  Test samples  : {m.get('num_samples', 'N/A')}",
        f"  Random seed   : {m.get('seed', 'N/A')}",
        f"  Eval time     : {_fmt(m.get('eval_time_s'), '.2f')} s",
        f"  Device        : {m.get('device', 'N/A')}",
        "",
        "[ METRICS ]",
        f"  SI-SNR  : {_fmt(si_snr)} dB",
        f"  SI-SDR  : {_fmt(m.get('si_sdr_dB'))} dB",
        f"  STOI    : {_fmt(stoi)}   (pystoi extended STOI; range 0–1)",
        f"  PESQ    : {_fmt(pesq)}   (wideband; range 1.0–4.5)",
        "",
        "[ TARGET COMPARISON ]",
        f"  Target SI-SNR > 15.0 dB  |  Achieved: {_fmt(si_snr)} dB  |  {pf(si_snr, TARGETS['SI-SNR_dB'])}",
        f"  Target STOI   > 0.85     |  Achieved: {_fmt(stoi)}        |  {pf(stoi,   TARGETS['STOI'])}",
        f"  Target PESQ   > 2.5      |  Achieved: {_fmt(pesq)}        |  {pf(pesq,   TARGETS['PESQ'])}",
        "",
        "[ NOTES ]",
        "  - SI-SNR and SI-SDR are computed on GPU tensors (batch-averaged).",
        "  - STOI uses pystoi extended STOI (ESTOI); 1.0 is perfect intelligibility.",
        "  - PESQ wideband scale: 1.0 (bad) – 4.5 (excellent). MOS-LQO mapping.",
    ]
    if pesq_note:
        lines += ["", "[ HOW TO ENABLE PESQ ]", pesq_note]

    lines += ["", "=" * 65]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Final report saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="IntelliComm — Complete Evaluation")
    parser.add_argument("--checkpoint",    default="checkpoints/best.pth")
    parser.add_argument("--dataset_root",  default="Dataset")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--limit",         type=int, default=500,
                        help="Samples to evaluate (0 = all test samples)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Fixed RNG seed for reproducible sample selection")
    parser.add_argument("--audio_samples", type=int, default=5,
                        help="Number of before/after audio trios to save")
    parser.add_argument("--plot_samples",  type=int, default=3,
                        help="Number of waveform+spectrogram plots to save")
    parser.add_argument("--results_dir",   default="results")
    parser.add_argument("--manifest_csv",  default="",
                        help="Path to manifest.csv (overrides dataset_root)")
    args = parser.parse_args(argv)

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_path   = Path(args.checkpoint)
    results_dir = Path(args.results_dir)
    audio_dir   = results_dir / "audio"
    plots_dir   = results_dir / "plots"
    csv_path    = results_dir / "final_evaluation.csv"
    txt_path    = results_dir / "final_evaluation.txt"
    report_path = results_dir / "final_report.txt"

    # ── Status banner ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("IntelliComm — Complete Evaluation")
    logger.info("=" * 60)
    logger.info("pystoi (STOI) : %s", "✅ available" if HAVE_STOI else "❌ not installed")
    logger.info("pesq   (PESQ) : %s", "✅ available" if HAVE_PESQ else
                "❌ not installed  →  pip install pesq  (needs C++ Build Tools)")
    logger.info("matplotlib    : %s", "✅ available" if HAVE_MPL else "❌ not installed")
    logger.info("=" * 60)

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("configuration", {})

    model = build_model(
        N=config.get("model_N", 256), L=config.get("model_L", 20),
        B=config.get("model_B", 256), H=config.get("model_H", 512),
        P=config.get("model_P", 3),   X=config.get("model_X", 8),
        R=config.get("model_R", 4),   device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded (%s params)", f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Load TEST split (never touches train/val) ─────────────────────────────
    if args.manifest_csv:
        from train import ManifestDataset
        logger.info("Loading dataset from manifest: %s", args.manifest_csv)
        test_ds = ManifestDataset(args.manifest_csv, segment_samples=64000)
        # Wrap it if we need to mock properties?
        # Subset slicing and DataLoader work with any Dataset.
    else:
        logger.info("Loading TEST dataset from: %s", args.dataset_root)
        test_ds = SpeechEnhancementDataset(
            dataset_root=args.dataset_root, split="test", rebuild_manifest=False
        )
    total = len(test_ds)
    logger.info("Total test segments available: %d", total)

    # Fixed-seed reproducible random sampling
    limit = args.limit if args.limit > 0 else total
    limit = min(limit, total)
    rng_sampler = random.Random(args.seed)
    indices = rng_sampler.sample(range(total), limit)
    test_ds = Subset(test_ds, indices)
    logger.info("Evaluating %d samples (seed=%d)", limit, args.seed)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    total_sisnr  = 0.0
    total_sisdr  = 0.0
    stoi_scores: list[float] = []
    pesq_scores: list[float] = []
    n_batches    = 0
    saved_audio  = 0
    saved_plots  = 0

    t_start = time.perf_counter()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            noisy    = batch["noisy"].to(device, non_blocking=True)   # [B, 1, T]
            clean    = batch["clean"].to(device, non_blocking=True)   # [B, 1, T]
            enhanced = model(noisy)                                    # [B, 1, T]

            # GPU metrics (fast, batch-level)
            total_sisnr += si_snr_metric(enhanced, clean)
            total_sisdr += si_sdr_metric(enhanced, clean)
            n_batches   += 1

            # Per-sample perceptual metrics (CPU)
            B = enhanced.shape[0]
            for i in range(B):
                enh_np   = _to_numpy(enhanced[i])
                cln_np   = _to_numpy(clean[i])
                noisy_np = _to_numpy(noisy[i])

                s = compute_stoi(cln_np, enh_np)
                p = compute_pesq(cln_np, enh_np)
                if s is not None:
                    stoi_scores.append(s)
                if p is not None:
                    pesq_scores.append(p)

                # Save audio trios
                global_idx = step * args.batch_size + i + 1
                if saved_audio < args.audio_samples:
                    tag = f"sample_{global_idx:03d}"
                    save_wav(audio_dir / f"{tag}_noisy.wav",    noisy[i].cpu())
                    save_wav(audio_dir / f"{tag}_clean.wav",    clean[i].cpu())
                    save_wav(audio_dir / f"{tag}_enhanced.wav", enhanced[i].cpu())
                    logger.info("Saved audio trio: %s", tag)
                    saved_audio += 1

                # Save waveform + spectrogram plots
                if saved_plots < args.plot_samples:
                    tag = f"sample_{global_idx:03d}"
                    save_waveform_plot(
                        noisy_np, cln_np, enh_np,
                        plots_dir / f"{tag}_waveform.png", tag,
                    )
                    save_spectrogram_plot(
                        noisy_np, cln_np, enh_np,
                        plots_dir / f"{tag}_spectrogram.png", tag,
                    )
                    saved_plots += 1

            if step % 25 == 0:
                logger.info("  Batch %d/%d | elapsed=%.1fs", step, len(loader),
                            time.perf_counter() - t_start)

    t_end = time.perf_counter()
    eval_time = t_end - t_start

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg_sisnr = total_sisnr / n_batches if n_batches else 0.0
    avg_sisdr = total_sisdr / n_batches if n_batches else 0.0
    avg_stoi  = float(np.mean(stoi_scores)) if stoi_scores else None
    avg_pesq  = float(np.mean(pesq_scores)) if pesq_scores else None

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Samples  : %d", limit)
    logger.info("  SI-SNR   : %.4f dB", avg_sisnr)
    logger.info("  SI-SDR   : %.4f dB", avg_sisdr)
    logger.info("  STOI     : %s", f"{avg_stoi:.4f}" if avg_stoi is not None else
                "N/A — install pystoi")
    logger.info("  PESQ     : %s", f"{avg_pesq:.4f}" if avg_pesq is not None else
                "N/A — install pesq (needs C++ Build Tools)")
    logger.info("  Time     : %.2f s", eval_time)
    logger.info("")
    logger.info("TARGET COMPARISON:")
    logger.info("  SI-SNR > 15.0 dB  →  %s", "✅ PASS" if avg_sisnr >= 15.0 else "❌ NOT PASS")
    if avg_stoi is not None:
        logger.info("  STOI   > 0.85     →  %s", "✅ PASS" if avg_stoi >= 0.85 else "❌ NOT PASS")
    else:
        logger.info("  STOI   > 0.85     →  NOT MEASURED (pystoi missing)")
    if avg_pesq is not None:
        logger.info("  PESQ   > 2.5      →  %s", "✅ PASS" if avg_pesq >= 2.5 else "❌ NOT PASS")
    else:
        logger.info("  PESQ   > 2.5      →  NOT MEASURED (pesq package missing)")
    logger.info("=" * 60)

    # ── Save all outputs ──────────────────────────────────────────────────────
    now = datetime.now().isoformat(timespec="seconds")
    metrics = {
        "timestamp":   now,
        "checkpoint":  str(ckpt_path),
        "num_samples": limit,
        "seed":        args.seed,
        "si_snr_dB":   round(avg_sisnr, 6),
        "si_sdr_dB":   round(avg_sisdr, 6),
        "stoi":        round(avg_stoi, 6) if avg_stoi is not None else "N/A",
        "pesq":        round(avg_pesq, 6) if avg_pesq is not None else "N/A",
        "eval_time_s": round(eval_time, 2),
        "device":      str(device),
    }

    save_csv(csv_path, metrics)
    save_txt_summary(txt_path, metrics)
    save_final_report(report_path, metrics)

    logger.info("")
    logger.info("Outputs written to:")
    logger.info("  Audio     : %s/", audio_dir)
    logger.info("  Plots     : %s/", plots_dir)
    logger.info("  CSV       : %s", csv_path)
    logger.info("  Summary   : %s", txt_path)
    logger.info("  Report    : %s", report_path)


if __name__ == "__main__":
    main()
