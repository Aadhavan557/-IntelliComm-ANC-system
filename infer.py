"""
infer.py
========
IntelliComm – AI-Based Speech Enhancement for Military/Field Communication
Inference Script

Loads checkpoints/best.pth and enhances a single noisy WAV/FLAC file.

Usage:
  python infer.py --input path/to/noisy_audio.wav
  python infer.py --input audio.wav --output results/audio/
  python infer.py --input audio.wav --checkpoint checkpoints/best.pth

Output files (original is NEVER overwritten):
  results/audio/input_noisy.wav   - copy of your input at 16 kHz mono
  results/audio/enhanced.wav      - enhanced speech output
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TARGET_SR = 16000          # Conv-TasNet training sample rate
SEGMENT_SAMPLES = 32000    # Process in 2-second chunks to save VRAM


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load any WAV/FLAC file and return (waveform [1, T], sr)."""
    wav, sr = torchaudio.load(str(path))
    # Downmix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def resample_to_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return wav
    logger.info("Resampling from %d Hz → %d Hz", sr, TARGET_SR)
    return AF.resample(wav, sr, TARGET_SR)


@torch.no_grad()
def enhance_waveform(model: torch.nn.Module, wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Run Conv-TasNet on a full-length waveform by processing in fixed-size
    overlapping chunks to avoid OOM on long audio.

    Returns enhanced waveform [1, T] on CPU.
    """
    model.eval()
    wav = wav.to(device)  # [1, T]
    T   = wav.shape[-1]

    # If the audio fits in a single chunk, process directly
    if T <= SEGMENT_SAMPLES:
        out = model(wav.unsqueeze(0)).squeeze(0)   # [1, T]
        return out.cpu()

    # Chunk-based processing with 50% overlap and Hann blending
    hop       = SEGMENT_SAMPLES // 2
    out_buf   = torch.zeros(1, T, device=device)
    weight    = torch.zeros(1, T, device=device)
    window    = torch.hann_window(SEGMENT_SAMPLES, device=device).unsqueeze(0)  # [1, S]

    start = 0
    while start < T:
        end  = min(start + SEGMENT_SAMPLES, T)
        seg  = wav[:, start:end]

        # Zero-pad the last chunk if shorter than SEGMENT_SAMPLES
        pad  = SEGMENT_SAMPLES - seg.shape[-1]
        if pad > 0:
            seg = torch.nn.functional.pad(seg, (0, pad))

        enh = model(seg.unsqueeze(0)).squeeze(0)   # [1, S]

        # Remove padding from output if we padded the input
        actual_len = end - start
        enh_trimmed = enh[:, :actual_len]
        win_trimmed = window[:, :actual_len]

        out_buf[:, start:end] += enh_trimmed * win_trimmed
        weight[:, start:end]  += win_trimmed

        start += hop

    # Avoid division by zero at edges
    out_buf = out_buf / (weight + 1e-8)
    return out_buf.cpu()


def main(argv=None):
    parser = argparse.ArgumentParser(description="IntelliComm — Speech Enhancement Inference")
    parser.add_argument("--input",       required=True,    help="Path to noisy input WAV/FLAC")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pth")
    parser.add_argument("--output_dir",  default="results/audio", help="Output directory")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    ckpt_path  = Path(args.checkpoint)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ─────────────────────────────────────────────────────────
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
    logger.info("Model loaded.")

    # ── Load audio ─────────────────────────────────────────────────────────
    logger.info("Loading input: %s", input_path)
    wav, sr = load_audio(input_path)
    duration = wav.shape[-1] / sr
    logger.info("Input: %.2f s  |  %d Hz  |  %d channels (before downmix)", duration, sr, 1)

    wav_16k = resample_to_16k(wav, sr)

    # ── Save noisy copy ─────────────────────────────────────────────────────
    noisy_out = out_dir / "input_noisy.wav"
    torchaudio.save(str(noisy_out), wav_16k.cpu(), TARGET_SR)
    logger.info("Saved noisy input copy: %s", noisy_out)

    # ── Enhance ─────────────────────────────────────────────────────────────
    logger.info("Enhancing speech ...")
    t0 = time.perf_counter()
    enhanced = enhance_waveform(model, wav_16k, device)
    elapsed  = time.perf_counter() - t0
    rtf = elapsed / duration
    logger.info("Enhancement done in %.2f s  (RTF: %.3fx)", elapsed, rtf)

    # ── Save enhanced audio ─────────────────────────────────────────────────
    enhanced_out = out_dir / "enhanced.wav"
    torchaudio.save(str(enhanced_out), enhanced.float(), TARGET_SR)
    logger.info("Saved enhanced output: %s", enhanced_out)

    logger.info("")
    logger.info("Done!")
    logger.info("  Noisy  : %s", noisy_out)
    logger.info("  Enhanced: %s", enhanced_out)


if __name__ == "__main__":
    main()
