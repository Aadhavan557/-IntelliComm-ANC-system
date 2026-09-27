"""
batch_infer.py
==============
IntelliComm – AI-Based Speech Enhancement for Military/Field Communication
Batch Inference Script

Processes all WAV and FLAC files in a directory (and subdirectories) through
the Conv-TasNet model and saves the enhanced output to an output directory,
preserving the directory structure.

Usage:
  python batch_infer.py --input_dir path/to/noisy_audio/ --output_dir results/batch_enhanced/
  python batch_infer.py --input_dir data/ --checkpoint checkpoints/best.pth
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio

from model import build_model
from infer import load_audio, resample_to_16k, enhance_waveform, TARGET_SR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="IntelliComm — Batch Speech Enhancement Inference")
    parser.add_argument("--input_dir",   required=True,    help="Directory containing noisy audio files")
    parser.add_argument("--output_dir",  default="results/batch_enhanced", help="Output directory for enhanced audio")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pth", help="Model checkpoint path")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    ckpt_path = Path(args.checkpoint)

    if not in_dir.exists() or not in_dir.is_dir():
        logger.error("Input directory not found or is not a directory: %s", in_dir)
        sys.exit(1)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    # Find all audio files
    audio_extensions = {".wav", ".flac"}
    audio_files = []
    for root, _, files in os.walk(in_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(Path(root) / file)

    if not audio_files:
        logger.warning("No audio files found in %s", in_dir)
        sys.exit(0)
    
    logger.info("Found %d audio files in %s", len(audio_files), in_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("configuration", {})

    model = build_model(
        N=config.get("model_N", 256), L=config.get("model_L", 20),
        B=config.get("model_B", 256), H=config.get("model_H", 512),
        P=config.get("model_P", 3),   X=config.get("model_X", 8),
        R=config.get("model_R", 4),   device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded successfully.")

    # Process files
    logger.info("Starting batch inference...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t0_total = time.perf_counter()
    total_duration = 0.0

    # Using tqdm for progress bar
    for file_path in tqdm(audio_files, desc="Enhancing audio", unit="file"):
        try:
            # Maintain folder structure
            rel_path = file_path.relative_to(in_dir)
            out_file_path = out_dir / rel_path
            out_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Process
            wav, sr = load_audio(file_path)
            duration = wav.shape[-1] / sr
            total_duration += duration
            
            wav_16k = resample_to_16k(wav, sr)
            
            with torch.no_grad():
                enhanced = enhance_waveform(model, wav_16k, device)
            
            torchaudio.save(str(out_file_path), enhanced.float(), TARGET_SR)
            
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path, e)

    total_time = time.perf_counter() - t0_total
    rtf = total_time / total_duration if total_duration > 0 else 0
    
    logger.info("==================================================")
    logger.info("Batch Inference Complete!")
    logger.info("Processed %d files.", len(audio_files))
    logger.info("Total audio duration: %.2f seconds", total_duration)
    logger.info("Total processing time: %.2f seconds", total_time)
    logger.info("Overall Real-Time Factor (RTF): %.3fx", rtf)
    logger.info("Outputs saved to: %s", out_dir)
    logger.info("==================================================")

if __name__ == "__main__":
    main()
