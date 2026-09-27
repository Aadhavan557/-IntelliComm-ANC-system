"""
train.py
========
IntelliComm - AI-Based Speech Enhancement for Military/Field Communication
Streamlined Training Pipeline for Conv-TasNet (RTX 3050 Optimized)
"""

import argparse
import csv
import gc
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

warnings.filterwarnings("ignore", message=".*mix_with_snr.*", category=RuntimeWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# Project modules
from model import build_model
from loss import combined_loss, si_snr_metric, si_sdr_metric
from dataset_loader import SpeechEnhancementDataset

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "subset_manifest": "dataset_subset/train_10000/manifest.csv",
    "dataset_root": "Dataset",
    "segment_samples": 32000,
    "target_sr": 16000,
    
    "batch_size": 2,
    "epochs": 30,
    "early_stopping_patience": 5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "clip_grad_norm": 5.0,
    "l1_weight": 0.1,
    "num_workers": 2,
    "pin_memory": True,
    
    "use_amp": True,
    "seed": 42,
    
    "model_N": 256, "model_L": 20, "model_B": 256,
    "model_H": 512, "model_P": 3, "model_X": 8, "model_R": 4,
    
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "results_dir": "results",
}

import torchaudio

# ---------------------------------------------------------------------------
# ManifestDataset
# ---------------------------------------------------------------------------
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv: Union[str, Path], segment_samples: int = 32000) -> None:
        self.segment_samples = segment_samples
        manifest_csv = Path(manifest_csv)
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            self._rows = list(csv.DictReader(f))
        if not self._rows:
            raise RuntimeError("manifest.csv is empty.")
        logger.info("ManifestDataset: %d samples from %s", len(self._rows), manifest_csv)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self._rows[idx]
        noisy, _ = torchaudio.load(row["noisy_path"])
        clean, _ = torchaudio.load(row["clean_path"])
        
        # Enforce segment length (crop or pad if needed)
        if noisy.shape[1] > self.segment_samples:
            start = random.randint(0, noisy.shape[1] - self.segment_samples)
            noisy = noisy[:, start:start + self.segment_samples]
            clean = clean[:, start:start + self.segment_samples]
            
        return {
            "noisy": noisy,
            "clean": clean,
            "snr_db": float(row.get("snr_db", 0)),
        }

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("CUDA detected: %s (%.1f GB VRAM)", name, vram)
    else:
        dev = torch.device("cpu")
        logger.warning("No CUDA device found. Training will be VERY slow on CPU.")
    return dev

def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def _oom_message() -> str:
    return (
        "\n" + "=" * 70 + "\n"
        "CUDA OUT OF MEMORY (RTX 3050 6 GB)\n"
        "=" * 70 + "\n"
        "Suggestions to fix:\n"
        "  1. Reduce --batch_size (Try: --batch_size 1)\n"
        "  2. Close other applications using GPU memory\n"
        "=" * 70 + "\n"
    )

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool, l1_weight: float) -> Tuple[float, float]:
    model.eval()
    total_loss, total_sisnr, n_batches = 0.0, 0.0, 0
    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            pred = model(noisy)
            loss = combined_loss(pred, clean, l1_weight=l1_weight)
        total_loss += loss.item()
        total_sisnr += si_snr_metric(pred, clean)
        n_batches += 1
        
    if n_batches == 0: return 0.0, 0.0
    return total_loss / n_batches, total_sisnr / n_batches

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, train_loss: float, val_loss: float, si_snr: float, best_val_loss: float, config: Dict) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "SI-SNR": si_snr,
        "best_validation_loss": best_val_loss,
        "configuration": config,
        "random_seed": config["seed"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Saved checkpoint: %s", path)

def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, device: torch.device) -> Tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info("Resuming from checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_validation_loss", float("inf"))
    logger.info("  Resumed from epoch %d | val_loss: %.4f | SI-SNR: %.2f | best_val_loss: %.4f", 
                epoch, ckpt.get("validation_loss", 0.0), ckpt.get("SI-SNR", 0.0), best_val_loss)
    return epoch, best_val_loss

def main(argv=None):
    parser = argparse.ArgumentParser(description="IntelliComm Training - RTX 3050 Optimized")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs to train")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per step")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--debug", action="store_true", help="Debug mode (very small sample subset, 1 iteration)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. checkpoints/latest.pth)")
    args = parser.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    if args.epochs is not None: config["epochs"] = args.epochs
    if args.batch_size is not None: config["batch_size"] = args.batch_size
    if args.patience is not None: config["early_stopping_patience"] = args.patience
    
    set_seed(config["seed"])
    device = get_device()
    
    # Init metrics
    logger.info("=" * 60)
    logger.info("PERFORMANCE MONITORING AT STARTUP")
    logger.info("GPU: RTX 3050")
    logger.info("VRAM: 6 GB")
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("=" * 60)
    
    ckpt_dir = Path(config["checkpoint_dir"])
    log_dir = Path(config["log_dir"])
    res_dir = Path(config["results_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    log_csv = log_dir / "training_log.csv"
    summ_csv = res_dir / "training_summary.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "validation_loss", "SI-SNR", "learning_rate", "epoch_time", "samples_per_second", "gpu_memory_usage"])
    
    model = build_model(N=config["model_N"], L=config["model_L"], B=config["model_B"], H=config["model_H"], P=config["model_P"], X=config["model_X"], R=config["model_R"], device=device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler("cuda", enabled=config["use_amp"] and device.type == "cuda")

    best_val_loss = float("inf")
    start_epoch = 0
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(Path(args.resume), model, optimizer, scheduler, device)

    logger.info("Loading Training Dataset: %s", config["subset_manifest"])
    train_ds = ManifestDataset(config["subset_manifest"], segment_samples=config["segment_samples"])
    
    logger.info("Loading Validation Dataset (existing data)")
    # We pass the root folder for validation. Validation dataset handles its own manifest internally.
    val_ds = SpeechEnhancementDataset(dataset_root=config["dataset_root"], split="validation", rebuild_manifest=False)
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("Training samples:   %d", len(train_ds))
    logger.info("Validation samples: %d", len(val_ds))
    logger.info("Batch size:         %d", config["batch_size"])
    logger.info("Maximum epochs:     %d", config["epochs"])
    logger.info("Patience:           %d", config["early_stopping_patience"])
    logger.info("=" * 60)
    
    if args.debug:
        logger.info("[DEBUG MODE] Reducing dataset to 10 samples.")
        train_ds = torch.utils.data.Subset(train_ds, list(range(10)))
        val_ds = torch.utils.data.Subset(val_ds, list(range(10)))
        config["epochs"] = 1

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, 
        num_workers=config["num_workers"], pin_memory=config["pin_memory"] and device.type == "cuda", 
        persistent_workers=(config["num_workers"] > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"] and device.type == "cuda"
    )

    best_si_snr = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    t_global_start = time.perf_counter()

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        model.train()
        total_loss, n_batches, n_samples = 0.0, 0, 0
        t_epoch_start = time.perf_counter()
        
        for step, batch in enumerate(train_loader):
            # In debug mode, strictly break after 2 iterations 
            # to verify forward/backward pass without looping entirely.
            if args.debug and step >= 2:
                break
                
            t_batch_start = time.perf_counter()
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            try:
                with autocast(device_type="cuda" if config["use_amp"] else "cpu", enabled=config["use_amp"]):
                    pred = model(noisy)
                    loss = combined_loss(pred, clean, l1_weight=config["l1_weight"])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(_oom_message())
                    clear_cuda_cache()
                    sys.exit(1)
                raise
                
            batch_time = time.perf_counter() - t_batch_start
            total_loss += loss.item()
            n_samples += noisy.shape[0]
            n_batches += 1
            
            if step % 50 == 0 or args.debug:
                mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_res = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
                sps = n_samples / max(time.perf_counter() - t_epoch_start, 1e-6)
                logger.info(f"  [Epoch {epoch} | Step {step}] loss={total_loss/n_batches:.4f} | batch_time={batch_time*1000:.1f}ms | {sps:.1f} samples/s | GPU Alloc: {mem_alloc:.0f} MB | Res: {mem_res:.0f} MB")
                
        train_loss = total_loss / max(n_batches, 1)
        epoch_time = time.perf_counter() - t_epoch_start
        sps = n_samples / max(epoch_time, 1e-6)
        
        logger.info(f"Epoch {epoch} finished. Validating...")
        val_loss, val_sisnr = validate(model, val_loader, device, config["use_amp"], config["l1_weight"])
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        logger.info(f"[Epoch {epoch} Summary] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SI-SNR: {val_sisnr:.2f} dB | Time: {epoch_time:.1f}s")
        
        # Log to CSV
        with open(log_csv, "a", newline="") as f:
            mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_sisnr, current_lr, epoch_time, sps, mem_alloc])
            
        # Checkpoints
        save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, scheduler, epoch, train_loss, val_loss, val_sisnr, best_val_loss, config)
        save_checkpoint(ckpt_dir / f"epoch_{epoch:02d}.pth", model, optimizer, scheduler, epoch, train_loss, val_loss, val_sisnr, best_val_loss, config)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_si_snr = val_sisnr
            best_epoch = epoch
            epochs_without_improvement = 0
            # Update latest.pth and epoch_XX.pth with new best_val_loss inside the condition is not needed because they just used the old one. We should actually pass the updated best_val_loss.
            # Let's save the best model now.
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scheduler, epoch, train_loss, val_loss, val_sisnr, best_val_loss, config)
            logger.info("New best model saved. Validation loss improved to %.4f", best_val_loss)
        else:
            epochs_without_improvement += 1
            logger.info("No improvement in validation loss for %d consecutive epoch(s).", epochs_without_improvement)
            if epochs_without_improvement >= config["early_stopping_patience"]:
                logger.info("Early stopping triggered! Model has not improved for %d epochs.", config["early_stopping_patience"])
                break
            
    # Save Summary
    total_time = (time.perf_counter() - t_global_start) / 60
    avg_epoch_time = total_time / (epoch - start_epoch) if epoch > start_epoch else 0
    with open(summ_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["total_epochs_completed", "best_epoch", "best_validation_loss", "best_si_snr", "total_training_time_min", "avg_epoch_time_min"])
        writer.writerow([epoch - start_epoch, best_epoch, best_val_loss, best_si_snr, total_time, avg_epoch_time])
        
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE. Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
