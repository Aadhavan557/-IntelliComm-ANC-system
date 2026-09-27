"""
realtime_infer.py
=================
IntelliComm – AI-Based Speech Enhancement for Military/Field Communication
Real-time Audio Pipeline

Captures live microphone audio, enhances it using Conv-TasNet,
and plays it back through the speaker with minimal latency.

Usage:
  python realtime_infer.py
  python realtime_infer.py --list-devices
  python realtime_infer.py --input-device 1 --output-device 2
"""

import argparse
import logging
import queue
import sys
import threading
import time
import numpy as np

import sounddevice as sd
import torch

from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TARGET_SR = 16000
BLOCK_SIZE = 8000     # 0.5 seconds of new audio per block
WINDOW_SIZE = 32000   # 2.0 seconds of context for the model

def list_devices():
    print(sd.query_devices())
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="IntelliComm — Real-time Speech Enhancement")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device ID")
    parser.add_argument("--output-device", type=int, default=None, help="Output device ID")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Model checkpoint path")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    
    import os
    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    # Audio queues
    q_in = queue.Queue()
    q_out = queue.Queue()

    # Audio callbacks
    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        
        # We assume 1 channel (mono)
        in_samples = indata[:, 0].copy()
        q_in.put(in_samples)
        
        try:
            # Non-blocking get for output. 
            # If we underrun, play silence.
            out_samples = q_out.get_nowait()
            outdata[:, 0] = out_samples
        except queue.Empty:
            outdata.fill(0)

    # Inference thread
    def inference_loop():
        # Keep a running buffer of past audio to provide context for Conv-TasNet
        context_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        
        # Warmup the model
        logger.info("Warming up GPU...")
        dummy_in = torch.zeros(1, 1, WINDOW_SIZE, device=device)
        with torch.no_grad():
            _ = model(dummy_in)
        logger.info("Ready for real-time inference.")

        # Real-time loop
        while True:
            # Block until we have a new chunk from the mic
            chunk = q_in.get()
            
            if chunk is None:  # Sentinel to stop
                break
                
            # Shift buffer left and append new chunk
            context_buffer[:-BLOCK_SIZE] = context_buffer[BLOCK_SIZE:]
            context_buffer[-BLOCK_SIZE:] = chunk
            
            # Inference
            t0 = time.perf_counter()
            with torch.no_grad():
                # [1, 1, T]
                wav_tensor = torch.from_numpy(context_buffer).unsqueeze(0).unsqueeze(0).to(device)
                enhanced = model(wav_tensor).squeeze().cpu().numpy()
            
            # Extract the newest enhanced part (the tail of the enhanced window)
            # Note: Because Conv-TasNet preserves temporal alignment, the output corresponds
            # 1:1 with the input. We take the last BLOCK_SIZE samples.
            new_enhanced = enhanced[-BLOCK_SIZE:]
            
            # Put to playback queue
            q_out.put(new_enhanced)
            
            rtf = (time.perf_counter() - t0) / (BLOCK_SIZE / TARGET_SR)
            if rtf > 1.0:
                logger.warning("Inference too slow! RTF=%.2f", rtf)

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    logger.info("Starting Audio Stream (Sample Rate: %d Hz, Block Size: %d)", TARGET_SR, BLOCK_SIZE)
    logger.info("Press Ctrl+C to stop.")

    try:
        with sd.Stream(device=(args.input_device, args.output_device),
                       samplerate=TARGET_SR, blocksize=BLOCK_SIZE,
                       dtype='float32', channels=1, callback=audio_callback):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error("Audio stream error: %s", e)
    finally:
        q_in.put(None) # stop inference thread
        inf_thread.join()

if __name__ == "__main__":
    main()
