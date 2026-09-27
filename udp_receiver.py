"""
udp_receiver.py
===============
IntelliComm – AI-Based Speech Enhancement for Military/Field Communication
UDP Audio Receiver & Enhancer

Simulates the base station receiver. Listens for incoming UDP audio packets,
enhances them using Conv-TasNet, and plays the output.

Usage:
  python udp_receiver.py --port 9999
  python udp_receiver.py --port 9999 --output-device 2
"""

import argparse
import logging
import queue
import socket
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
BLOCK_SIZE = 8000     # 0.5 seconds of new audio per inference block
UDP_PACKET_SIZE = 1600 # 100ms
WINDOW_SIZE = 32000   # 2.0 seconds of context for the model

def main():
    parser = argparse.ArgumentParser(description="IntelliComm — UDP Audio Receiver & Enhancer")
    parser.add_argument("--host", default="0.0.0.0", help="Listen IP address")
    parser.add_argument("--port", type=int, default=9999, help="Listen UDP port")
    parser.add_argument("--output-device", type=int, default=None, help="Output audio device ID")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Model checkpoint path")
    args = parser.parse_args()

    # Setup UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    # Non-blocking or timeout to allow graceful exit
    sock.settimeout(1.0)

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
    # The UDP thread puts small packets into a buffer.
    # When the buffer reaches BLOCK_SIZE, it puts it into q_in.
    q_in = queue.Queue()
    q_out = queue.Queue()

    # Playback callback
    def audio_callback(outdata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        try:
            # We expect outdata to be of size BLOCK_SIZE (e.g. 8000 frames)
            out_samples = q_out.get_nowait()
            outdata[:, 0] = out_samples
        except queue.Empty:
            # Underrun
            outdata.fill(0)

    # Inference thread
    def inference_loop():
        context_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        
        logger.info("Warming up GPU...")
        dummy_in = torch.zeros(1, 1, WINDOW_SIZE, device=device)
        with torch.no_grad():
            _ = model(dummy_in)
        logger.info("Ready for real-time inference.")

        while True:
            chunk = q_in.get()
            if chunk is None:
                break
                
            context_buffer[:-BLOCK_SIZE] = context_buffer[BLOCK_SIZE:]
            context_buffer[-BLOCK_SIZE:] = chunk
            
            with torch.no_grad():
                wav_tensor = torch.from_numpy(context_buffer).unsqueeze(0).unsqueeze(0).to(device)
                enhanced = model(wav_tensor).squeeze().cpu().numpy()
            
            new_enhanced = enhanced[-BLOCK_SIZE:]
            q_out.put(new_enhanced)

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    logger.info("IntelliComm Base Station Receiver Started")
    logger.info("Listening on UDP %s:%d", args.host, args.port)

    # UDP Receiver Loop
    def network_loop():
        # Buffer to accumulate small UDP packets into BLOCK_SIZE chunks
        buffer = []
        buffer_len = 0

        while True:
            try:
                data, addr = sock.recvfrom(UDP_PACKET_SIZE * 4) # 4 bytes per float32
                samples = np.frombuffer(data, dtype=np.float32)
                buffer.append(samples)
                buffer_len += len(samples)

                if buffer_len >= BLOCK_SIZE:
                    # We have enough data for an inference block
                    full_block = np.concatenate(buffer)
                    chunk = full_block[:BLOCK_SIZE]
                    q_in.put(chunk)
                    
                    # Keep remainder
                    remainder = full_block[BLOCK_SIZE:]
                    buffer = [remainder]
                    buffer_len = len(remainder)

            except socket.timeout:
                continue
            except Exception as e:
                logger.error("UDP loop error: %s", e)
                break

    net_thread = threading.Thread(target=network_loop, daemon=True)
    net_thread.start()

    # Playback Stream
    try:
        with sd.OutputStream(device=args.output_device, samplerate=TARGET_SR, 
                             blocksize=BLOCK_SIZE, dtype='float32', channels=1, 
                             callback=audio_callback):
            logger.info("Audio playback active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping receiver...")
    except Exception as e:
        logger.error("Playback error: %s", e)
    finally:
        sock.close()
        q_in.put(None)
        inf_thread.join()

if __name__ == "__main__":
    main()
