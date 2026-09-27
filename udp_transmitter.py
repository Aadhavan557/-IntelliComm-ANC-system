"""
udp_transmitter.py
==================
IntelliComm – AI-Based Speech Enhancement for Military/Field Communication
UDP Audio Transmitter

Simulates the field radio transmitter. Captures audio from the microphone
and sends it over a UDP socket in small PCM packets.

Usage:
  python udp_transmitter.py --host 127.0.0.1 --port 9999
"""

import argparse
import logging
import socket
import sys
import time

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TARGET_SR = 16000
BLOCK_SIZE = 1600  # 100ms per packet at 16kHz
PACKET_FORMAT = 'float32'

def main():
    parser = argparse.ArgumentParser(description="IntelliComm — UDP Audio Transmitter")
    parser.add_argument("--host", default="127.0.0.1", help="Target IP address")
    parser.add_argument("--port", type=int, default=9999, help="Target UDP port")
    parser.add_argument("--input-device", type=int, default=None, help="Input audio device ID")
    args = parser.parse_args()

    # Setup UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_addr = (args.host, args.port)

    logger.info("IntelliComm Field Transmitter Started")
    logger.info("Target: %s:%d", args.host, args.port)
    logger.info("Sample Rate: %d Hz | Packet size: %d samples", TARGET_SR, BLOCK_SIZE)

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        
        # indata is shape (frames, 1) and dtype float32
        # Convert to bytes and send over UDP
        packet_bytes = indata.tobytes()
        
        try:
            sock.sendto(packet_bytes, target_addr)
        except OSError as e:
            logger.error("Network error: %s", e)

    try:
        with sd.InputStream(device=args.input_device, samplerate=TARGET_SR, 
                            blocksize=BLOCK_SIZE, dtype=PACKET_FORMAT, 
                            channels=1, callback=audio_callback):
            logger.info("Microphone active. Streaming audio... (Press Ctrl+C to stop)")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping transmitter...")
    except Exception as e:
        logger.error("Audio stream error: %s", e)
    finally:
        sock.close()

if __name__ == "__main__":
    main()
