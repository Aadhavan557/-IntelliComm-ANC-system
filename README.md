# IntelliComm ANC System - Transmitter Side

## Overview
**IntelliComm** is a real-time, waveform-domain active noise cancellation (ANC) and speech enhancement system designed for military and field communication. Built around a powerful **Conv-TasNet** deep neural network, it processes noisy incoming audio and isolates human speech for clear operator communication.

This repository holds the code for the **Transmitter Side** (and core processing components), which handles everything from training the deep learning models to evaluating performance and performing real-time inference.

## 🚀 Key Features
* **AI-Powered Speech Enhancement:** Uses a compact Conv-TasNet architecture (10.79M parameters) that operates directly on the waveform (time-domain) for lower latency than frequency-domain masking.
* **Highly Optimized:** Utilizes Automatic Mixed Precision (AMP) to run smoothly on lower-tier hardware (like an NVIDIA RTX 3050 laptop GPU with 6GB VRAM), keeping memory footprint around ~173 MB during inference.
* **Loss Function:** Optimized using a combination of Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and L1 loss to regulate waveform amplitude.
* **Real-time Target Architecture:** Designed to integrate with real-time PyAudio streams and UDP/Socket audio bridges to communicate with field transmitters.
* **Robust Checkpointing:** Dynamic training loop with early stopping, dynamic loss tracking, and metrics logging to ensure the best model is preserved.

## 📊 Performance Achieved (Test Set)
- **SI-SNR:** 18.54 dB
- **SI-SDR:** 18.54 dB
- **STOI (median):** 0.79
- **VRAM Usage:** ~173 MB
- **Model Size:** 10.79 M parameters

## 📂 Repository Structure

### Core Modules
* `model.py`: PyTorch definitions for the Conv-TasNet architecture.
* `loss.py`: SI-SNR, SI-SDR, and combined loss definitions.
* `dataset_loader.py` & `preprocessing.py`: Tools for loading, mixing, and normalizing the noisy/clean paired audio datasets.
* `chunk_builder.py` & `replay_sampler.py`: Subset builders and data sampling for efficient training.

### Entry Points & Scripts
* `train.py`: Main script for training the Conv-TasNet model. Configured with early stopping and automatic best-model checkpointing.
* `eval.py`: Script to evaluate a trained model on the test dataset and generate metrics.
* `infer.py`: Offline inference on single audio files using the trained model.
* `batch_infer.py`: Run batch inference across multiple files or a directory.
* `realtime_infer.py`: Simulates or performs real-time audio enhancement on incoming streams.
* `udp_transmitter.py` & `udp_receiver.py`: Networking scripts for streaming real-time audio over a UDP socket bridge.

## 💻 How to Use

### 1. Training the Model
To start training the model with the dataset (defaults to 10k balanced samples), run:
```bash
python train.py --epochs 30 --batch_size 2
```
Checkpoints will be saved automatically to the `checkpoints/` directory. You can resume training using `--resume checkpoints/latest.pth`.

### 2. Evaluating
Run the evaluation script to test the model against the test dataset and compute STOI/SI-SNR:
```bash
python eval.py
```

### 3. Inference
To enhance a single audio file offline:
```bash
python infer.py --input noisy_audio.wav --output enhanced_audio.wav --checkpoint checkpoints/best.pth
```

### 4. Real-time / Networked Audio
To simulate field communications, start the UDP receiver to wait for noisy audio, then start the transmitter to send audio across the network:
```bash
# Terminal 1 (Receiver side / Base Station)
python udp_receiver.py

# Terminal 2 (Transmitter side / Field)
python udp_transmitter.py --input test_audio.wav
```

## 📖 Documentation
For deeper dives into the architecture and setup, please refer to:
* [`system_architecture.md`](./system_architecture.md): Full architectural diagrams, pipeline flows, hardware integration maps, and exact model hyperparameters.
* [`walkthrough.md`](./walkthrough.md): Quickstart information and previous sprint completions.
