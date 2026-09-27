# IntelliComm ANC System - Transmitter Side

## Overview
**IntelliComm** is a real-time, waveform-domain active noise cancellation (ANC) and speech enhancement system designed for military and field communication. It operates around a powerful **Conv-TasNet** deep neural network to process noisy incoming audio and enhance it for clear operator communication.

This repository holds the code for the **Transmitter Side**, which handles everything from training the deep learning models, to evaluating performance, and performing real-time inferences.

## Key Features
* **AI-Powered Speech Enhancement:** Uses a Conv-TasNet architecture to cleanly separate speech from heavy background noises.
* **Real-time Inference:** Built-in scripts to simulate or perform real-time UDP-based audio streaming and inference.
* **Robust Checkpointing & Training:** Highly configurable training loop with early stopping, dynamic loss tracking, and metrics logging.
* **Evaluation Metrics:** Built-in `stoi_analysis.py` and evaluation tools to measure Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and STOI.

## Repository Structure
* `train.py`: Main script for training the Conv-TasNet model. Configured with early stopping and automatic best-model checkpointing.
* `eval.py`: Script to evaluate a trained model on the test dataset and generate metrics.
* `infer.py`: Run offline inference on audio files using the trained model.
* `realtime_infer.py`: Simulates or performs real-time audio enhancement.
* `model.py` / `loss.py`: PyTorch definitions for the Conv-TasNet architecture and SI-SNR loss functions.
* `dataset_loader.py` / `preprocessing.py`: Tools for loading and pre-processing the noisy/clean paired audio datasets.
* `udp_transmitter.py` / `udp_receiver.py`: Networking scripts for streaming real-time audio between field and base station environments.

## How to Use

### 1. Training the Model
To start training the model with the dataset, run:
```bash
python train.py --epochs 30 --batch_size 2
```
Checkpoints will be saved automatically to the `checkpoints/` directory. You can resume training using `--resume checkpoints/latest.pth`.

### 2. Evaluating
Run the evaluation script to test the model against the test dataset:
```bash
python eval.py
```

### 3. Inference
To enhance a single audio file:
```bash
python infer.py --input noisy_audio.wav --output enhanced_audio.wav --checkpoint checkpoints/best.pth
```

## Documentation
For deeper dives into the architecture and setup, please refer to:
* `system_architecture.md`: Full architectural diagrams, pipeline flows, and hardware integration maps.
* `walkthrough.md`: Quickstart information and previous sprint completions.
