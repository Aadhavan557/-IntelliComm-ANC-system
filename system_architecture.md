# IntelliComm — System Architecture

**AI-Based Speech Enhancement for Military / Field Communication**
`Version 1.0 | Stage 6 of 9`

---

## 1. System Overview

IntelliComm is a real-time, waveform-domain speech enhancement system built around a **Conv-TasNet** deep neural network. It operates on the **receiver side** of a communication link: noisy incoming audio is enhanced before being heard by the operator.

```mermaid
graph LR
    subgraph FIELD["🎙️ Field (Transmitter Side)"]
        MIC["Microphone\n(Speaker in noise)"]
        TX["Radio / Comms\nTransmitter"]
    end

    subgraph CHANNEL["📡 Comms Channel"]
        RF["RF / IP Link\n(adds noise, distortion)"]
    end

    subgraph BASE["🖥️ Base Station (Receiver Side)"]
        RX["Radio / Comms\nReceiver"]
        PRE["Audio\nPreprocessing\n(16 kHz, mono)"]
        AI["Conv-TasNet\nEnhancement\n(GPU Inference)"]
        POST["Post-processing\n(normalise, clip)"]
        SPK["Operator\nHeadset / Speaker"]
    end

    MIC --> TX --> RF --> RX --> PRE --> AI --> POST --> SPK

    style FIELD fill:#1a1a2e,color:#eee,stroke:#4a90d9
    style CHANNEL fill:#16213e,color:#eee,stroke:#e94560
    style BASE fill:#0f3460,color:#eee,stroke:#4a90d9
    style AI fill:#e94560,color:#fff,stroke:#fff
```

---

## 2. Full Pipeline — Data Flow

```mermaid
flowchart TD
    A["🎙️ Noisy Input Audio\nWAV / FLAC / PCM stream"]
    B["Resample → 16,000 Hz\nDownmix → Mono\nNormalize amplitude"]
    C["Segment into 2-second chunks\n32,000 samples per chunk"]
    D["Encoder\nConv1d: 1 → 256 ch\nkernel=20, stride=10"]
    E["Bottleneck\n1×1 Conv 256 → 256\nGlobal Layer Norm"]
    F["TCN Separator\n4 repeats × 8 blocks\nDilated depthwise convs"]
    G["Mask Estimation\n1×1 Conv + Sigmoid\n256-channel soft mask"]
    H["Element-wise Masking\nEncoder features × Mask"]
    I["Decoder\nConvTranspose1d: 256 → 1\nkernel=20, stride=10"]
    J["Overlap-Add Reconstruction\n50% overlap, Hann window"]
    K["✅ Enhanced Speech\n16,000 Hz mono WAV"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

    style A fill:#2c2c54,color:#eee
    style K fill:#218c74,color:#eee
    style F fill:#e94560,color:#fff
    style G fill:#e94560,color:#fff
```

---

## 3. Conv-TasNet Architecture Detail

```mermaid
graph TD
    subgraph INPUT["Input: [B, 1, T]"]
        IN["Waveform\nB=batch, T=32000 samples"]
    end

    subgraph ENC["Encoder"]
        E1["Conv1d\n1→N=256, kernel=L=20, stride=10\n+ ReLU"]
    end

    subgraph SEP["Separator (TCN)"]
        S1["Global LayerNorm\n+ Bottleneck Conv1d 256→256"]
        S2["Repeat 1/4\nBlocks 1–8\nDilation: 1,2,4,8,16,32,64,128"]
        S3["Repeat 2/4\nBlocks 9–16\nDilation: 1,2,4,8,16,32,64,128"]
        S4["Repeat 3/4\nBlocks 17–24"]
        S5["Repeat 4/4\nBlocks 25–32"]
        S6["Skip-connection sum\nall 32 blocks"]
        S7["Conv1d 256→256\n+ Sigmoid → Mask"]
    end

    subgraph DEC["Decoder"]
        D1["Encoder features\n× Mask [element-wise]"]
        D2["ConvTranspose1d\n256→1, kernel=20, stride=10"]
        D3["Length match / pad\nOutput: [B, 1, T]"]
    end

    IN --> E1 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 --> D1
    E1 --> D1
    D1 --> D2 --> D3

    style SEP fill:#1a1a2e,color:#eee,stroke:#e94560
    style ENC fill:#0f3460,color:#eee,stroke:#4a90d9
    style DEC fill:#0f3460,color:#eee,stroke:#4a90d9
```

### Model Hyperparameters

| Symbol | Parameter | Value |
|:---:|:---|:---:|
| **N** | Encoder channels | 256 |
| **L** | Encoder kernel size | 20 (1.25 ms) |
| **B** | TCN bottleneck channels | 256 |
| **H** | TCN hidden (depthwise) channels | 512 |
| **P** | TCN depthwise kernel size | 3 |
| **X** | Conv blocks per TCN repeat | 8 |
| **R** | TCN repeats | 4 |
| — | **Total parameters** | **10,792,000** |
| — | Receptive field | ~1.3 s |
| — | Sample rate | 16,000 Hz |
| — | Segment length | 2.0 s (32,000 samples) |

---

## 4. Software Component Map

```mermaid
graph TD
    subgraph SCRIPTS["Entry Points"]
        TR["train.py\nTraining pipeline"]
        EV["eval.py\nEvaluation + metrics"]
        IN["infer.py\nSingle-file inference"]
    end

    subgraph CORE["Core Modules"]
        MD["model.py\nConvTasNet, build_model()"]
        LS["loss.py\nSI-SNR, SI-SDR, combined_loss()"]
        DL["dataset_loader.py\nSpeechEnhancementDataset\nNoisyPairDataset"]
        PP["preprocessing.py\nload_segment(), mix_with_snr()\nprobe_audio_file()"]
        RS["replay_sampler.py\nReplay buffer sampler"]
        CB["chunk_builder.py\nDataset subset builder"]
    end

    subgraph ARTIFACTS["Outputs"]
        CK["checkpoints/\nbest.pth, latest.pth\nepoch_XX.pth"]
        LG["logs/\ntraining_log.csv"]
        RE["results/\naudio/, plots/\nfinal_report.txt\nfinal_evaluation.csv"]
        DS["dataset_subset/\ntrain_10000/manifest.csv"]
    end

    TR --> MD & LS & DL & PP & RS
    EV --> MD & LS & DL & PP
    IN --> MD & PP
    DL --> PP
    CB --> DS

    TR --> CK & LG
    EV --> RE
    IN --> RE

    style MD fill:#e94560,color:#fff
    style SCRIPTS fill:#0f3460,color:#eee,stroke:#4a90d9
    style CORE fill:#16213e,color:#eee,stroke:#e94560
    style ARTIFACTS fill:#1a1a2e,color:#eee,stroke:#4a90d9
```

---

## 5. Training Pipeline

```mermaid
sequenceDiagram
    participant DS as Dataset (10k samples)
    participant DL as DataLoader
    participant MD as Conv-TasNet Model
    participant LF as Loss (SI-SNR + L1)
    participant OPT as AdamW + AMP
    participant SCH as CosineAnnealingWarmRestarts
    participant CK as Checkpoint

    loop Each Epoch (30 total)
        DS->>DL: Shuffle + batch (B=2)
        DL->>MD: noisy [2,1,32000] → CUDA
        MD->>LF: enhanced [2,1,32000]
        LF->>OPT: combined_loss (SI-SNR + 0.1×L1)
        OPT->>MD: backward + grad clip (norm=5.0)
        OPT->>SCH: step()
        MD->>CK: save latest.pth + epoch_XX.pth
        Note over CK: If val_loss improved → best.pth
    end
```

---

## 6. Evaluation Pipeline

```mermaid
flowchart LR
    A["checkpoints/best.pth\n10.79M params"] --> B["Load model\n→ CUDA eval mode"]
    C["Dataset test split\n11,016 segments\n80/10/10 deterministic split"] --> D["Random sample\n500 segments (seed=42)"]
    B --> E["Inference loop\nbatch_size=4"]
    D --> E
    E --> F1["SI-SNR: 18.54 dB ✅"]
    E --> F2["SI-SDR: 18.54 dB ✅"]
    E --> F3["STOI: 0.683 ❌\n(median 0.79)"]
    E --> F4["PESQ: pending\n(needs C++ Build Tools)"]
    E --> G["Audio trios\nresults/audio/\nsample_001_noisy/clean/enhanced"]
    E --> H["Waveform + Spectrogram plots\nresults/plots/"]
    F1 & F2 & F3 & F4 --> I["results/final_report.txt\nresults/final_evaluation.csv"]

    style F1 fill:#218c74,color:#fff
    style F2 fill:#218c74,color:#fff
    style F3 fill:#c0392b,color:#fff
    style F4 fill:#7f8c8d,color:#fff
```

---

## 7. Deployment Architecture (Target)

```mermaid
graph TD
    subgraph HARDWARE["Hardware Layer"]
        GPU["NVIDIA RTX 3050\n6 GB VRAM (CUDA 12.4)"]
        CPU["Intel CPU\nDataLoader workers"]
        MEM["16 GB RAM"]
        DISK["NVMe SSD\nDataset + Checkpoints"]
    end

    subgraph RUNTIME["Runtime Layer"]
        PY["Python 3.9"]
        PT["PyTorch 2.6 + CUDA"]
        TA["torchaudio 2.6"]
        NP["NumPy, SciPy, matplotlib"]
        PS["pystoi (STOI metric)"]
    end

    subgraph CURRENT["Current Stage (✅ Complete)"]
        TR["Offline Training\ntrain.py"]
        EV["Offline Evaluation\neval.py"]
        SI["Single-file Inference\ninfer.py"]
    end

    subgraph NEXT["Next Stages (⏭️)"]
        RT["Real-time Audio Pipeline\n(Step 8)"]
        TX["Transmitter Integration\n(Step 9)"]
        API["REST / gRPC API\n(optional)"]
    end

    GPU --> PT
    CPU --> PY
    PT --> TR & EV & SI
    TR --> EV --> SI
    SI --> RT --> TX

    style CURRENT fill:#0f3460,color:#eee,stroke:#218c74
    style NEXT fill:#1a1a2e,color:#eee,stroke:#4a90d9
    style GPU fill:#e94560,color:#fff
```

---

## 8. Dataset Structure

```mermaid
graph TD
    ROOT["Dataset/"]
    ROOT --> CS["clean_speech/\n~9,127 test files\n80% train · 10% val · 10% test\n(fixed seed=42 split)"]
    ROOT --> NO["noise/\n15,693 noise files\n(battlefield, wind, static…)"]
    ROOT --> MX["mixtures/\n10,000 pre-baked\nnoisy+clean pairs"]
    ROOT --> MD["metadata/\nmanifest.json\n(train=73k · val=9.1k · test=9.1k)"]

    DS["dataset_subset/\ntrain_10000/\nmanifest.csv\n(10k balanced training manifest)"]

    ROOT --> MD
    DS -. "pointer to" .-> MX

    style ROOT fill:#16213e,color:#eee
    style MX fill:#0f3460,color:#eee
    style DS fill:#218c74,color:#eee
```

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| **Model domain** | Waveform (time-domain) | No STFT required; lower latency than TF-masking |
| **Architecture** | Conv-TasNet | State-of-art for single-channel SE; compact (10.7M params) |
| **Loss function** | SI-SNR + 0.1 × L1 | Scale-invariant; L1 regularises waveform amplitude |
| **Precision** | AMP (fp16 + fp32) | 30–40% speedup on RTX 3050; no quality loss |
| **Segmentation** | 2 s / 32,000 samples | Fits 6 GB VRAM at batch=2; good temporal context |
| **Overlap-add** | 50% hop, Hann window | Smooth reconstruction of long audio in inference |
| **Split strategy** | 80/10/10 file-level (seed=42) | Prevents speaker leakage across splits |
| **Training data** | 10,000 balanced samples | Memory / time budget for RTX 3050 laptop GPU |

---

## 10. Performance Achieved

| Metric | Value | Target | Status |
|:---|:---|:---|:---|
| SI-SNR (test) | **18.54 dB** | > 15.0 dB | ✅ PASS |
| SI-SDR (test) | **18.54 dB** | — | ✅ |
| STOI (test, avg) | **0.683** | > 0.85 | ❌ (median 0.79) |
| PESQ | Pending | > 2.5 | ⚠️ |
| Parameters | 10.79 M | — | Compact |
| Inference speed | ~4 samples/s (batch=4) | Real-time target → Step 8 | ⏭️ |
| VRAM usage | ~173 MB | < 6 GB | ✅ |

---

## 11. Next Steps (Steps 7–9)

```mermaid
gantt
    title IntelliComm — Remaining Stages
    dateFormat  YYYY-MM-DD
    section Step 7 — Audio Inference
    infer.py single-file (done)     :done, 2026-09-16, 1d
    Batch inference script          :active, 2026-09-17, 1d
    section Step 8 — Real-time Pipeline
    PyAudio / sounddevice stream    :2026-09-18, 2d
    Ring buffer + chunk inference   :2026-09-19, 2d
    Latency measurement             :2026-09-20, 1d
    section Step 9 — Transmitter Integration
    Socket / UDP audio bridge       :2026-09-21, 2d
    End-to-end latency test         :2026-09-23, 1d
    Field demo                      :milestone, 2026-09-24, 0d
```
