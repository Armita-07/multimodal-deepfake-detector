# Module 5: Audio Deepfake & Synthetic Speech Detection

## 1. Executive Summary

Module 5 provides end-to-end synthetic speech and voice clone detection for the **Multimodal Deepfake & AI-Generated Content Detector**. Using a fine-tuned **Wav2Vec 2.0** self-supervised foundational speech model, it analyzes raw acoustic waveforms to classify audio as **Authentic (Real)** or **AI-Generated / Spoofed (Fake)**.

Evaluated on the benchmark **ASVspoof 2019 Logical Access (LA)** test split across 4,000 speech samples (2,000 real, 2,000 spoofed), the classifier achieved **99.83% overall accuracy** with **100% recall on fake speech** (0 deepfakes missed).

---

## 2. Experimental Results & Performance Metrics

### Table 1: Test Set Classification Report (ASVspoof 2019 LA)
| Class | Precision | Recall | F1-Score | Support (Audio Files) |
| :--- | :---: | :---: | :---: | :---: |
| **Real (Bonafide)** | **1.00** (100.0%) | **0.9965** (99.65%) | **1.00** | 2,000 |
| **Fake (Spoofed)** | **0.9965** (99.65%) | **1.00** (100.0%) | **1.00** | 2,000 |
| **Overall Accuracy** | — | — | **0.9983 (99.83%)** | **4,000** |
| **Macro Average** | 1.00 | 1.00 | 1.00 | 4,000 |
| **Weighted Average** | 1.00 | 1.00 | 1.00 | 4,000 |

### Table 2: Confusion Matrix Analysis
| Actual \ Predicted | Predicted REAL | Predicted FAKE | Total Samples |
| :--- | :---: | :---: | :---: |
| **Actual REAL (Bonafide)** | **1,993** (True Negative) | **7** (False Positive) | 2,000 |
| **Actual FAKE (Spoof)** | **0** (False Negative) | **2,000** (True Positive) | 2,000 |
| **Total** | 1,993 | 2,007 | 4,000 |

---

## 3. Key Discussion & Academic Insights

1. **Zero False Negatives ($FN = 0$)**:
   - The model detected **all 2,000 spoofed samples**, yielding a **100% Fake Recall**. In security and deepfake forensics, false negatives represent the most hazardous failure mode (allowing a malicious voice clone or impersonation attack to bypass verification undetected).
2. **Extremely Low False Alarm Rate ($FPR = 0.35\%$)**:
   - Out of 2,000 genuine human speech recordings, only **7 samples** were erroneously classified as fake. A high precision on real audio prevents genuine callers and recordings from being falsely penalized.
3. **Superiority of Raw Waveform Foundations over Spectrograms**:
   - Conventional audio classifiers convert waveforms into 2D Log-Mel Spectrograms or MFCCs before applying computer vision CNNs. This lossy time-frequency transformation discards fine-grained phase inconsistencies and high-frequency harmonic artifacts produced by modern neural vocoders (e.g., HiFi-GAN, WaveGlow, Diffusion models). Wav2Vec 2.0 processes raw $16\text{ kHz}$ waveforms directly, capturing sub-millisecond vocoder synthesis artifacts.

---

## 4. Technical Architecture & Methodology

### 4.1 Pipeline Architecture
```
Incoming Audio (MP3 / WAV / FLAC / OGG)
                   │
                   ▼
  1. Audio Ingestion & Resampling (16 kHz Mono via Librosa)
                   │
                   ▼
  2. Fixed Temporal Windowing (Max 10s clip / 4s training slice)
                   │
                   ▼
  3. Feature Extraction (Wav2Vec 2.0 CNN Feature Encoder - Frozen)
                   │
                   ▼
  4. Contextual Encoding (12-layer Transformer - 768-dim embeddings)
                   │
                   ▼
  5. Mean Temporal Pooling (Aggregates sequence frames into fixed vector)
                   │
                   ▼
  6. Binary Classification Head (Linear 768→128 → ReLU → Dropout(0.3) → Linear 128→2)
                   │
                   ▼
  7. Softmax Normalization → P(Fake) ∈ [0.0, 1.0]
```

### 4.2 Technical Parameters
| Parameter | Setting / Value |
| :--- | :--- |
| **Target Sample Rate** | $16,000\text{ Hz}$ ($16\text{ kHz}$, single-channel mono) |
| **Foundation Backbone** | `facebook/wav2vec2-base` (~95M parameters) |
| **Feature Encoder** | Multi-layer temporal CNN (frozen to retain low-level acoustic representations) |
| **Classifier Head** | 2-layer MLP with intermediate projection to 128 dimensions and 30% dropout |
| **Loss Function** | Cross-Entropy Loss |
| **Optimizer** | AdamW ($\text{learning rate} = 10^{-5}$) |
| **Batch Size & Epochs**| Batch size = 8, 5 Epochs |
| **Training Hardware** | NVIDIA Tesla T4 GPU (Google Colab) |
| **Inference Weight Footprint** | ~15 MB (classification head state dictionary) |

---

## 5. Dataset Description: ASVspoof 2019 Logical Access (LA)

The **ASVspoof 2019 Logical Access** benchmark is the gold standard for voice anti-spoofing and deepfake speech evaluation:
- **Genuine Speech**: Human speech recorded from various speakers across multiple acoustic environments.
- **Spoofed Speech**: Created using state-of-the-art speech synthesis technologies:
  - **Text-to-Speech (TTS)**: Autoregressive and non-autoregressive neural architectures.
  - **Voice Conversion (VC)**: Spectral mapping, cycle-consistent generative networks, and neural vocoder reconstruction.
- **Protocol Evaluation**: Strict partition protocol preventing speaker identity overlap between training and evaluation partitions.

---

## 6. Implementation & Code Artifacts

- `backend/src/audio/model.py`: Implements `AudioClassifier` and lazy loader with GPU/CPU fallback.
- `backend/src/audio/predict.py`: Public API endpoint `predict_audio(path: str) -> float` satisfying the system interface contract.
- `backend/data_pipeline/prepare_asvspoof.py`: Automated protocol parser sorting ASVspoof into `real/` and `fake/` splits.
- `backend/notebooks/train_audio_clf.py`: Self-contained training, evaluation, and plotting notebook.
- `backend/test_audio.py`: Standalone CLI batch evaluation utility for local verification.
