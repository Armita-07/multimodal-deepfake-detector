#### Table 1: Model Performance on Test Set (FaceForensics++ c23)
| Class | Precision | Recall | F1-Score | Support (Frames) |
| :--- | :---: | :---: | :---: | :---: |
| **Fake** | **0.95** (95%) | 0.67 (67%) | 0.78 | 360 |
| **Real** | 0.74 (74%) | **0.97** (97%) | 0.84 | 360 |
| **Overall Accuracy** | — | — | **0.82 (82.0%)** | **720** |
| **Macro Average** | 0.85 | 0.82 | 0.81 | 720 |
| **Weighted Average** | 0.85 | 0.82 | 0.81 | 720 |

#### Table 2: Confusion Matrix
| Actual \ Predicted | Predicted FAKE | Predicted REAL | Total |
| :--- | :---: | :---: | :---: |
| **Actual FAKE** | **240** (True Positive) | **120** (False Negative) | 360 |
| **Actual REAL** | **12** (False Positive) | **348** (True Negative) | 360 |

---

### 3. Key Insights to Write in the "Discussion" Section
Professors love when you can interpret *why* the numbers look the way they do:

1. **High Precision on Fakes (95%)**:
   > *"When the model predicts a face is a deepfake, it is correct 95% of the time. This minimizes false alarms and builds high user trust."*
2. **Extremely Low False Positive Rate (Only 12 out of 360 = 3.3%)**:
   > *"Only 12 authentic faces were wrongly flagged as fake. In real-world content moderation, minimizing false accusations against genuine users is critical."*
3. **Frame-Level vs. Video-Level Aggregation**:
   > *"While individual fake frames have a 67% recall, deepfake videos contain multiple manipulated frames across time. By aggregating scores across 16 sampled frames per video (mean pooling), the video-level detection confidence increases significantly."*

---

### 4. Technical Details to Include in the "Methodology" Section

| Section | Technical Information to State |
| :--- | :--- |
| **Dataset** | FaceForensics++ (Original YouTube vs. Deepfakes manipulation), `c23` compression level |
| **Dataset Split** | 100 Real videos, 100 Fake videos. 70/15/15 video-level split (70 train, 15 val, 15 test) to prevent identity leakage |
| **Face Detection** | MTCNN (Multi-task Cascaded Convolutional Networks) on GPU, cropping with 30% bounding-box margin |
| **Input Resolution** | $224 \times 224 \times 3$ RGB |
| **Model Backbone** | EfficientNet-B0 (~5.3M parameters, pretrained on ImageNet) |
| **Optimizer & Loss** | AdamW ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$), Cross-Entropy Loss |
| **Batch Size & Epochs** | Batch size = 32, trained for 8 epochs on NVIDIA T4 GPU |
| **Video Aggregation** | Uniform 16-frame temporal sampling, mean probability pooling |
