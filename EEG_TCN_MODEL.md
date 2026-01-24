# BrainStorm 2026 Track 1 - Neural Decoder Implementation

## Overview

This project implements a neural decoder for real-time auditory stimulus classification from ECoG (electrocorticography) recordings. The goal is to predict sound frequency (Hz) or silence at every timestep from a 1024-channel micro-ECoG array.

## Current Best Model: EEG-TCNet

**Score: 61.1/100** | Accuracy: 48.6% | Latency: 35ms | Size: 257KB

### Architecture

EEG-TCNet combines two proven architectures for brain signal processing:

```
Input: (batch, seq_len, (B+1), 16, 16 spatial grids)
              │
              ▼
┌─────────────────────────────────────────────┐
│         STAGE 1: EEGNet Block (3D)          │
│  ─────────────────────────────────────────  │
│  Temporal Conv3D ((B+1)→F1, k=32)           │
│      │                                      │
│  Depthwise Spatial Conv3D (F1→F1*D)         │
│  - Kernel over full 16x16 grid              │
│  - Learns spatial patterns across cortex    │
│      │                                      │
│  Pointwise Conv3D (F1*D→F2)                 │
│  - Mixes features across channels           │
│      │                                      │
│  BatchNorm + ELU + Dropout(0.4)             │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│         STAGE 2: TCN Blocks (x3)            │
│  ─────────────────────────────────────────  │
│  Dilated Causal Convolutions                │
│  - Dilation rates: 1, 2, 4                  │
│  - Kernel size: 4                           │
│  - Channels: 32                             │
│  - Receptive field: ~128ms                  │
│      │                                      │
│  Residual connections for each block        │
│  BatchNorm + ELU + Dropout(0.4)             │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│         STAGE 3: Classifier                 │
│  ─────────────────────────────────────────  │
│  Linear (32 → 9 classes)                    │
│  Classes: 0Hz (silence), 120Hz, 224Hz,      │
│           421Hz, 789Hz, 1479Hz, 2772Hz,     │
│           5195Hz, 9736Hz                    │
└─────────────────────────────────────────────┘
              │
              ▼
        Output: Predicted frequency
```

### Preprocessing (Input Features)

Raw voltages are transformed into bandpower features, mapped to 31x32 spatial grids,
an electrode mask channel is appended, and the grid is padded to 32x32 then pooled
to 16x16. The final per-timestep tensor has shape `(B+1, 16, 16)` and is flattened
for model input.

### Key Design Decisions

1. **Depthwise Separable Convolutions**: Reduces parameters by ~10x compared to standard convolutions while maintaining performance.

2. **Dilated Causal Convolutions**: Enables the model to "see" further back in time without increasing parameters. With dilations [1, 2, 4] and kernel size 4, the receptive field spans ~128ms of history.

3. **Causal Architecture**: All operations only use past data, ensuring real-time streaming compatibility.

4. **Class Weighting**: Handles the severe class imbalance (67% silence, 33% sounds) by weighting the loss function inversely proportional to class frequency.

5. **Spatial Grid Input**: Features are mapped to 16x16 spatial grids with an electrode mask channel to preserve geometry and indicate missing electrodes.
6. **Input Normalization**: Bandpower features are z-scored using training statistics and reused for validation/inference.

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| F1 | 16 | Temporal filter count |
| D | 2 | Depth multiplier |
| F2 | 32 | Separable filter count |
| temporal_kernel | 32 | Temporal kernel size (samples) |
| tcn_channels | 32 | TCN hidden dimension |
| tcn_kernel_size | 4 | TCN kernel size |
| tcn_layers | 3 | Number of TCN blocks |
| dropout | 0.4 | Dropout rate |
| context_window | 128 | Samples of history for inference |
| height, width | 16, 16 | Pooled spatial grid size |
| input_channels | B+1 | Bandpower channels + electrode mask |
| learning_rate | 1e-3 | Adam learning rate |
| epochs | 40 | Training epochs |

---

## Results Comparison

| Model | Accuracy | Lag | Size | Total Score |
|-------|----------|-----|------|-------------|
| MLP Baseline | 33.8% (16.9 pts) | 65ms (11.5 pts) | 1.01MB (11.1 pts) | **39.5** |
| EEG-TCNet v1 | 39.6% (19.8 pts) | 42ms (15.1 pts) | 104KB (23.0 pts) | **58.0** |
| **EEG-TCNet v2** | **48.6%** (24.3 pts) | **35ms** (16.4 pts) | **257KB** (20.4 pts) | **61.1** |

### Score Breakdown

The competition uses a weighted scoring formula:
- **50%** Balanced Accuracy (linear scaling)
- **25%** Prediction Lag (exponential decay: `exp(-6 × lag/500) × 25`)
- **25%** Model Size (exponential decay: `exp(-4 × size_mb/5) × 25`)

---

## Data Characteristics

| Property | Value |
|----------|-------|
| Channels | 1024 (micro-ECoG grid) |
| Sampling Rate | 1000 Hz |
| Training Samples | 90,386 (~90 seconds) |
| Validation Samples | 22,596 (~23 seconds) |
| Classes | 9 (silence + 8 frequencies) |
| Class Imbalance | 67% silence, 33% sounds |
| Signal Range | [-2193, 1555] μV |
| Avg Segment Duration | 275ms |

---

## Potential Improvements

### 1. Spectral Feature Engineering (High Priority)

The competition hints suggest that **spectral power changes** in specific frequency bands are highly informative for ECoG decoding.

**Recommended approach:**
```python
# Extract power in key frequency bands (causal filtering)
bands = {
    'high_gamma': (70, 150),   # Most informative for ECoG
    'gamma': (30, 70),
    'beta': (13, 30),
}

# Use causal IIR bandpass filters (e.g., Butterworth)
# Compute instantaneous power via Hilbert transform or squared signal
```

**Expected impact:** +5-15% accuracy improvement based on literature.

**Why it works:** Raw voltage contains noise and irrelevant oscillations. High-gamma band (70-150 Hz) specifically correlates with local neural activity and is the gold standard for ECoG decoding.

### 2. Channel Selection/Reduction (Medium Priority)

Not all 1024 channels are equally informative. Some electrodes may be outside the auditory cortex or have poor signal quality.

**Approaches:**
- Mutual information between each channel and labels
- Gradient-based importance from trained model
- PCA/spatial ICA for dimensionality reduction

**Expected impact:** Faster inference, potentially better accuracy by removing noise.

### 3. Larger Temporal Context (Medium Priority)

Current receptive field is ~128ms. Sound stimuli may have longer temporal signatures.

**Approaches:**
- Increase TCN layers (4-5 layers)
- Increase dilation rates ([1, 2, 4, 8, 16])
- Larger context window (256-512 samples)

**Trade-off:** Larger context = more parameters = larger model size.

### 4. Data Augmentation (Medium Priority)

Limited training data (~90 seconds) may cause overfitting.

**Approaches:**
- Time warping (slight speed variations)
- Channel dropout (randomly zero some electrodes)
- Gaussian noise injection
- Mixup between samples of same class

**Expected impact:** Better generalization, reduced overfitting.

### 5. Attention Mechanisms (Lower Priority)

Add channel attention to dynamically weight electrode importance.

**Approaches:**
- Squeeze-and-Excitation blocks
- Self-attention over channels
- Cross-attention between time and space

**Trade-off:** Increases model complexity and size.

### 6. Model Compression (If Size Becomes Issue)

Current model is 257KB, well under the 25MB limit. If larger models are needed:

**Approaches:**
- Quantization (float32 → int8): ~4x size reduction
- Pruning: Remove small weights
- Knowledge distillation: Train small model to mimic large one

---

## File Structure

```
brainstorm/
├── ml/
│   ├── base.py              # Abstract base class (DO NOT MODIFY)
│   ├── mlp.py               # Simple MLP baseline
│   ├── logistic_regression.py
│   ├── eeg_tcnet.py         # Current best model
│   └── metrics.py           # Evaluation metrics (DO NOT MODIFY)
├── evaluation.py            # Model evaluation (DO NOT MODIFY)
└── ...

eeg_tcnet.pt                 # Trained model weights
model_metadata.json          # Model path and class info
```

---

## Usage

### Training
```python
from brainstorm.loading import load_raw_data, load_channel_coordinates
from brainstorm.preprocessing import design_bandpass_sos, preprocess_spatial_features
from brainstorm.ml.eeg_tcnet import EEGTCNet

train_features, train_labels = load_raw_data("./data", step="train")
coords = load_channel_coordinates()

bands = [(1, 4), (4, 8), (8, 12), (13, 30), (30, 55), (65, 100), (100, 150), (150, 250)]
sos_bank = design_bandpass_sos(bands, fs_hz=1000)
train_spatial, stats = preprocess_spatial_features(
    train_features.values, coords, sos_bank, fs_hz=1000, tau_s=0.05
)

X = train_spatial.reshape(train_spatial.shape[0], -1)

model = EEGTCNet(
    input_channels=train_spatial.shape[1],
    height=train_spatial.shape[2],
    width=train_spatial.shape[3],
    F1=16, D=2, F2=32,
    tcn_channels=32, tcn_layers=3,
    dropout=0.4, context_window=128
)

model.fit(
    X=X,
    y=train_labels["label"].values,
    epochs=40,
    batch_size=32,
    seq_len=128
)
```

### Inference
```python
model = EEGTCNet.load()
prediction = model.predict(sample)  # sample shape: (1024,)
```

---

## References

1. **EEG-TCNet**: Ingolfsson et al., "EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces" (2020) - [arXiv:2006.00622](https://arxiv.org/abs/2006.00622)

2. **EEGNet**: Lawhern et al., "EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces" (2018) - [arXiv:1611.08024](https://arxiv.org/abs/1611.08024)

3. **TCN**: Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018) - [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)

---

## Next Steps

1. **Immediate**: Implement high-gamma band power extraction (70-150 Hz)
2. **Short-term**: Channel selection based on mutual information
3. **Medium-term**: Experiment with larger temporal context
4. **If needed**: Model compression via quantization
