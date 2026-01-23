# BrainStorm

## Problem Statement

The core promise of a Brain-Computer Interface (BCI) lies in its ability to translate thought into action. This requires Machine Learning models that can decipher the complex, non-linear relationship between raw neural firing patterns and user intent. However, achieving high accuracy is only half the battle.

To be viable for real-world daily use, these models must run on "edge devices"—implanted or wearable hardware with strict limitations on power, memory, and compute capacity. A massive server-grade model is useless if it cannot run on the chip sitting next to the brain.

This creates a fundamental engineering tension: **increasing model size often improves performance, but real-world constraints demand efficiency**. Finding the sweet spot where high-fidelity decoding meets ultra-low-latency and minimal compute footprint is the central challenge.

**The Challenge:** In this track, you will develop a neural decoder that is accurate, fast, and lightweight. Using high-density ECoG recordings from animal auditory cortex, you must build a model capable of decoding auditory stimuli in real-time.

## Constraints & Requirements

**The Task:** This is a streaming classification task. You must predict the specific sound frequency (in Hz) presented to the animal, or the absence of sound (0 Hz), for every single time-sample in the dataset.

- **Data Source**: Raw voltage recordings from a 1024-channel micro-ECoG array.
    
    > **Hint:** Neural information is often sparse and frequency-specific. Successful teams often look beyond raw voltage, exploring spectral power changes in specific bands to extract features relevant to the auditory cortex.

- **Small Data Regime**: You will be provided with a limited training set. Your model must be data-efficient, capable of learning robust patterns from just a few examples to generalize to the unseen test set.

- **Strict "Streaming" Inference**: Real-time BCIs cannot "see the future." Your model must operate **causally**:
  - Inference must happen one sample at a time
  - You can track history (past states), but you **cannot** use future data points
  - Non-causal filters (e.g., standard bidirectional filtering) are not allowed
  
  The evaluation harness enforces this by feeding your model data sequentially.

## Scoring

Your submission is scored on a **0–100 scale** based on three weighted factors:

| Weight | Metric | Description |
|--------|--------|-------------|
| **50%** | Balanced Accuracy | Primary metric. Accuracy across all classes, weighted by frequency. |
| **25%** | Prediction Lag | Time delay between stimulus onset and correct classification. |
| **25%** | Model Size | Size of your saved model file on disk (in MB). |

**Key insight:** Latency and size scoring are **non-linear**—small, fast models are rewarded disproportionately. See the [Evaluation Guide](evaluation.md) for formulas and optimization strategies.
