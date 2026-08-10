# Action-Conditioned Stickman Animation Generator (RNN/LSTM)

This repository contains a deep learning framework designed to generate realistic 2D stickman animations conditioned on specific actions (e.g., *run, walk, jump, boxing, handsup, etc.*). The project is built using TensorFlow and Keras, employing a recurrent auto-regressive pipeline with LSTMs.

---

## Table of Contents
1. [Pipeline & Architectural Overview](#pipeline--architectural-overview)
2. [Optimizations & Technical Improvements](#optimizations--technical-improvements)
3. [Mathematical Foundations & Loss Functions](#mathematical-foundations--loss-functions)
4. [Detailed Q&A (Project Intricacies)](#detailed-qa-project-intricacies)
5. [Viewer FAQs (Anticipated Questions)](#viewer-faqs-anticipated-questions)
6. [Shortcomings & Room for Improvement](#shortcomings--room-for-improvement)
7. [Future Scope](#future-scope)

---

## Pipeline & Architectural Overview

The generation pipeline works as follows:

```
[Raw 3D Joint Dataset] 
          │
          ▼
[Extract 13 2D Joints]
          │
          ▼
[Normalize: Hip to Origin, Torso to 1.0]
          │
          ▼
[Generate Sequences: 20-frame seeds]
          │
          ▼
[LSTM Generator Model] ◄────────────────┐
          │                             │
          ▼                             │
[Auto-regressive Output: Frame t+1]     │ (Feedback Loop)
          │                             │
          ▼                             │
[Re-Normalize Preds] ───────────────────┘
```

### Data Preprocessing & Normalization
* **Keypoint Extraction**: 13 crucial 2D coordinates are extracted from 3D motion data (representing shoulders, hips, knees, ankles, elbows, wrists, and head).
* **Centering**: The mid-point of the hips is shifted to the origin `(0, 0)`.
* **Torso Scaling**: The entire frame is scaled so the distance between the shoulder mid-point and hip mid-point (torso length) is exactly `1.0`.
* **Orientation**: The Y-axis is flipped so positive coordinates represent positions above the hips (head/arms) and negative coordinates represent positions below (legs).

---

## Project Folder and File Structure

```text
StickmanAnimation-RNN/
├── ActionConditionedLSTM/                      # Core model and logic scripts
│   ├── config.py                               # Centralized hyperparameter and path configurations
│   ├── Dataset_Preprocessor.py                 # Loads raw data, extracts 2D joints, normalizes, and sequences
│   ├── Train_Action_Model.py                   # Standard Keras training script (Teacher Forcing)
│   ├── Generate_Action_Animations.py           # Generation script for standard model
│   ├── Preview_Action_Animations.py            # Animates and plots predictions via matplotlib
│   ├── Visualize_Model_Metrics.py              # Plots training vs validation loss curves
│   └── losses.py                               # Custom weighted loss definitions
├── dataset/
│   ├── split_dataset.py                        # Preprocessing script to categorize raw HumanAct12 3D files
│   ├── HumanAct12/                             # Original raw 3D pose data
│   └── HumanAct12_Categorized/                 # Sorted 3D pose files by action label
├── iofiles/                                    # System outputs
│   ├── action_rnn_model.h5                     # Saved standard weights
│   ├── training_history.json                   # Standard model metrics JSON
│   ├── generated_action_motion.npy             # Raw coordinate outputs from generator
│   └── comparison.gif                          # Rendered matplotlib side-by-side GIFs
├── pose_extractor.py                           # Standalone legacy joint extractor
└── README.md                                   # Project documentation
```

---

## Mathematical Foundations & Loss Functions

### Custom Weighted Loss Function
The model utilizes a custom loss combining Mean Squared Error (MSE) and Mean Per Joint Position Error (MPJPE):
$$\mathcal{L} = w_1 \cdot \text{MSE} + w_2 \cdot \text{MPJPE}$$

Where:
* **MSE** measures overall coordinate squared deviations:
  $$\text{MSE} = \frac{1}{T \cdot J \cdot 2} \sum_{t=1}^{T} \sum_{j=1}^{J} \|\mathbf{p}_{t, j}^{\text{true}} - \mathbf{p}_{t, j}^{\text{pred}}\|^2$$
* **MPJPE** calculates the average Euclidean distance between predicted and ground-truth joint positions:
  $$\text{MPJPE} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \sqrt{\|\mathbf{p}_{t, j}^{\text{true}} - \mathbf{p}_{t, j}^{\text{pred}}\|^2 + \epsilon}$$
  *(With a stabilizing term $\epsilon = 10^{-8}$)*

In practice, we use $w_1 = 1.0$ and $w_2 = 2.0$ to guide the gradient flow with spatial Euclidean distances.

### Scheduled Sampling Probability Decay
During training with scheduled sampling, the probability $P_{ss}$ of passing a model prediction (instead of ground truth) to the next timestep decays linearly from `0.0` (pure Teacher Forcing) to a target threshold $T_{prob}$ over $E$ epochs:
$$P_{ss}(epoch) = \min\left(T_{prob}, \, \frac{epoch}{E} \cdot T_{prob}\right)$$
We set $T_{prob} = 0.5$ over $E=40$ epochs.

---

## Detailed Q&A (Project Intricacies)

#### Q: Why did the validation loss not converge initially?
**A**: The dataset preprocessor loaded classes sequentially (e.g. all `run` sequences, then all `walk`, then all `jump_handsup`). When we used Keras's `validation_split=0.15`, the split took the last 15% of the data—meaning the validation set was almost entirely made up of `jump_handsup` sequences. Since the training split contained virtually zero `jump_handsup` sequences, the model couldn't generalize to it, causing `val_loss` to plateau while `loss` plummeted. Shuffling the dataset arrays synchronously prior to training solved this.

#### Q: Why was there a sudden decrease in the size of the skeleton in the preview right after ground truth frames ended?
**A**: In autoregressive sequence generation, the model must predict coordinates under uncertainty. To minimize MSE, it tends to predict the "average" coordinates, drawing the outer joints closer to the center (regression to the mean). When this shrunken prediction is fed back into the LSTM as input for the next step, the shrinkage compounds recursively. We fixed this by mathematically forcing hip-centering and torso-normalization to `1.0` on every predicted frame before sending it back as feedback.

#### Q: Why isn't this scale constraint handled inside the model itself?
**A**: Standard dense output layers project arbitrary float values with no inherent concept of bones or physics. Enforcing this inside the model would require predicting joint *angles* relative to parent bones (Forward Kinematics) instead of absolute `(x, y)` coordinates, or introducing a physical differentiable constraint in the loss function.

#### Q: How is the action label handled and how does it contribute to motion generation?
**A**: Actions are converted into one-hot vectors, projected through a `Dense(16)` layer to create a dense embedding, and concatenated with the pose coordinates at every timestep. This gives the LSTM a continuous guiding vector that acts as a "control knob" indicating which motion manifold (running, jumping, boxing) it should execute.

#### Q: Why select an action embedding dimension of exactly 16?
**A**: To balance the input features. The pose coordinates represent 26 features. If the action vector was too large (e.g. 128), it would dominate the concatenated input, causing the model to ignore physical kinematics. An embedding size of 16 keeps the inputs balanced (`26 + 16 = 42`) while providing enough capacity to map the relationship between 6 discrete actions.

#### Q: Why are we using Layer Normalization instead of Batch Normalization?
**A**: Recurrent networks process sequential state dependencies. Batch Normalization normalizes across the batch dimension, which changes and fluctuates depending on batch sizes and sequence lengths. Layer Normalization normalizes across the features of a single sample's timestep, preserving temporal recurrent dynamics.

#### Q: Why is Scheduled Sampling necessary?
**A**: Standard recurrent training uses "Teacher Forcing" (always feeding in the correct ground truth frame). However, at inference/test time, the model must feed in its own predictions. This discrepancy is called **exposure bias**. Scheduled sampling slowly introduces the model's own predictions during training so it learns how to recover from its own errors.

#### Q: Why doesn't the model repeat cyclic actions (like jumping again after landing) even though pre-jump sub-sequences exist in dataset sliding windows?
**A**: Two key factors:
1. **Exposure Bias / Prediction Smoothing**: Autoregressively generated input frames are slightly smoothed by MSE loss, lacking the sharp micro-cues (like exact knee-flex angles) present in true ground-truth pre-jump frames.
2. **Dataset Majority Bias**: Most motion clips end after a single jump landing. Since MSE forces the model to predict the statistical average across all possible futures, the majority path (standing still after landing) wins over repeating the jump.

---

## Shortcomings & Room for Improvement
While the current system is highly functional, several details can be improved:
1. **Window-based Validation Split (Data Leakage)**: Currently, validation sequences are split after sliding-window sequence generation. Since sequences are generated with a stride of 1 frame, adjacent sequences overlap by 19 frames. Thus, the validation set shares almost identical frames with the training set. A better approach is splitting by *original video files* before slicing into sequences.
2. **Manual Recurrent Loop Speed**: In `Train_Action_Model_Scheduled.py`, custom autoregressive stepping is done in a raw Python loop which is slower than using optimized CUDA-backed Keras sequence layers.

---

## Future Scope
* **Forward Kinematics Integration**: Refactor the model to predict joint angles (rotations) instead of raw coordinates, locking joint lengths permanently to avoid structural distortion.
* **Transformer-based Generative Models**: Replacing the LSTM recurrent backbone with a Temporal Transformer (like Motion Transformer) to capture longer-range dependencies and eliminate sequential generation constraints.
* **Action-Conditioned Variational Autoencoder (Action-VAE)**:
  * **Injecting Motion Diversity**: Instead of mapping actions to a static, deterministic path, a VAE encodes sequences into a continuous latent space parameterized by a mean ($\mu$) and standard deviation ($\sigma$). By sampling a random latent vector $z \sim \mathcal{N}(\mu, \sigma^2)$ and combining it with the action label, the decoder can generate infinite variations of the *same* action (e.g., a high-energy run, a sluggish run, or a run starting with the left leg vs. the right leg).
  * **Tackling Autoregressive Shortcomings**: The classic bottleneck of auto-regressive LSTMs is exposure bias and error accumulation (the motion deteriorating or freezing over long rollouts). A VAE decoder can generate the entire multi-frame motion sequence at once from a single sampled latent point $z$. Because it generates the trajectory holistically rather than frame-by-frame, compounding coordinate errors are completely bypassed, guaranteeing realistic, non-decaying animations.

