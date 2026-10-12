# Endoscopic Lesion Detection & Risk Grading System
**Review 1 Presentation Content & Outline (Detailed Technical Version)**

*This document contains a highly detailed, step-by-step breakdown of your entire project pipeline. Use this to create technically rich slides for your Review 1 PPT.*

---

## 1. Introduction & Problem Statement
*   **Topic:** Automated Computer-Aided Detection (CAD) for segmenting gastrointestinal polyps using Explainable AI (XAI).
*   **The Challenge:** Doctors face high miss-rates (up to 25%) during colonoscopies due to fatigue and the subtle visual differences between polyps and healthy tissue. 
*   **The "Black Box" Problem:** Traditional Deep Learning models just output a mask without explaining *how* they found the polyp. If doctors can't understand the AI's reasoning, they won't trust it.
*   **Our Objective:** Build a lightweight, highly accurate polyp segmentation system (Attention U-Net) that operates within 4GB VRAM constraints and visually explains its decisions (Grad-CAM).

## 2. Literature Review & Novelty
*   **Base Paper:** *"Attention-Guided U-Net With Grad-CAM for Explainable Polyp Segmentation in Colonoscopy Images"* (IEEE Access, 2025).
*   **What Others Did:** Standard U-Nets pass all features from the encoder to the decoder equally. This means background noise (reflections, healthy walls, medical instruments) confuses the model.
*   **Our Contribution & Novelty:**
    1.  **Constraint-Aware Engineering:** Optimized for 4GB VRAM by standardizing image sizes to $256 \times 256$ while preserving sharp boundary contours.
    2.  **Joint Loss Optimization:** Polyps are often tiny compared to the whole image. We used a combined **BCE + Dice Loss** function to force the AI to care about boundary overlap, not just raw pixel counts.
    3.  **Explainability Pipeline:** Directly hooked Grad-CAM into the model's bottleneck to provide a post-hoc visual indication of regions contributing to the segmentation output.

---

## 3. Methodology Phase 1: Data Processing & Transformation
*This is exactly how we handled the images before they touched the AI.*

*   **Dataset Engine (`src/data/dataset.py`):**
    *   Sourced the **Kvasir-SEG dataset**, containing 1,000 paired files (RGB Endoscopy Images + Grayscale Ground Truth Masks drawn by doctors).
    *   Built a custom PyTorch `Dataset` class that safely pairs every image to its exact mask.

*   **Joint Preprocessing (`src/preprocessing/transform.py`):**
    *   **Resizing Rule:** We resized images to $256 \times 256$ using **Bilinear** interpolation (for smooth colors), but resized the Masks using **Nearest Neighbor** interpolation to prevent the binary labels (0 or 1) from blurring into decimals.
    *   **Spatial Augmentation:** We randomly applied Horizontal/Vertical flips and 90°/180°/270° rotations. *Crucially*, these spatial changes were applied simultaneously to *both* the image and the mask so they stayed perfectly aligned.
    *   **Color Augmentation:** We applied brightness, contrast, and saturation jitter *only* to the image (to simulate different endoscopy cameras) without breaking the mask.
    *   **Normalization:** Normalized pixel values using standard ImageNet distribution vectors (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

---

## 4. Methodology Phase 2: The Attention U-Net Architecture
*This is the exact math and structure of our AI model (`src/models/attention_unet.py`).*

*   **The Encoder (Feature Extraction):**
    *   Uses 4 stages of `DoubleConv` blocks (Convolution $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\times$ 2) followed by `MaxPool2d` to extract features (shapes, textures) and compress the image size.
*   **The Bottleneck:** 
    *   The deepest part of the network containing the richest, most complex abstract features of the image, but at the lowest resolution.
*   **Attention Gates (The "Filter"):**
    *   Instead of blindly passing skip-connections $x$ from the Encoder to the Decoder, we pass them through an Attention Gate.
    *   The gate uses a gating signal $g$ (coarse, high-level context from the layer below) to determine *where* to look.
    *   It projects both $g$ and $x$ through $1 \times 1$ convolutions ($W_g$, $W_x$), adds them, applies a ReLU, and then a Sigmoid function $\psi$ to generate **Attention Weights** ($\alpha$) between $0$ and $1$.
    *   It multiplies the skip-connection $x$ by $\alpha$, effectively muting background pixels and highlighting polyp pixels.
*   **The Decoder (Mask Reconstruction):**
    *   Upsamples the feature maps and concatenates them with the *Attended* skip-connections to reconstruct the image back to $256 \times 256$, outputting a binary mask map (Polyp = 1, Background = 0).

---

## 5. Methodology Phase 3: Training & Optimization
*How we taught the model (`src/training/trainer.py`).*

*   **Loss Function Strategy:**
    *   We defined $\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{BCE}} + 1.0 \times \mathcal{L}_{\text{Dice}}$.
    *   **BCE (Binary Cross Entropy):** Punishes the model if it incorrectly classifies an individual pixel.
    *   **Dice Loss:** Looks at the macro picture. It calculates the intersection (overlap) between the prediction and the ground truth, divided by the total area. This prevents the model from just guessing "background" everywhere to get a high score.
*   **Training Loop:**
    *   Used the **Adam Optimizer**.
    *   Applied a threshold of $0.5$ (via Sigmoid) to the model's raw logits to convert them into hard binary predictions.
    *   Saved the best model weights (`review_1_best.pth`) based purely on Validation Dice scores, ensuring the model generalizes to unseen data.

---

## 6. Evaluation, Results & Explainability (Grad-CAM)
*Evaluating our model's visual interpretability signals (`src/explainability/grad_cam.py`).*

*   **Quantitative Results:**
    *   Calculated strict evaluation metrics on an isolated test set.
    *   *You will paste the exact numbers for **Dice Coefficient**, **IoU (Intersection over Union)**, **Precision**, and **Recall** from your `metrics.json` file here.*
    *   **Robustness Metrics:** We report the **Median** and **Interquartile Range (IQR)** (e.g., Median Dice ~0.91) to show typical performance, as the mean is skewed by a bimodal distribution.
    *   **Failure Analysis:** We explicitly track complete misses (Dice < 0.1, roughly 4% of images) which motivates our upcoming Confidence Score feature.
*   **Grad-CAM (Gradient-weighted Class Activation Mapping):**
    *   We registered hooks directly into the `target_layer` (the bottleneck) during the backward pass.
    *   By calculating the gradient of the segmentation output score with respect to the bottleneck's feature maps, we applied **Global Average Pooling** to find out which neurons contributed the most to the "Polyp" prediction.
    *   We applied a ReLU to the gradients (to only show features that *positively* influenced the decision), and upsampled this to a $256 \times 256$ heatmap.
*   **Qualitative Results:**
    *   Generated exactly 5-column verification grids showing: 
        1. Original Input Image
        2. Ground Truth Doctor's Mask
        3. Attention U-Net Prediction
        4. Grad-CAM Heatmap (showing the AI focusing exactly on the polyp region and ignoring the colon walls).
        5. The Heatmap overlaid on the original image for doctors to verify.

---

## 7. Next Phase (Project Roadmap)
*Our system is built across three planned stages.*

*   **Review 1 (Foundation):** "We built the segmentation foundation and demonstrated quantitative performance and explainability."
*   **Review 2 (Core Intelligence):** "We extend the system with enhanced preprocessing, polyp classification, classifier-based explainability, and prediction confidence."
    *   **Advanced Preprocessing:** CLAHE, Laplacian blur, color normalization.
    *   **Classification:** ResNet50/EfficientNet-B3 classifier (Normal vs. Polyp).
    *   **Classifier Grad-CAM:** Explain classification decisions.
    *   **Confidence Score:** Statistically grounded prediction reliability.
*   **Final Review (Integration):** "We integrate the complete pipeline into patient-level reporting and the final interactive application."
