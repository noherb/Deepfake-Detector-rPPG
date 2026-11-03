Inspired by FakeCatcher (built in 2020 by Ciftci, Demir and Yin from Binghamton University and Intel Labs)

# rPPG-Based Deepfake Detection Pipeline

An experimental computer vision and signal processing pipeline designed to detect video deepfakes by analyzing physiological pulse incoherence (Remote Photoplethysmography - rPPG) across facial regions.

## Overview

This project implements a lightweight variant of the FakeCatcher approach. Instead of relying purely on spatial deep learning artifacts, it evaluates temporal pulse consistency. 

The pipeline extracts raw RGB signals from four facial regions of interest (forehead, cheeks, chin), applies the Plane-Orthogonal-to-Skin (POS) algorithm, isolates the cardiac frequency band using a Butterworth filter, and computes spectral coherence features to train a robust classifier.

## Technical Stack

- **Computer Vision:** OpenCV, Google MediaPipe (FaceLandmarker)
- **Signal Processing:** SciPy (POS algorithm, Butterworth bandpass filter, magnitude-squared coherence)
- **Machine Learning:** scikit-learn (SVM / Random Forest, StandardScaler, GroupKFold validation)
- **Dataset:** FaceForensics++ (c23 compressed subset)

## Pipeline Architecture

1. **ROI Extraction:** MediaPipe tracks facial landmarks to isolate consistent skin patches across frames.
2. **Signal Extraction (POS):** Converts RGB streams into a robust blood volume pulse (BVP) signal independent of skin tone.
3. **Bandpass Filtering:** Applies a 4th-order Butterworth filter (0.7 - 3.0 Hz) to isolate cardiac activity (42-180 BPM).
4. **Windowed Coherence Analysis:** Computes magnitude-squared coherence across region pairs using sliding windows to capture temporal stability.
5. **Rigorous Validation:** Employs `GroupKFold` cross-validation grouped by source video ID to prevent identity leakage between training and testing sets.

## Evaluation & Results

### 1. rPPG Signal Validation (UBFC-RPPG Dataset 1)
To validate the physiological extraction module independently, the system was tested on UBFC-RPPG Dataset 1 against ground-truth heart rate data:
- **Performance:** Achieved a Mean Absolute Error (MAE) of **0.9 BPM**, confirming that the POS algorithm and Butterworth filtering accurately capture real cardiac pulses under favorable conditions.

### 2. Deepfake Classification & Engineering Challenges (FaceForensics++ c23)
Scaling the pipeline to compressed deepfake datasets highlights key technical hurdles to address in future iterations:
- **Technical Bottleneck:** Heavy lossy video compression (c23 format) significantly attenuates the subtle subsurface micro-color variations required for reliable rPPG estimation on standard consumer-grade video. 
- **Active Roadmap:** Overcoming compression-induced noise remains a primary engineering goal. Future work focuses on implementing adaptive noise suppression, exploring multi-band frequency analysis, and coupling rPPG coherence with spatial artifact features while preserving strict leak-free validation.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/noherb/Deepfake-Detector-rPPG.git
   cd rppg-deepfake-detector
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the trailing pipeline:
    ```bash
   python main.py
   
---

UBFC-RPPG:\
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.