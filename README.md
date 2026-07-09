# LSA64- Sign Language Recognition using CNN-LSTM and MediaPipe

A deep learning system that recognizes Argentine Sign Language (LSA64) gestures from video, using MediaPipe Holistic landmarks as input to a CNN-LSTM architecture.

## Overview
This project explores how input representation and dataset quality affect sign language recognition accuracy. Instead of feeding raw video frames into the model, MediaPipe Holistic is used to extract 1,629-dimensional landmark features per frame (pose, hands, and face), which are then passed through a CNN-LSTM network for classification.

## Key Results
- **Test Accuracy:** 98.59%
- **Classes:** 64 signs (LSA64 dataset)
- **Model Size:** 847,488 parameters
- **Input:** MediaPipe Holistic landmarks (1,629 features/frame, 30 frames/sequence)
- **Inference Time:** ~123 ms per sample

A key finding: switching from raw video input to MediaPipe landmarks, combined with a larger per-class training set, improved accuracy by 36 percentage points over an earlier baseline (62.79% on WLASL Top-15) — despite LSA64 covering more than 4x as many sign classes.

## Architecture
- **Feature extraction:** MediaPipe Holistic (pose + hands + face landmarks)
- **Spatial encoding:** TimeDistributed Dense layers (256 → 128 → 64) with BatchNormalization and Dropout
- **Temporal modeling:** Stacked Bidirectional LSTM layers (128 → 64 units)
- **Classification head:** Dense(128) → Dense(64, softmax)

## Dataset
[LSA64](http://facundoq.github.io/datasets/lsa64/) — Argentine Sign Language dataset, 64 signs performed by multiple signers, standardized to 30 frames per sequence.

## Tech Stack
- TensorFlow / Keras
- MediaPipe
- OpenCV
- scikit-learn
- NumPy, Matplotlib, Seaborn

## Project Structure
├── sign-language-recognition.ipynb   # Main notebook: data loading, MediaPipe extraction, model training, evaluation
└── README.md

## How to Run
1. Clone this repository.
2. Open the notebook in Jupyter, Kaggle, or Google Colab.
3. Install dependencies:
   pip install mediapipe opencv-python tensorflow scikit-learn matplotlib seaborn
4. Run all cells to reproduce the preprocessing, training, and evaluation pipeline.

## Author
Zoya Rasool
