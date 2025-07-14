# Lung Cancer Detection Using Deep Learning and Axiomatic Attribution

## Overview
This repository contains the implementation of a deep learning-based system for lung cancer detection, developed as part of an 8th-semester B.Tech project in Computer Science & Engineering at the Central Institute of Technology, Kokrajhar. The system classifies lung histopathology images into three categories: **Adenocarcinoma**, **Benign**, and **Squamous Cell Carcinoma**, achieving a test accuracy of **95%** with the CNN-2 model. Axiomatic attribution using Integrated Gradients enhances interpretability by generating heatmaps to highlight critical image regions, aiding radiologists in clinical decision-making.

## Project Details
- **Authors**: Hrishab Kakoty
- **Institution**: Central Institute of Technology, Kokrajhar, Bodoland Territorial Areas Districts, Assam, India

## Features
- **Dataset**: 15,000 lung histopathology images, split into training, validation, and test sets, categorized into Adenocarcinoma (lung_aca), Benign (lung_n), and Squamous Cell Carcinoma (lung_scc).
- **Models**:
  - **CNN-1**: Lightweight convolutional neural network with three convolutional blocks for initial experimentation, achieving 99% test accuracy but showing overfitting (validation accuracy: 0.5).
  - **CNN-2**: Deeper architecture with four convolutional blocks (16, 32, 64, 128 filters), achieving **95% test accuracy** and 94% validation accuracy, with balanced performance.
  - **MobileNet V2**: Lightweight model fine-tuned for the dataset, achieving 83% test accuracy, underperforming due to dataset complexity.
- **Technologies**: TensorFlow, Keras, Integrated Gradients for interpretability.
- **Data Augmentation**: Random rotation (0.2), horizontal/vertical flipping (0.2), and zooming (0.1) applied on-the-fly during training to enhance model robustness.
- **Optimization**: Adam optimizer with a learning rate of 0.001, categorical cross-entropy loss, and techniques like EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- **Interpretability**: Integrated Gradients generates heatmaps to visualize critical image regions (e.g., cancerous nodules, cellular abnormalities), compared with Keras Saliency for localized feature highlighting.

lung-cancer-detection/
├── data/                   # Dataset directory (not included)
├── models/                 # Saved models (e.g., cnn_lung_model_train_test_val.keras)
├── heatmaps/               # Generated heatmaps
├── src/
│   ├── data_preparation.py # Dataset loading and augmentation
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Model evaluation script
│   ├── generate_heatmaps.py# Heatmap generation with Integrated Gradients
├── requirements.txt        # Dependencies
├── README.md               # This file
├── report.pdf              # Project report

