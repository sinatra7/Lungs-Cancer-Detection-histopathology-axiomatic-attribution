# Lung Cancer Detection Using Axiomatic Attribution In Deep Learning

A deep learning-based system for automated lung cancer detection from CT scan images using Convolutional Neural Networks (CNN) with explainability through Integrated Gradients.

## 📋 Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

Lung cancer remains a leading cause of mortality worldwide. This project leverages deep learning techniques to classify lung CT scans into three categories:
- **Normal** - No signs of cancer
- **Benign** - Non-cancerous growth
- **Malignant** - Cancerous growth

The system integrates **Integrated Gradients** for model explainability, providing visual explanations that build trust with clinicians and enhance diagnostic transparency.

## 💡 Motivation

The increasing burden of lung cancer diagnosis on healthcare systems necessitates automated solutions to:
- Reduce diagnostic delays
- Improve detection accuracy
- Minimize human errors
- Enable early intervention
- Support radiologists and healthcare professionals

## ✨ Features

- **High Accuracy Classification**: Achieves 94% testing accuracy
- **Multi-class Detection**: Classifies CT scans into Normal, Benign, and Malignant categories
- **Explainable AI**: Integrated Gradients visualization for transparent predictions
- **Efficient Architecture**: Optimized CNN model using Keras
- **Robust Preprocessing**: Image normalization, noise reduction, and data augmentation
- **Early Stopping Mechanism**: Prevents overfitting with intelligent training termination

## 📊 Dataset

**IQ-OTHNCCD Lung Cancer Dataset** (from Kaggle)
- **Total Images**: 1,097 CT scan images
- **Normal Cases**: 416 images
- **Malignant Cases**: 561 images
- **Benign Cases**: 120 images

### Preprocessing Techniques
- Resizing to 256×256 pixels for uniformity
- Gaussian filtering for noise reduction
- Data augmentation (rotation, flipping, brightness adjustment)
- Normalization for consistent pixel intensity

### Dataset Examples

<img width="433" height="455" alt="Screenshot 2025-10-29 021206" src="https://github.com/user-attachments/assets/d0525152-a218-457c-9e42-e4597249581e" />


## 🏗️ Model Architecture

### CNN Structure
```
Input Layer (128x128 grayscale images)
    ↓
Conv2D (32 filters, 5x5) + ReLU + MaxPooling
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling
    ↓
Flatten
    ↓
Dense (256) + ReLU + BatchNormalization
    ↓
Dense (128) + ReLU + Dropout (0.3) + BatchNormalization
    ↓
Dense (3) + Softmax
```

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: Early Stopping, ReduceLROnPlateau
- **Stopping Criteria**: Validation accuracy > 90%

<img width="1201" height="556" alt="Screenshot 2025-10-29 020330" src="https://github.com/user-attachments/assets/73dbc725-4f46-472a-b10e-37bbc7436dfd" />


## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| Training Accuracy | 98% |
| Validation Accuracy | 93% |
| Testing Accuracy | 94% |

### Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 1.00 | 0.84 | 0.91 | 25 |
| Malignant | 1.00 | 0.91 | 0.95 | 110 |
| Normal | 0.86 | 1.00 | 0.92 | 85 |
| **Weighted Avg** | **0.95** | **0.94** | **0.94** | **220** |

<img width="286" height="248" alt="Screenshot 2025-10-29 022425" src="https://github.com/user-attachments/assets/07cd57ea-e133-46e1-9f38-e325c2c53f6d" />

<img width="477" height="344" alt="Screenshot 2025-10-29 022435" src="https://github.com/user-attachments/assets/2745d41e-0b71-4fd5-8637-410a4b62e992" />


### Explainability Results

The model uses **Integrated Gradients** to highlight regions in CT scans that contribute most to predictions, providing:
- Attribution heatmaps for feature importance
- Thresholded visualizations for critical regions
- Overlay images for clinical interpretation

<img width="597" height="206" alt="lungsgradiant" src="https://github.com/user-attachments/assets/b70f3273-b17e-4661-91fd-8e0868690e89" />


## 🚀 Installation

### Prerequisites
- Python 3.8+
- Anaconda (recommended)
- CUDA-capable GPU (optional, for faster training)

### Required Libraries
```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### Training the Model

```python
# Load and preprocess data
from preprocessing import load_dataset, preprocess_images

X_train, X_val, y_train, y_val = load_dataset('path/to/dataset')

# Train model
from model import build_cnn_model

model = build_cnn_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
```

### Making Predictions

```python
# Load trained model
from keras.models import load_model

model = load_model('lung_cancer_model.h5')

# Predict on new CT scan
prediction = model.predict(new_image)
class_names = ['Benign', 'Malignant', 'Normal']
predicted_class = class_names[np.argmax(prediction)]
```

### Generating Explainability Visualizations

```python
from explainability import integrated_gradients

# Generate attribution map
attributions = integrated_gradients(model, image, baseline=None)
visualize_attributions(image, attributions)
```

## 🛠️ Technologies Used

- **Development Environment**: Anaconda, Jupyter Notebook
- **Programming Language**: Python 3.8+
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Image Processing**: OpenCV
- **Data Analysis**: NumPy, Pandas
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: Integrated Gradients

## 👥 Contributors

- **Hrishab Kakoty** - Roll No. 202102022116
- **Julon Senapoti** - Roll No. 202102022094
- **Jitu Boro** - Roll No. 202102022095
- **Shrutarsi Pandit** - Roll No. 202102022098

**Supervisor**: Dr. Anup Kumar Barman, Assistant Professor, Dept. of CSE

**Institution**: Central Institute of Technology Kokrajhar, Assam, India

## 📄 License

This project is submitted as partial fulfillment for the Bachelor of Technology degree in Computer Science & Engineering at Central Institute of Technology Kokrajhar.

## 🙏 Acknowledgments

Special thanks to:
- **Dr. Anup Kumar Barman** for invaluable guidance and supervision
- **Dr. Apurbalal Senapati**, Head of Department, CSE
- Faculty and staff of the Computer Science & Engineering Department, CIT Kokrajhar
- Contributors to the IQ-OTHNCCD dataset on Kaggle

## 📚 References

1. Erhan, D., et al. "Visualizing Higher-Layer Features of a Deep Network." arXiv:0912.5323
2. Sundararajan, M., et al. "Axiomatic Attribution for Deep Networks." ICML, 2017
3. Ribeiro, M.T., et al. ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." arXiv:1602.04938
4. Chollet, F. "Deep Learning with Python." Manning Publications, 2018

---

**Project Duration**: 7th Semester, December 2024

**Contact**: For queries regarding this project, please contact the contributors through their institutional email addresses.

**Note**: This is an academic project developed as part of the B.Tech curriculum. The model is intended for research and educational purposes and should not be used as a substitute for professional medical diagnosis.
