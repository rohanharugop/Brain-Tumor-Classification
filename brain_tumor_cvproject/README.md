# Brain Tumor Classification using Deep Learning

## Overview

This project implements a Convolutional Neural Network (CNN) to classify brain MRI scans into four different categories of tumors. The model achieves high accuracy in distinguishing between Glioma, Meningioma, Pituitary tumors, and normal brain scans.

## Dataset Structure

```
Dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Model Architecture

```
Input (128x128x1)
│
├── Conv2D(32, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten
├── Dense(128) + ReLU
├── Dropout(0.6)
└── Dense(4) + Softmax
```

## Key Features

- Image preprocessing and normalization
- Data visualization tools
- Model training with validation
- Confusion matrix analysis
- Error categorization
- Prediction confidence scores

## Training Configuration

- Image Size: 128x128 pixels
- Batch Size: 16
- Epochs: 10
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-entropy

## Usage

1. Prepare your dataset in the required structure
2. Run the training script:

```python
python test_v1.ipynb
```

3. The trained model will be saved as 'Trained_brain_tumer_classifier.h5'

## Model Performance

The model includes:

- Training/validation accuracy curves
- Confusion matrix visualization
- Detailed error analysis
- Per-class performance metrics

## Files Description

- `test_v1.ipynb`: Main implementation notebook
- `Trained_brain_tumer_classifier.h5`: Saved model weights
- `Dataset/`: Contains training and testing data

## License

This project is available for educational and research purposes.

## Acknowledgments

Please cite this repository if you use this code for your research.
