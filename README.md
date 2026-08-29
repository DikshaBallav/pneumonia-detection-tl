# pneumonia-detection-tl

Pneumonia Detection Using Transfer Learning (VGG16)

PROJECT OVERVIEW: 

This project focuses on the automated detection of pneumonia from chest X-ray images using deep learning and transfer learning techniques. Instead of training a convolutional neural network from scratch, a pre-trained model is leveraged to extract meaningful features from medical images, improving performance while reducing training time and computational cost.

PROBLEM STATEMENT:

Pneumonia is a serious respiratory infection that can be life-threatening if not diagnosed early. Manual interpretation of chest X-rays is time-consuming and depends heavily on expert radiologists. This project aims to assist medical diagnosis by building an AI-based binary classification system to identify pneumonia cases accurately.

APPROACH:

* Utilized Transfer Learning with a pre-trained CNN model (VGG16).
* The convolutional base was used for feature extraction, while custom fully connected layers were added for classification.

* The model classifies chest X-ray images into-

Pneumonia

Normal

* Applied image preprocessing and data augmentation to improve generalization.

KEY FEATURES:
* Binary image classification: Pneumonia vs Normal

* Transfer Learning using pre-trained VGG16

* Image preprocessing and visualization

* Model training and evaluation

* Performance metrics: Accuracy, Precision, Recall
  
TECHNOLOGIES USED:

* Python

* TensorFlow / Keras

* VGG16 (Pre-trained on ImageNet)

* NumPy
  
* Matplotlib

* Scikit-learn

DATASET:

The project uses a Chest X-Ray dataset organized into training and testing directories with two classes:

* PNEUMONIA

* NORMAL

The dataset is not included in the repository due to size constraints.

PROJECT WORKFLOW:
1) Data Loading & Visualization:
* Load chest X-ray images
* Visualize sample images from each class
   
2) Data Preprocessing:
* Image resizing to 224 × 224
* Normalization
* Batch generation using Keras utilities

3) Model Architecture:
* VGG16 as base model (pre-trained weights)
* Freezing convolutional layers
* Custom fully connected layers for classification

4) Model Training:
*Binary classification setup
* Optimizer: Adam
* Loss function: Binary Crossentropy

5) Model Evaluation:
* Accuracy
* Precision
* Recall

MODEL ARCHITECTURE:

Input Image (224×224×3)
    ↓
VGG16 Base Model (Frozen)
    ↓
Flatten Layer
    ↓
Dense Layer (ReLU)
    ↓
Dense Layer (Sigmoid)
    ↓
Binary Output (Pneumonia / Normal)

RESULTS:

The model demonstrates strong performance in detecting pneumonia cases from chest X-ray images, highlighting the effectiveness of transfer learning in medical imaging applications.

CONCLUSION:

This project shows that transfer learning is a powerful and practical approach for medical image classification, especially when dataset size and computational resources are limited. The system can serve as a supportive tool for healthcare professionals in early pneumonia detection.

## 🚀 Live Demo

👉 **[Pneumonia Detection — Live Demo](https://pneumonia-detection-tl-nrg4poshj7z6aynduiealq.streamlit.app/)**


