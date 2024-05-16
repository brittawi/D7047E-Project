# Pneumonia Detection with Chest X-Ray Images

## Overview
This project aims to detect pneumonia in chest X-ray images using convolutional neural networks (CNNs). We explore the effectiveness of transfer learning from a pre-trained ResNet model and compare it with a simple CNN architecture.

## Architecture
This project employs two models for pneumonia detection in chest X-ray images:
- `models.py`: Implementation for our own simple CNN and the Lightning Module that we use for both models.
- `simpleCNN.ipynb`: First Implementation of our own simple CNN model.
- `simpleCNN_tuning.ipynb`: Implementation of hyperparameter tuning, using a complexer CNN model that is defined in the models.py.
- `Project-1-ResNet50.ipynb`: Implementation of a model based on ResNet with transfer learning.

## Evaluation
The models are evaluated using accuracy, F1-score, sensitivity, and specificity metrics:
- **Accuracy:** Provides an overall estimate of model correctness, but may be influenced by dataset imbalance. We expect an accuracy above 80%.
- **F1-score:** A trade-off between precision and sensitivity/recall.
- **Sensitivity:** Indicates the proportion of positive cases correctly diagnosed.
- **Specificity:** Indicates the proportion of negative cases correctly diagnosed.

## Conclusion
Through this project, we aim to develop an accurate and reliable model for pneumonia detection in chest X-ray images.
