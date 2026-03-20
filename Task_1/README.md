```markdown
# Task 1: MNIST Multi-Algorithm Classification

## Overview
This task demonstrates the application of Object-Oriented Programming (OOP) in Machine Learning. It provides a unified interface to work with different classification algorithms.

## Implementation Details
- **Architecture:** Uses an Interface-based approach (`MnistClassifierInterface`).
- **Algorithms:**
  - **Random Forest**: Classical ensemble method (Scikit-learn).
  - **Neural Network**: 2-layer MLP (TensorFlow).
  - **CNN**: Convolutional layers for spatial feature extraction.

## How to Run
1. **Training:**
   ```bash
   python Task_1/train.py
    This will train all three models and save them into Task_1/models/.
2. Interface Usage:
    from Task_1.classes import MnistClassifier
    clf = MnistClassifier(algorithm='cnn')
    clf.load("Task_1/models/mnist_cnn.h5")