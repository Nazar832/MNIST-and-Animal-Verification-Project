MNIST Classification & Animal Verification

This repository contains solutions for two main tasks: an OOP-based MNIST classification system and a multi-modal animal verification pipeline.

## 📂 Project Structure
- **Task_1/**: MNIST classification using various algorithms (RF, NN, CNN).
  - `classes.py`: OOP implementation of model interfaces and logic.
  - `train.py`: Script to train and save MNIST models.
- **Task_2/**: Animal verification pipeline (NLP + CV).
  - **CV_model/**: Image classification logic (MobileNetV2).
  - **NER_model/**: Named Entity Recognition logic.
  - `pipeline.py`: Main logic that integrates both models - extracting animals from text and checking if such animals are present in given images.
  - `class_names.json`: Mapping for 15 animal classes.
- **Demo.ipynb**: Comprehensive Jupyter Notebook models evaluation, and live demonstrations.

## 🛠️ Setup & Requirements
- **Environment:** Python 3.10 - 3.13 (64-bit is strictly required for TensorFlow).
- **Global Installation:**
  ```bash
  pip install -r requirements.txt

## 🚀 Quick Start
- To see the project in action, open Demo.ipynb. It contains:
  - Task 1 Evaluation: Performance metrics for RF, NN, and CNN.
  - Task 2 Pipeline: Real-time verification of text-image pairs.

## Dataset
The dataset used for this project can be downloaded from https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset/data. Please place it in the Task_2/CV_model/data/ directory before training.
