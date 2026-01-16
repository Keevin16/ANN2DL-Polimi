# Artificial Neural Networks & Deep Learning Projects

**Politecnico di Milano - Fall 2025**

This repository contains the source code and technical documentation for the competitive challenges of the **Artificial Neural Networks & Deep Learning (AN2DL)** course. These projects focus on applying deep learning architectures to constrained, real-world datasets involving high-dimensional time series and low-resolution medical imaging.

---

## Challenge 1: Multivariate Time-Series Classification
**Task:** Pain Level Detection from Kinematic Data

The objective was to classify subject pain status (`no_pain`, `low_pain`, `high_pain`) using kinematic time-series data. The dataset presented significant challenges regarding class imbalance and a high-dimensional setting where larger models were prone to overfitting.

### Methodology
We developed a robust pipeline prioritizing feature engineering and hybrid architectures:
* **Preprocessing:** Implemented **2D Sinusoidal Positional Encoding** to model temporal dependencies and merged categorical features into a binary "pirate" flag to handle morphological anomalies.
* **Architecture:** Designed a hybrid **1D-CNN** (for feature extraction) followed by an **LSTM** (for temporal dependencies) and a **Self-Attention** layer to recalibrate the importance of inputs across the sequence.
* **Optimization:** Addressed class imbalance by adjusting class weights to be inversely proportional to their frequencies and optimized hyperparameters using **Optuna**.
* **Validation:** Used **Stratified Group 10-Fold Cross-Validation** to ensure subject-level separation and prevent data leakage.

**🏆 Result:** 0.92 Weighted F1-Score.

📄 **[View Technical Report](./FirstChallenge/report/AN2DL___theoverfitters-1.pdf)**

---

## Challenge 2: Breast Cancer Sub-typing via Foundation Models
**Task:** Low-Resolution Whole Slide Image (WSI) Classification

This project addressed the classification of breast cancer molecular subtypes using low-resolution WSIs. The primary challenge was the inability of standard ImageNet-pretrained networks to capture histological features (like tumor architecture) from low-magnification inputs.

### Methodology
We implemented a transfer learning pipeline leveraging domain-specific foundation models:
* **Feature Extraction:** Utilized **UNI**, a foundation model pre-trained on 100M+ pathology patches. This pathology-specific model significantly outperformed general-purpose models like ResNet50 and EfficientNet.
* **Aggregation:** Employed **Minkowski Mean ($p=3$)** to aggregate patch embeddings into a single slide-level representation, which provided superior regularization on a small training set of 581 usable images.
* **Classification:** Deployed **CatBoost** on the frozen embeddings to mitigate the overfitting issues observed with deep neural network classifiers on high-dimensional feature vectors.

**🏆 Result:** 0.4469 Test F1-Score (representing a significant improvement over baseline models).

📄 **[View Technical Report](./SecondChallenge/report/challenge2_andyetitlearns.pdf)**

---

## 🛠 Tech Stack
* **Frameworks:** PyTorch, TensorFlow/Keras, Optuna
* **Models:** Hybrid CNN-LSTM, Attention Mechanisms, Vision Transformers (UNI, Virchow2, Phikon), CatBoost
* **Techniques:** Transfer Learning, 2D Sinusoidal Embeddings, Stratified Group K-Fold, Minkowski Pooling

## 📂 Repository Structure
* `FirstChallenge/`: Source code and notebooks for the kinematic time-series classification task.
* `SecondChallenge/`: Source code and notebooks for the WSI breast cancer sub-typing task.
* `skeleton/`: Baseline notebooks and exercise sessions provided by the course.

---

### Authors
* **Raul Agolli**
* **Kevin Pio Abate**
* **Andrea Ascenzo Mastroberti**
* **Simone Mauro** (Challenge 2)
