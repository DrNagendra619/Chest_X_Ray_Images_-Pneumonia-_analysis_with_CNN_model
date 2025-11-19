# Chest_X_Ray_Images_-Pneumonia-_analysis_with_CNN_model
Chest_X_Ray_Images_(Pneumonia)_analysis_with_CNN_model
# 🫁 Chest X-Ray Analysis: Pneumonia Detection with CNN

This repository contains a **Jupyter Notebook** pipeline that demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** for the binary classification of **Chest X-Ray Images**. The model is designed to automate the process of diagnosing **Pneumonia** (viral or bacterial) versus **Normal** cases.

The pipeline is structured around the widely used **Chest X-Ray Images (Pneumonia) Kaggle dataset**.

## 🚀 Key Features

* **Deep Learning Classification:** Implements a custom **Convolutional Neural Network (CNN)** using Keras/TensorFlow, optimized for medical image classification.
* **Image Data Handling:** Uses `ImageDataGenerator` for efficient loading, normalization, and **Data Augmentation** (rescaling, zooming, flipping) to increase dataset size and prevent overfitting.
* **Kaggle Dataset Integration:** Designed to process the standard dataset structure, which typically includes separate directories for `train/`, `val/`, and `test/` data split by condition (`NORMAL` and `PNEUMONIA`).
* **Model Training:** Includes training steps with appropriate callbacks (e.g., EarlyStopping, ModelCheckpoints) to optimize and save the best performing model weights.
* **Performance Visualization:** Generates and plots essential metrics:
    * **Loss and Accuracy Curves** (Training vs. Validation).
    * **Confusion Matrix** (Heatmap).
    * **ROC Curve** and **AUC** score.
* **Classification Report:** Provides a final, detailed summary of Precision, Recall, and F1-Score for the classification task.

---

## 🔬 Analysis Overview

| Component | Method / Tool | Purpose |
| :--- | :--- | :--- |
| **Dataset** | Chest X-Ray Images (Pneumonia) | Medical images for binary classification (Normal vs. Pneumonia). |
| **Model** | Convolutional Neural Network (CNN) | Feature extraction and classification optimized for image data. |
| **Preprocessing** | Data Augmentation | Increases the diversity of the training set to improve model generalization. |
| **Classification** | Binary Classification | Predicts the presence of Pneumonia (Positive) or Normal (Negative). |

---

## 🛠️ Prerequisites and Setup

### 📦 Data Requirement

This notebook requires access to the **Chest X-Ray Images (Pneumonia) Dataset** (e.g., from Kaggle). The file structure must be as follows:

***Note:*** *You must ensure the image data is downloaded and organized into this structure in your working environment.*

### 🖥️ Requirements

This pipeline requires a computational environment capable of handling deep learning tasks, preferably with **GPU acceleration**. You need the following Python libraries installed:

* `tensorflow` / `keras`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (sklearn)

### ⚙️ Execution

1.  **Download** the `Chest_X_Ray_Images_(Pneumonia)_analysis_with_CNN_model.ipynb` file.
2.  **Upload the image dataset** and ensure the paths in the notebook are correctly set up.
3.  **Execute** all cells sequentially.

---

## 📊 Expected Output

The notebook generates the following critical plots and metrics:

| Output | Description |
| :--- | :--- |
| **Accuracy/Loss Plot** | Line plots showing training and validation accuracy/loss over epochs. |
| **Confusion Matrix** | Heatmap visualizing the True Positives, True Negatives, False Positives, and False Negatives on the test set. |
| **Classification Report** | Console output summarizing Precision, Recall, and F1-Score for the classification task. |
| **ROC Curve / AUC** | Graphical representation of model performance, yielding the Area Under the Curve (AUC) score. |
