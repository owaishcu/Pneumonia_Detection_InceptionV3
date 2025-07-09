# Pneumonia_Detection_InceptionV3
Fine-tuning Inception-V3 for Pneumonia Detection on PneumoniaMNIST dataset.
# Pneumonia Detection using Inception-V3 Transfer Learning

## Project Objective
Fine-tune a pre-trained Inception-V3 model to accurately distinguish between pneumonia and normal chest X-ray images from the PneumoniaMNIST dataset. The goal is to report the model's performance using appropriate evaluation metrics.

## Dataset
**Name:** PneumoniaMNIST
**Source:** https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data 

## Approach Overview
This project leverages transfer learning with Google's Inception-V3 architecture, pre-trained on ImageNet, for image classification. Key steps included:
1.  **Data Preprocessing:** Resizing grayscale images to 299x299 pixels and converting to 3-channel RGB as required by Inception-V3. Pixel values normalized to [-1, 1].
2.  **Class Imbalance Mitigation:** Addressing the imbalanced nature of the dataset (more normal cases than pneumonia) using class weighting during training.
3.  **Overfitting Prevention:** Implementing data augmentation techniques (random rotations, shifts, zooms, flips, brightness adjustments) and Dropout layers in the custom classification head.
4.  **Model Training:** Fine-tuning the Inception-V3 base with a custom dense classification head. Training was managed with Early Stopping and Model Checkpointing to save the best performing model based on validation loss.

## How to Reproduce
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com//Pneumonia_Detection_InceptionV3.git](https://github.com/owaishcu/Pneumonia_Detection_InceptionV3.git)
    cd Pneumonia_Detection_InceptionV3
    ```
2.  **Set up Python Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download Dataset:** 

## Hyper-parameter Choices
* **Base Model:** Inception-V3 (pre-trained on ImageNet)
* **Input Image Size:** 299x299x3
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Learning Rate:** (You used default Adam, or a schedule if you added it - specify if you tweaked it)
* **Epochs:** 10 (with Early Stopping patience of 5 on `val_loss`)
* **Batch Size:** 32
* **Dropout Rate:** 0.5 (in custom head)
* **Regularization:** (e.g., L1/L2 if you added it to dense layers, or mention 'None' if not used beyond Dropout)
* **Class Imbalance Handling:** Class weighting based on training set distribution.

## Evaluation Metrics Justification
For this binary classification task with a potential class imbalance (Pneumonia vs. Normal), the following metrics were chosen:
* **Accuracy:** Overall correctness. Useful as a baseline but can be misleading with imbalance.
* **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** Measures the model's ability to distinguish between classes across various thresholds. Less sensitive to class imbalance than accuracy.
* **Precision:** Of all predicted positive (Pneumonia) cases, how many were actually positive? Important to minimize false positives (e.g., healthy patient misdiagnosed with pneumonia).
* **Recall:** Of all actual positive (Pneumonia) cases, how many did the model correctly identify? Important to minimize false negatives (e.g., pneumonia patient misdiagnosed as healthy).
* **F1-Score:** The harmonic mean of Precision and Recall. Provides a single score that balances both, especially useful for imbalanced datasets.
* **Balanced Accuracy:** The average of recall obtained on each class. It's a more appropriate metric than simple accuracy for imbalanced datasets, as it accounts for the unequal number of samples in each class.

## Results and Insights




## Results and Insights

Test Loss: 0.3168
Test Accuracy (Keras): 0.8718
Test AUC (Keras): 0.8555
Test Precision (Keras): 0.9564
Test Recall (Keras): 0.9495
F1-Score (scikit-learn): 0.9031
Balanced Accuracy (scikit-learn): 0.8436
