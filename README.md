# Pneumonia_Detection_InceptionV3
Fine-tuning Inception-V3 for Pneumonia Detection on PneumoniaMNIST dataset.
# Pneumonia Detection using Inception-V3 Transfer Learning

## Project Objective
Fine-tune a pre-trained Inception-V3 model to accurately distinguish between pneumonia and normal chest X-ray images from the PneumoniaMNIST dataset. The goal is to report the model's performance using appropriate evaluation metrics.

## Dataset
**Name:** PneumoniaMNIST
**Source:** Available via MedMNIST (or similar sources). The dataset consists of grayscale images of size 28x28 pixels.

## Approach Overview
This project leverages transfer learning with Google's Inception-V3 architecture, pre-trained on ImageNet, for image classification. Key steps included:
1.  **Data Preprocessing:** Resizing grayscale images to 299x299 pixels and converting to 3-channel RGB as required by Inception-V3. Pixel values normalized to [-1, 1].
2.  **Class Imbalance Mitigation:** Addressing the imbalanced nature of the dataset (more normal cases than pneumonia) using class weighting during training.
3.  **Overfitting Prevention:** Implementing data augmentation techniques (random rotations, shifts, zooms, flips, brightness adjustments) and Dropout layers in the custom classification head.
4.  **Model Training:** Fine-tuning the Inception-V3 base with a custom dense classification head. Training was managed with Early Stopping and Model Checkpointing to save the best performing model based on validation loss.

## How to Reproduce
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  **Set up Python Environment:** (Recommended: create a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download Dataset:** Ensure `pneumoniamnist.npz` is available in your working directory. (You might need to manually download this from MedMNIST or another source if `kagglehub` is not used).
4.  **Run the Colab Notebook:** Open `your_notebook_name.ipynb` in Google Colab.
    * Mount your Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) to save model checkpoints.
    * Run all cells sequentially. The notebook contains steps for data loading, preprocessing, model definition, training, and evaluation.

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
You're very welcome! Now that the core modeling work is done, let's get you set up with the other deliverables.

Part 1: Setting up your GitHub Repository
This is where you'll store your code and provide documentation for others (or your instructors) to understand and reproduce your work.

1. Create a New GitHub Repository:

Go to GitHub and log in.

Click the "New" button (usually green) on the left sidebar, or the "+" sign in the top right corner and select "New repository."

Repository Name: Choose something descriptive, e.g., Pneumonia_Detection_InceptionV3, PneumoniaMNIST_TransferLearning, etc.

Description: Add a short description (e.g., "Fine-tuning Inception-V3 for Pneumonia Detection on PneumoniaMNIST dataset.").

Public/Private: Choose Public if it's for an assignment or project showcase.

Add a README file: Check this box. GitHub will automatically create a basic README.md for you.

Click "Create repository."

2. Upload your Colab Notebook (.ipynb file):

In your Colab notebook, go to File > Download > Download .ipynb.

On your new GitHub repository page, click the "Add file" dropdown and select "Upload files."

Drag and drop your downloaded .ipynb file into the upload area.

Add a commit message (e.g., "Add initial notebook with model training and evaluation").

Click "Commit changes."

3. Create requirements.txt:
This file lists all the Python libraries your code depends on, making it easy for others to set up their environment.

In your Colab notebook, run the following code in a new cell:

Python

!pip freeze > requirements.txt
This will create a file named requirements.txt in your Colab environment.

To download it, run:

Python

from google.colab import files
files.download('requirements.txt')
Go back to your GitHub repository. Click "Add file" > "Create new file."

Name the file requirements.txt.

Open the downloaded requirements.txt file on your computer with a text editor, copy its entire content, and paste it into the new file on GitHub.

Commit the new file.

4. Update your README.md:
This file is crucial for anyone looking at your repository. You should make it informative based on the assignment requirements.

Here's a template for your README.md:

Markdown

# Pneumonia Detection using Inception-V3 Transfer Learning

## Project Objective
Fine-tune a pre-trained Inception-V3 model to accurately distinguish between pneumonia and normal chest X-ray images from the PneumoniaMNIST dataset. The goal is to report the model's performance using appropriate evaluation metrics.

## Dataset
**Name:** PneumoniaMNIST
**Source:** Available via MedMNIST (or similar sources). The dataset consists of grayscale images of size 28x28 pixels.

## Approach Overview
This project leverages transfer learning with Google's Inception-V3 architecture, pre-trained on ImageNet, for image classification. Key steps included:
1.  **Data Preprocessing:** Resizing grayscale images to 299x299 pixels and converting to 3-channel RGB as required by Inception-V3. Pixel values normalized to [-1, 1].
2.  **Class Imbalance Mitigation:** Addressing the imbalanced nature of the dataset (more normal cases than pneumonia) using class weighting during training.
3.  **Overfitting Prevention:** Implementing data augmentation techniques (random rotations, shifts, zooms, flips, brightness adjustments) and Dropout layers in the custom classification head.
4.  **Model Training:** Fine-tuning the Inception-V3 base with a custom dense classification head. Training was managed with Early Stopping and Model Checkpointing to save the best performing model based on validation loss.

## How to Reproduce
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  **Set up Python Environment:** (Recommended: create a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download Dataset:** Ensure `pneumoniamnist.npz` is available in your working directory. (You might need to manually download this from MedMNIST or another source if `kagglehub` is not used).
4.  **Run the Colab Notebook:** Open `your_notebook_name.ipynb` in Google Colab.
    * Mount your Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) to save model checkpoints.
    * Run all cells sequentially. The notebook contains steps for data loading, preprocessing, model definition, training, and evaluation.

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
(Copy and paste your final evaluation results here)
--- Model Evaluation Results on Test Set ---
Test Loss: 0.3168
Test Accuracy (Keras): 0.8718
Test AUC (Keras): 0.8555
Test Precision (Keras): 0.9564
Test Recall (Keras): 0.9495
F1-Score (scikit-learn): 0.9031
Balanced Accuracy (scikit-learn): 0.8436
