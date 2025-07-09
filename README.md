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
    git clone [https://github.com/owaishcu/Pneumonia_Detection_InceptionV3.git](https://github.com/owaishcu/Pneumonia_Detection_InceptionV3.git)
    cd Pneumonia_Detection_InceptionV3
    ```
2.  **Set up Python Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```


## Hyper-parameter Choices and their Justification
* **Base Model:** Inception-V3 (pre-trained on ImageNet)
* **Input Image Size:** 299x299x3
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Learning Rate:** 0.001 (default value for Adam)
* **Epochs:** 10 (with Early Stopping patience of 5 on `val_loss`)
* **Batch Size:** 32
* **Dropout Rate:** 0.5
* **Regularization:** Dropout (rate=0.5). No L1/L2 regularization used.
* **Class Imbalance Handling:** Class weighting based on training set distribution.
* **Justification for Hyperparameter Choices**
 
1.	Base Model: Inception-V3 (pre-trained on ImageNet)
	
	Transfer Learning: For medical imaging tasks with relatively small datasets (like PneumoniaMNIST), training a deep CNN from scratch is difficult and highly prone to overfitting due to the large number of parameters. Using a model pre-trained on a massive, diverse dataset like ImageNet allows us to leverage features learned from millions of general images (e.g., edges, textures, patterns) that are often transferable to new domains.
	Inception-V3 Choice: Inception-V3 is a powerful and widely-used architecture known for its good balance between computational efficiency and accuracy. Its "inception modules" allow it to learn multi-scale features, which can be beneficial for various image patterns, including those in X-rays.
2.	Input Image Size: 299x299x3
	 This specific size is dictated by the pre-trained Inception-V3 model. It was trained on ImageNet with 299x299 pixel RGB images. Resizing our 28x28 grayscale images to this specific input size allows us to correctly utilize the pre-trained weights of the Inception-V3 base.
3.	Optimizer: Adam
	 Adam (Adaptive Moment Estimation) is a popular and generally recommended optimization algorithm for deep learning. It's an adaptive learning rate optimization algorithm that computes individual adaptive learning rates for different parameters. It combines the benefits of AdaGrad and RMSProp, making it efficient, robust to hyperparameter tuning, and typically converges quickly, especially on complex models and larger datasets.
4.	Loss Function: Binary Crossentropy
	 This is the standard and most appropriate loss function for binary classification problems where the output is a probability (0 to 1) via a sigmoid activation function (which we used in the final dense layer). It measures the dissimilarity between the true binary labels and the predicted probabilities.
5.	Learning Rate: 0.001 (default for Adam)
	 This is a commonly used default learning rate for the Adam optimizer and often serves as a good starting point. It represents a balance – not too large to cause oscillations or divergence, and not too small to make training excessively slow. For many tasks, the default Adam learning rate provides solid performance without extensive fine-tuning.
6.	Epochs: 10 (maximum with Early Stopping)
	 Setting a maximum number of epochs (like 10) ensures that training doesn't run indefinitely. However, the primary control mechanism here is EarlyStopping. The 10 epochs serve as an upper bound, but the model is expected to stop much earlier if val_loss doesn't improve, thus preventing overfitting and saving computational resources. (Initially I used 20 epochs but my colab GPU just couldn't handle that. :P )
7.	Batch Size: 32
	 A batch size of 32 is a very common and generally good default choice in deep learning.
	Memory Efficiency: It's small enough to fit into GPU memory for most typical setups.
	Training Stability: Larger batches provide more stable gradient estimates but can lead to poorer generalization. Smaller batches introduce more noise but can help the model escape local minima and generalize better. 32 often strikes a good balance.
	Computational Efficiency: It's often a good compromise between slow updates (large batch) and frequent, noisy updates (small batch).
8.	Dropout Rate: 0.5 (in custom head)
	 Dropout is a powerful regularization technique. A rate of 0.5 is a common starting point and a widely accepted heuristic. It means that during each training step, 50% of the neurons in that layer are randomly "dropped out" (their outputs are set to zero). This forces the remaining neurons to learn more robust features and prevents them from co-adapting too much, which significantly reduces overfitting.
9.	Class Imbalance Handling: Class Weighting
	 The PneumoniaMNIST dataset is imbalanced (more normal cases than pneumonia). Without intervention, the model would likely become biased towards the majority class, performing well on normal cases but poorly on pneumonia cases (high false negatives). Class weighting assigns a higher penalty to misclassifications of the minority class (pneumonia). This encourages the model to pay more attention to the minority class examples during training, improving its ability to correctly identify pneumonia cases.
10.	Early Stopping Patience: 5 epochs
	This hyperparameter defines how many epochs the model will continue training without an improvement in the monitored metric (val_loss) before stopping. A patience of 5 is a reasonable value. It provides enough buffer to allow for small fluctuations in val_loss without stopping too prematurely, while still preventing prolonged training that would lead to significant overfitting.
11.	Model Checkpointing:
	 This isn't strictly a "hyperparameter" for model performance, but it's a critical part of the training workflow. It ensures that the model's weights corresponding to the best performance on the validation set (lowest val_loss) are saved to disk. This means even if training runs for many more epochs and starts to overfit, or if the training process is interrupted, you can always retrieve the best version of your model.


## Evaluation Metrics Justification
For this binary classification task with a potential class imbalance (Pneumonia vs. Normal), the following metrics were chosen:
* **Accuracy:** Overall correctness. Useful as a baseline but can be misleading with imbalance.
* **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** Measures the model's ability to distinguish between classes across various thresholds. Less sensitive to class imbalance than accuracy.
* **Precision:** Of all predicted positive (Pneumonia) cases, how many were actually positive? Important to minimize false positives (e.g., healthy patient misdiagnosed with pneumonia).
* **Recall:** Of all actual positive (Pneumonia) cases, how many did the model correctly identify? Important to minimize false negatives (e.g., pneumonia patient misdiagnosed as healthy).
* **F1-Score:** The harmonic mean of Precision and Recall. Provides a single score that balances both, especially useful for imbalanced datasets.
* **Balanced Accuracy:** The average of recall obtained on each class. It's a more appropriate metric than simple accuracy for imbalanced datasets, as it accounts for the unequal number of samples in each class.

## Results and Insights
**Results**
Test Loss: 0.3168
Test Accuracy (Keras): 0.8718
Test AUC (Keras): 0.8555
Test Precision (Keras): 0.9564
Test Recall (Keras): 0.9495
F1-Score (scikit-learn): 0.9031
Balanced Accuracy (scikit-learn): 0.8436
**Insights:**
* The model achieved a good balance between identifying positive cases (high Recall) and ensuring those positive predictions were correct (high Precision), as reflected in the F1-Score of 0.9031.
* The Balanced Accuracy of 0.8436 is a more reliable indicator of overall performance than raw accuracy, given the dataset's imbalance.
* The AUC score of 0.8555 further confirms the model's reasonable discriminative power.
* Despite the small input image size (28x28) compared to ImageNet images (299x299), transfer learning with Inception-V3 proved effective.
* The combination of data augmentation and class weighting likely contributed to the model's robust performance.
