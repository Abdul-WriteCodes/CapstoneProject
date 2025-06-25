# 📘 Sentiment Analysis

This project builds a sentiment classification model using IMDB movie reviews, cleaned and vectorized into a machine learning pipeline.  
The model was trained using **Stochastic Gradient Descent (SGD)** with `log_loss` and **Logistic Regression**, and evaluated on test data.

<p align="center">
	<img src="assets/Image.png" alt="Project Overview" style="width:100%; max-width:800px;" />
</p>

---

## 🧠 Model Overview

- **Models Used:**
  - `SGDClassifier` with `loss="log_loss"` and `TfidfVectorizer`
  - `LogisticRegression` with `TfidfVectorizer`
- **Text Processing:** Custom `text_cleaning` function (from `text_utils.py`)
- **Vectorization:** TF-IDF with top 5,000 features
- **Train/Test Split:** 80/20

---

## ☁️ Word Cloud Visualization of IMDb Text Reviews

The word cloud illustrate the recurring words in the text data which was analyzed

<p align="center">
	<img src="assets/word_cloud.png" alt="word_cloud" style="width:100%; max-width:800px;" />
</p>

---

## 📊 Evaluation Result

The high accuracy scores from both 5-fold cross-validation and the test set indicate that the model not only learned the underlying patterns in the training data effectively, but also generalizes well to unseen data.

| Metric                    | SGDClassifier | LogisticRegression |
|---------------------------|---------------|--------------------|
| Cross-Validation Accuracy | 87.30%        | 88.15%             |
| Test Accuracy             | 86.25%        | 87.80%             |

---

## 📋 Classification Report
The classification report presents key evaluation metrics — precision, recall, F1-score, and accuracy — that measure the model’s performance in predicting positive and negative sentiment labels from the dataset

<p align="center">
	<img src="assets/Classification Report.png" alt="Classification Report" style="width:100%; max-width:800px;" />
</p>

---

## 🔢 Confusion Matrix
The confusion matrix displays the count of true positives, true negatives, false positives, and false negatives, giving insight into how well the model is classifying each class
<p align="center">
	<img src="assets/Confusion_Matrix.png" alt="Confusion_Matrix" style="width:100%; max-width:800px;" />
</p>

---

## 📈 ROC Curve
AUC values closer to 1.0 suggest the model can correctly rank positive instances higher than negative ones across a wide range of thresholds, reflecting robust classification performance
<p align="center">
	<img src="assets/ROC_Curve.png" alt="ROC_Curve" style="width:100%; max-width:800px;" />
</p>

---

## 🎯 Precision-Recall Curve

The high AUC score indicates that the model achieves a strong trade-off between precision and recall, demonstrating effective discrimination between positive and negative reviews
<p align="center">
	<img src="assets/Precision_Recall.png" alt="Precision_Recall" style="width:100%; max-width:800px;" />
</p>
