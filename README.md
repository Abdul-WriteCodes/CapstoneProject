

# **Sentiment Analyis**
This project builds a sentiment classification model using IMDB movie reviews, cleaned and vectorized into a machine learning pipeline. 
The model was trained using Stochastic Gradient Descent (SGD) with log_loss and Logistic Regression evaluated on test data.
<p align="center">
	<img src="assets/Image.png" alt="Image" style="width:100%; max-width:800px; "/>
</p>
---
Model Overview
Model Used: 
        SGDClassifier (loss="log_loss", with TfidfVectorizer)
		Logistic Regression with TfidfVectorizer
Text Processing: Custom text_cleaning function (from text_utils.py)
Vectorization: TF-IDF (top 5000 features)
Training/Test Split: 80/20

---
Word Cloud Visualization of the IMDb Text reviews
<p align="center">
	<img src="assets/word_cloud.png" alt="word_cloud" style="width:100%; max-width:800px; "/>
</p>

---
Evaluation Result
| Metric                      | SGDClassifier        | LogisticRegression    |
|-----------------------------|----------------------|------------------------|
| Cross-Validation Accuracy   | 87.30%               | 88.15%                 |
| Test Accuracy               | 86.25%               | 87.80%                 |

---
Classification Report
<p align="center">
	<img src="assets/Classification Report.png" alt="Classification Report" style="width:100%; max-width:800px; "/>
</p>
---
Confusion Matrix
<p align="center">
	<img src="assets/Confusion_Matrix.png" alt="Confusion_Matrix" style="width:100%; max-width:800px; "/>
</p>
---
ROC Curve

<p align="center">
	<img src="assets/ROC_Curve.png" alt="ROC_Curve" style="width:100%; max-width:800px; "/>
</p>
---
Precision Recall Curve
<p align="center">
	<img src="assets/Precision_Recall.png" alt="Precision_Recall" style="width:100%; max-width:800px; "/>
</p>

Quick Start