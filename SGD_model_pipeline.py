# train_pipelines.py
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:47:39 2025
@author: HP
"""



import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


from wordcloud import WordCloud
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix



# Import top-level text cleaning functions
from text_utils import batch_text_cleaning, text_cleaning

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\IMDB%20Dataset.csv")

# Clean the text once for dataset quality
df["review"] = df["review"].apply(text_cleaning)

# ----Exploratory Data Analysis----
# Join all the cleaned reviews into one string
all_text = " ".join(df["review"])

# Create the word cloud
wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      colormap='viridis', max_words=200).generate(all_text)

# Plot the Word Cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Cleaned Reviews")



# Encode sentiment labels
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split features and labels
X = df["review"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use a named, importable function for FunctionTransformer
text_cleaner = FunctionTransformer(batch_text_cleaning, validate=False)

# ---- SGDClassifier pipeline ----
sgd_pipeline = Pipeline([
    ("cleaner", text_cleaner),
    ("vectorizer", TfidfVectorizer(max_features=5000)),
    ("classifier", SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, random_state=42))
])
sgd_pipeline.fit(X_train, y_train)

#Model Evaluation
sgd_y_pred = sgd_pipeline.predict(X_test)
#Test performance
sgd_perf=cross_val_score(sgd_pipeline,X_train,
                           y_train, cv=5, scoring="accuracy")
print(f"cross validation accuracy: {sgd_perf}")

sgd_test_perf=accuracy_score(y_test, sgd_y_pred)
print(f"Test accuracy for stochastic gradient boosting: {sgd_test_perf}")
print(classification_report(y_test, sgd_y_pred))

joblib.dump(sgd_pipeline, "sgd_pipeline.pkl")

#--- Visualization of Results----
#Classification Report
report_dict=classification_report(y_test, sgd_y_pred, output_dict=True)
report_df= pd.DataFrame(report_dict).transpose()
report_df= report_df.iloc[:-3, :]
plt.figure(figsize=(8,4))
sns.heatmap(report_df.iloc[:, :3], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report")
plt.ylabel("classes")
plt.xlabel("metrics")
plt.tight_layout()
plt.show()



#ROC Curve
# Get predicted probabilities for positive class 
y_proba = sgd_pipeline.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (positive)

# Binarize the true labels (convert to 0 and 1)
y_true = y_test

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', lw=2, 
         label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', 
         lw=2, linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-Stochastic Gradient Boosting Model')
plt.legend(loc="lower right")
plt.grid(True)

#Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve-SGD model')
plt.grid(True)

#Confusion matrix
cm = confusion_matrix(y_test, sgd_y_pred, 
                      labels=[0, 1])
# Step 3: Plot using seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["negative", "positive"], 
            yticklabels=["negative", "positive"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix-sgd_model')

plt.show()


print("✅ Pipeline trained and saved: sgd_pipeline.pkl")


report_dict=classification_report(y_test, sgd_y_pred, output_dict=True)
report_df= pd.DataFrame(report_dict).transpose()
report_df= report_df.iloc[:-3, :]
plt.figure(figsize=(8,4))
sns.heatmap(report_df.iloc[:, :3], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report")
plt.ylabel("classes")
plt.xlabel("metrics")
plt.tight_layout()
plt.show()