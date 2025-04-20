# 📧 Email Classification: Legitimate vs. Spam

This project presents a **comparative analysis** of various machine learning and deep learning techniques for classifying emails as either **legitimate** or **spam**. Three main feature extraction approaches were explored: **Bag of Words (BoW)**, **TF-IDF**, and **Word Embedding**, each paired with several classification models.

---

## 🚀 Methods and Models

### 1. **Bag of Words (BoW)**
Using BoW vectorization, we trained two traditional classifiers:

- **Random Forest (RF)**  
  - **Accuracy:** 93%  
  - **Recall:** 92%  
  - **Precision:** 91%

- **XGBoost Classifier**  
  - **Accuracy:** 98%  
  - **Recall:** 98%  
  - **Precision:** 97%

---

### 2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
Using TF-IDF vectorization, we evaluated:

- **Random Forest (RF)**  
  - **Accuracy:** 95%  
  - **Recall:** 95%  
  - **Precision:** 95%

- **XGBoost Classifier**  
  - **Accuracy:** 95%  
  - **Recall:** 95%  
  - **Precision:** 95%

---

### 3. **Word Embedding**
We trained deep learning models using Keras with word embeddings:

- **LSTM (Long Short-Term Memory)**  
  - **Accuracy:** 71%  
  - **Recall:** 67%  
  - **Precision:** 67%

- **Bi-Directional LSTM (Bi-LSTM)**  
  - **Accuracy:** 71%  
  - **Recall:** 67%  
  - **Precision:** 67%

---

## 📊 Comparative Analysis

From our results:
- **XGBoost** with **BoW** performed the best in terms of all metrics.
- **TF-IDF** maintained a strong balance across both RF and XGBoost.
- **Deep learning models** (LSTM, Bi-LSTM) underperformed in comparison, likely due to dataset size or model optimization constraints.

---

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras for deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
