import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset with 'latin-1' encoding to fix UnicodeDecodeError
data = pd.read_csv("./datasets/spam.csv", encoding='latin-1')

# Display columns to verify structure
print("Dataset columns:", data.columns)

# 2. Select text and label columns based on the lab manual format ('labels' and 'message')
if 'labels' in data.columns and 'message' in data.columns:
    x_data = data['message']
    y_data = data['labels']
elif 'v1' in data.columns and 'v2' in data.columns:  # Alternative standard format
    x_data = data['v2']
    y_data = data['v1']
else:
    # Fallback to positional columns if names vary
    x_data = data.iloc[:, 1]
    y_data = data.iloc[:, 0]

# 3. Feature Extraction using CountVectorizer (Tokenization & Vectorization)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x_data)
y = y_data

# 4. Train-Test Split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. K-NN Classifier Implementation (matching the lab manual objective)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# 6. Prediction and Performance Evaluation
y_pred = knn.predict(x_test)

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
