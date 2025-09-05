---
layout: notebook
title: "Introduction to Machine Learning with Scikit-learn"
date: 2024-01-20
tags: [machine-learning, scikit-learn, python, classification]
categories: [Machine Learning]
excerpt: "A beginner-friendly introduction to machine learning using Python's Scikit-learn library"
---

# Introduction to Machine Learning with Scikit-learn

This notebook provides a hands-on introduction to machine learning using Python's Scikit-learn library. We'll cover data preprocessing, model training, evaluation, and prediction.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Making Predictions](#making-predictions)
6. [Conclusion](#conclusion)

## Introduction

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. In this notebook, we'll explore:

- Supervised learning (classification and regression)
- Data preprocessing techniques
- Model evaluation metrics
- Cross-validation

## Data Preparation

Let's start by importing the necessary libraries and loading our dataset:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ml_dataset.csv')
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
```

### Data Exploration

```python
# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())
```

### Data Preprocessing

```python
# Handle missing values
df = df.fillna(df.median())

# Encode categorical variables
le = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

## Model Training

Now let's split our data and train a machine learning model:

```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
```

## Model Evaluation

Let's evaluate our model's performance:

```python
# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### Feature Importance

```python
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

## Making Predictions

Let's make predictions on new data:

```python
# Example: Make prediction on new data
new_data = pd.DataFrame({
    'feature1': [1.5],
    'feature2': [2.3],
    'feature3': [0.8]
    # Add more features as needed
})

# Preprocess new data
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)
prediction_proba = model.predict_proba(new_data_scaled)

print(f"Prediction: {prediction[0]}")
print(f"Prediction Probability: {prediction_proba[0]}")
```

## Conclusion

In this notebook, we've covered the fundamental steps of machine learning:

1. **Data Preparation**: Loading, exploring, and preprocessing data
2. **Model Training**: Training a Random Forest classifier
3. **Model Evaluation**: Assessing performance using various metrics
4. **Feature Importance**: Understanding which features contribute most to predictions
5. **Making Predictions**: Using the trained model on new data

### Key Takeaways

- Data preprocessing is crucial for model performance
- Feature scaling can significantly improve results
- Model evaluation should include multiple metrics
- Understanding feature importance helps with model interpretability

### Next Steps

- [ ] Try different algorithms (SVM, Neural Networks, etc.)
- [ ] Implement cross-validation for better model evaluation
- [ ] Explore hyperparameter tuning
- [ ] Build a complete ML pipeline

---

*This notebook provides a solid foundation for machine learning with Scikit-learn. Experiment with different datasets and algorithms to deepen your understanding!*