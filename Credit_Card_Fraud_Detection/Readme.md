# Credit Card Fraud Detection

This notebook demonstrates the implementation of a credit card fraud detection system using various machine learning techniques. The goal is to identify fraudulent transactions based on the dataset provided. 

## Table of Contents
1. [Introduction](#Introduction)
2. [Dataset Description](#Dataset-Description)
3. [Data Preprocessing](#Data-Preprocessing)
4. [Modeling](#Modeling)
   - [Logistic Regression](#Logistic-Regression)
   - [Random Forest](#Random-Forest)
   - [Artificial Neural Network](#Artificial-Neural-Network)
5. [Model Evaluation](#Model-Evaluation)
6. [Clustering Analysis](#Clustering-Analysis)
7. [Conclusion](#Conclusion)

## Introduction
Credit card fraud detection is a critical task in the finance industry. Detecting fraudulent transactions can save significant amounts of money and protect users from financial losses. This notebook explores different machine learning models to classify transactions as fraudulent or non-fraudulent.

## Dataset Description
The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

## Data Preprocessing
- Importing necessary libraries
- Loading the dataset
- Handling missing values
- Scaling features
- Splitting the data into training and testing sets

## Modeling

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators=100)
model_RF.fit(X_train, y_train)
```

### Artificial Neural Network
```python
from keras.models import Sequential
from keras.layers import Dense

model_ANN = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_ANN.summary()
```

## Model Evaluation
After training the models, they are evaluated using various metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

```python
from sklearn.metrics import classification_report, confusion_matrix

# Example for Logistic Regression
y_pred_LR = model_LR.predict(X_test)
print(classification_report(y_test, y_pred_LR))
```

## Clustering Analysis
KMeans clustering is applied to analyze the distribution of fraudulent and non-fraudulent transactions.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0]['V1'], X[y_kmeans == 0]['V2'], s=100, c='red', label='Non-Fraud')
plt.scatter(X[y_kmeans == 1]['V1'], X[y_kmeans == 1]['V2'], s=100, c='blue', label='Fraud')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.legend()
plt.show()
```

## Conclusion
The notebook demonstrates the implementation and evaluation of different machine learning models for credit card fraud detection. It also includes clustering analysis to understand the distribution of transactions. The models and techniques used provide a foundation for building more robust fraud detection systems.

Feel free to explore the code, modify parameters, and experiment with different models and preprocessing techniques to improve the detection accuracy.
