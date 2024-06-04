# Detecting Fake News with Python and Machine Learning
![image](https://github.com/HAQ-NAWAZ-MALIK/Google-Colab-projects/assets/86514900/c7ae3000-ca76-458f-9408-6955cf64f9ed)

This project aims to build a machine learning model to detect fake news using Python. The model is trained on a dataset of news articles labeled as either "FAKE" or "REAL".

## Dataset

The dataset used in this project is obtained from a Google Drive link (https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view). It is a CSV file containing news articles and their labels.

![image](https://github.com/HAQ-NAWAZ-MALIK/Google-Colab-projects/assets/86514900/1a5ea29a-ceb0-453f-ab85-50a14058c716)



## Prerequisites

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Implementation

1. Import necessary libraries and read the dataset.
2. Perform exploratory data analysis and data visualization.
3. Split the dataset into training and testing sets.
4. Initialize a TfidfVectorizer to convert text data into numerical features.
5. Fit and transform the training data, and transform the testing data using the TfidfVectorizer.
6. Initialize a PassiveAggressiveClassifier model and train it on the transformed training data.
7. Predict on the testing set and calculate the accuracy score.
8. Plot the confusion matrix to visualize the model's performance.
![image](https://github.com/HAQ-NAWAZ-MALIK/Google-Colab-projects/assets/86514900/933bf865-60e0-4bdc-9be8-a60ffb2c60f5)

## Results

The trained model achieves an accuracy of around 93% in classifying news articles as "FAKE" or "REAL". The confusion matrix shows that the model correctly identifies most of the fake news articles with a low false positive rate.

## Conclusion

This project demonstrates how  to detect fake news using Machine Learning Python and popular libraries like Scikit-learn, Matplotlib, and Seaborn. The model achieves a good accuracy, but it's important to note that it may still make some mistakes, and further improvements could be explored.
