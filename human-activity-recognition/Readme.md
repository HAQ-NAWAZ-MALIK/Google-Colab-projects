
# Human Activity Recognition using Machine Learning

This Jupyter Notebook implements various machine learning models to classify human activities based on data collected from smartphones' accelerometers and gyroscopes. The goal is to accurately classify six different activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying.

## Dataset

The dataset used in this notebook is the Human Activity Recognition dataset, which includes recordings from 30 participants performing the six activities while wearing a smartphone on their waist. The dataset is split into a training set and a test set.

## Exploratory Data Analysis

The notebook starts with exploratory data analysis (EDA) to gain insights into the dataset. It includes:

- Checking for duplicates and missing values
- Visualizing the distribution of activities
- Analyzing the distributions of various features across different activities
- Dimensionality reduction using PCA and t-SNE for data visualization

## Data Preprocessing

The data is preprocessed by dropping the 'subject' and 'Activity' columns, as they are not required for training the models.

## Models

The following machine learning models are implemented and evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest

For each model, hyperparameter tuning is performed using the RandomizedSearchCV from scikit-learn. The models are trained on the training set and evaluated on the test set using various performance metrics, including accuracy, confusion matrix, and classification report.

## Usage

1. Import the required libraries and load the dataset.
2. Perform exploratory data analysis and data preprocessing.
3. Split the data into training and test sets.
4. Train and evaluate each machine learning model using the provided code.
5. Analyze the performance metrics and choose the best-performing model for your use case.

## Dependencies

The following Python libraries are required to run this notebook:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Make sure to install these libraries before running the notebook.

## Acknowledgments

This notebook was created by Haq NAwaz Malik . The dataset used in this project is available at [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones].
```
