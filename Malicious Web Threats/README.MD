# Malicious Web Threats Analysis

This Jupyter Notebook analyzes a dataset containing records of suspicious web traffic interactions. The goal is to detect and classify these interactions as potentially malicious or benign using machine learning techniques.

## Dataset

The dataset used in this notebook is the "Cybersecurity Malicious Web Threats" dataset from Kaggle. It contains information about web traffic records flagged as suspicious, including bytes transferred, source and destination IP addresses, protocol, response codes, and detection types.

## Data Preprocessing

The notebook begins with data preprocessing steps, including:

1. Handling missing data
2. Converting time-related columns to datetime format
3. Standardizing text data (e.g., converting country codes to uppercase)

## Data Transformation

Several data transformation techniques are applied to prepare the data for machine learning models:

1. Feature Engineering: A new feature 'duration_seconds' is created to measure the duration of each web session.
2. Scaling: Numerical features like 'bytes_in', 'bytes_out', and 'duration_seconds' are standardized using z-score scaling.
3. One-Hot Encoding: The categorical feature 'src_ip_country_code' is one-hot encoded.

## Exploratory Data Analysis (EDA)

The notebook performs various exploratory data analysis tasks, including:

1. Descriptive Statistics: Summary statistics for numerical features.
2. Correlation Analysis: Investigating relationships between numerical features.
3. Distribution Analysis: Examining the distribution of key features using visualizations like histograms and box plots.

## Machine Learning Models

Two machine learning models are implemented and evaluated:

1. Random Forest Classifier
2. Neural Network (with various architectures)

The models are trained on the transformed data, and their performance is evaluated using accuracy scores, classification reports, and visualizations of the training history.

## Usage

1. Import the required libraries and load the dataset.
2. Follow the data preprocessing and transformation steps outlined in the notebook.
3. Explore the dataset using the provided Exploratory Data Analysis (EDA) techniques.
4. Train and evaluate the machine learning models using the provided code.
5. Analyze the model performance metrics and visualizations to gain insights into the suspicious web threat interactions.

## Dependencies

The following Python libraries are required to run this notebook:

- pandas
- seaborn
- networkx
- matplotlib
- scikit-learn
- tensorflow

Make sure to install these libraries before running the notebook.

## Acknowledgments

This notebook was created by HAQ NAWAZ MALIK. The dataset used in this project is available on Kaggle: [Cybersecurity Suspicious Web Threat Interactions](https://www.kaggle.com/datasets/jancsg/cybersecurity-suspicious-web-threat-interactions/data).
