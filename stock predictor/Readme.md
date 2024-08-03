# Stock Price Prediction

## Project Description

This Jupyter Notebook explores stock price prediction using machine learning models. The goal is to predict whether the closing price of a stock will be higher or lower the next day based on historical data. 

## Setup Instructions

1. **Install Required Libraries:**

(If `xgboost-gpu` doesn't work, you might need to build XGBoost from source with GPU support - see [https://xgboost.readthedocs.io/en/stable/build.html](https://xgboost.readthedocs.io/en/stable/build.html))

2. **CUDA and Drivers:** Ensure you have CUDA Toolkit and compatible NVIDIA drivers installed if you want to leverage GPU acceleration with XGBoost.

3. **Data:** Place your stock price data in a CSV file (e.g., `stock_data.csv`).  Make sure it has columns like 'Date', 'Open', 'High', 'Low', 'Close', etc.

## How to Run

1. **Open the Jupyter Notebook:** Launch Jupyter Notebook and open the notebook file (e.g., `stock_prediction.ipynb`).

2. **Execute Cells:** Run the cells in the notebook sequentially. You might need to adjust the file path to your data in the data loading section.

## Analysis

The notebook performs the following steps:

1. **Feature Engineering:**  Calculates new features like 'open-close' (difference between opening and closing prices) and 'low-high' (difference between low and high prices).

2. **Target Variable:** Creates a binary target variable indicating whether the closing price increased (1) or decreased (0) the next day.

3. **Data Splitting:** Splits the data into training and validation sets.

4. **Model Training:** Trains three models:
   - Logistic Regression
   - Support Vector Machine (SVC)
   - XGBoost (with GPU acceleration if available)

5. **Evaluation:** Evaluates the models using the ROC AUC score on both the training and validation sets.

## Results

The notebook prints the training and validation accuracy (ROC AUC) for each model. 

## Further Exploration

- **Hyperparameter Tuning:** Experiment with hyperparameters of the models to improve performance.
- **Feature Engineering:** Explore additional features that might be predictive of stock price movements.
- **Model Selection:**  Evaluate other machine learning models for this task.
