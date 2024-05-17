# Hate Speech Detection using Deep Learning

This Jupyter Notebook implements a deep learning model for hate speech detection using TensorFlow and Keras. The model is trained on a dataset of labeled tweets, and it classifies the tweets into three categories: hate speech, offensive language, or neither.

## Dataset

The dataset used in this notebook is a CSV file containing labeled tweets. The file should have the following columns:

- `tweet`: The text of the tweet.
- `class`: The label of the tweet, with 0 representing "neither", 1 representing "offensive language", and 2 representing "hate speech".

## Model Architecture

The model consists of the following layers:

1. An Embedding layer to convert the input text into vector representations.
2. A Bidirectional LSTM layer to capture the context and learn the patterns in the sequence.
3. A Dense layer with ReLU activation for non-linearity.
4. A BatchNormalization layer for stable and fast training.
5. A Dropout layer to prevent overfitting.
6. A final Dense layer with softmax activation for multi-class classification.

## Usage

1. Import the necessary libraries and load the dataset.
2. Preprocess the text data by removing punctuations, converting to lowercase, and removing stopwords.
3. Split the data into training and validation sets.
4. Tokenize the text and pad the sequences to a fixed length.
5. Define the model architecture and compile the model with appropriate loss and metrics.
6. Train the model with the training data and validate it on the validation data.
7. Evaluate the model's performance using accuracy and loss metrics.

## Dependencies

The following Python libraries are required to run this notebook:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- WordCloud
- TensorFlow
- Keras

Make sure to install these libraries before running the notebook.

## Acknowledgments

This notebook was created by HAQ NAWAZ MALIK  The dataset used in this project is available in same Repo
