# Training BERT to Analyze IMDB Movie Reviews

## Overview

This project demonstrates how to train a BERT (Bidirectional Encoder Representations from Transformers) model to perform sentiment analysis on IMDB movie reviews. The notebook includes steps for loading data, preprocessing, model training, and evaluation.

## Features

- Utilizes a pretrained BERT model for fine-tuning on the IMDB dataset.
- Covers data preprocessing, including tokenization and padding.
- Includes model training and evaluation steps.
- Provides visualizations for understanding model performance.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.7+
- Jupyter Notebook
- Transformers Library (Hugging Face)
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional, for visualizations)

You can install the required packages using pip:

```bash
pip install torch transformers pandas numpy scikit-learn jupyter matplotlib
```

## Getting Started

1. **Clone the repository**:
   

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Training_BERT_To_Analyze_IMDB_Movie_Reviews.ipynb
   ```

4. **Run the cells** in the notebook to preprocess the data, train the model, and evaluate its performance.

## Project Structure

```
bert-imdb-sentiment-analysis/
│
├── Training_BERT_To_Analyze_IMDB_Movie_Reviews.ipynb  # Main Jupyter Notebook
├── README.md                                         # Project Documentation
├── requirements.txt                                  # List of dependencies
└── data/                                             # Directory for dataset (if any)
```

## How to Use

1. **Load the Data**: Ensure you have the IMDB movie reviews dataset. You can download the dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or use any other source.

2. **Data Preprocessing**: The notebook includes steps for tokenizing the text data using the BERT tokenizer, padding, and creating attention masks.

3. **Model Training**: Fine-tune the pretrained BERT model on the IMDB dataset. The notebook provides code for setting up the training loop, optimizer, and loss function.

4. **Model Evaluation**: Evaluate the model's performance on a test set. The notebook includes code for calculating accuracy, precision, recall, and F1-score, as well as visualizing the results.

## Example

this is a brief example of how to use the notebook for training and evaluating the BERT model on IMDB movie reviews:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pretrained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize input
inputs = tokenizer("This movie was fantastic!", return_tensors="pt", padding=True, truncation=True, max_length=512)

# Predict sentiment
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

# Print the sentiment
if predictions.item() == 1:
    print("Positive sentiment")
else:
    print("Negative sentiment")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

