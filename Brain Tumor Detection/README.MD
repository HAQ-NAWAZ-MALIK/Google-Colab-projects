# Brain Tumor Detection Using Image Processing Techniques

This Jupyter Notebook explores various image processing techniques and machine learning models for detecting brain tumors from MRI images. The notebook covers the following key aspects:

## Dataset

The dataset used in this project is the "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle. It contains MRI images of the brain, categorized into two classes: "yes" (tumor present) and "no" (no tumor).

## Data Preprocessing

The notebook includes steps for loading and preprocessing the images, such as resizing, normalization, and smoothing using Gaussian filtering.

## Image Enhancement and Feature Extraction

Several image enhancement techniques are employed, including:

- Local Binary Patterns (LBP)
- Histogram of Oriented Gradients (HOG)
- Scale-Invariant Feature Transform (SIFT)
- VGG16 pre-trained model for feature extraction
- Prewitt and Butterworth Low Pass filters

These techniques are used to extract relevant features from the MRI images, which are then used for classification.

## Classification

The notebook implements and evaluates various machine learning models for brain tumor classification, including:

- Random Forest Classifier
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)
- Fully Connected Neural Network

The performance of these models is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Evaluation and Comparative Analysis

A detailed comparative analysis is provided, summarizing the performance of different models, feature extraction techniques, and image enhancement methods. This analysis helps in understanding the trade-offs and selecting the most appropriate approach for the brain tumor detection task.

## Requirements

To run this notebook, you'll need to have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- scikit-learn
- scikit-image
- Tensorflow
- Keras
- PyTorch
- Torchvision
- Torchsummary

You can install these libraries using pip or conda, depending on your Python environment.

## Usage

1. Download the "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle and extract it to a local directory.
2. Update the file paths in the notebook to point to the location of the extracted dataset.
3. Run the notebook cells in order to execute the code and explore the results.

Note: This notebook assumes a basic understanding of Python, machine learning, and image processing concepts.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the repository.
