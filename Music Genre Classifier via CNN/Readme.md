# Music Genre Classifier via CNN

This project aims to classify 30-second audio clips into different genres using TensorFlow and Librosa. The audio samples in .wav format are preprocessed by calculating their Mel-Frequency Cepstral Coefficients (MFCCs), which provide a representation of how the energy varies over time for different perceived frequency bands. In this case, 13 frequency bands are used.

## Dataset

The dataset used is the GTZAN dataset, which contains 999 audio files of various genres. The data is preprocessed and stored in a separate JSON file for better organization and readability. The JSON file contains a dictionary-like data structure with the MFCCs and corresponding genre labels.

## Preprocessing

The `preprocess_data` function processes the audio files in the dataset. It slices each file into multiple segments and generates 13-band MFCCs for each slice. The resulting MFCCs and labels are stored in a dictionary, which is then written to a JSON file.

## Model Architecture

The model architecture is designed using TensorFlow's Keras API. It consists of several convolutional layers, max-pooling layers, batch normalization layers, and a dropout layer. The final layers are a flatten layer, a dense layer with 64 units and ReLU activation, and a dense output layer with softmax activation.

## Training and Evaluation

The data is split into training, validation, and test sets. The model is compiled with the RMSprop optimizer and sparse categorical cross-entropy loss. It is trained for 30 epochs with a batch size of 32.

The training and validation accuracy and loss are plotted using Matplotlib. Additionally, a function `make_prediction` is provided to test the model's predictions on a single audio file.

## Saving and Loading the Model

The trained model is saved to a file using Python's pickle module. Later, the model can be loaded from the same file for further use or inference.

## Additional Features

The notebook includes a function `preprocess_audio` that preprocesses a single audio file for inference. It loads the audio file, pads it to the expected duration, extracts the Mel spectrogram features, and resizes them to match the model's input shape.

The final training and validation accuracies are printed, and a plot of the training and validation accuracy and loss is generated using the `plot_history` function.

## Usage

1. Install the required dependencies: TensorFlow, Librosa, NumPy, Pandas, Matplotlib, and scikit-learn.
2. Run the notebook cells in order to preprocess the data, train the model, and evaluate its performance.
3. Use the `make_prediction` function to test the model's predictions on a single audio file.
4. Optionally, save and load the trained model using the provided code snippets.

Note: The notebook assumes the GTZAN dataset is available in the specified location (`/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/`). Adjust the paths as needed for your local setup.
