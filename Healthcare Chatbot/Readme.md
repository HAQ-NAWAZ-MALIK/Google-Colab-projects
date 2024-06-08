# Healthcare Chatbot: Symptom Diagnosis and Precaution Recommendations

This project is a Python-based healthcare chatbot designed to assist users in diagnosing potential health conditions based on their reported symptoms. The chatbot utilizes machine learning models, specifically a decision tree classifier and a support vector machine (SVM), trained on a healthcare dataset to make predictions.

## Features

- Interactive symptom input: Users can enter their symptoms through a command-line interface, and the chatbot will provide guidance on selecting the appropriate symptom from potential matches.
- Disease prediction: Based on the user's reported symptoms, the chatbot predicts the most likely health condition(s) using the trained machine learning models.
- Condition description: The chatbot provides a detailed description of the predicted condition(s) to help users understand the potential health issue.
- Precaution recommendations: Along with the diagnosis, the chatbot suggests relevant precautions or measures that users can take for the predicted condition.
- Further assistance: After providing the diagnosis and precautions, the chatbot offers additional options for users, such as connecting with a medical professional, finding nearby healthcare facilities, or learning more about the diagnosed condition.

## Installation

To run this project locally, you'll need to have Python and the required libraries installed. You can create a new conda environment and install the dependencies by running the following commands:

```
conda create -n healthcare-chatbot python=3.7
conda activate healthcare-chatbot
pip install pandas scikit-learn pyttsx3
```

Additionally, you'll need to download the required data files (`Training.csv`, `Testing.csv`, `Symptom_severity.csv`, `symptom_Description.csv`, and `symptom_precaution.csv`) and place them in the appropriate directories (`/Data` and `/MasterData`) within the project.

## Usage

1. Open the `Healthcare_Chatbot.ipynb` file in a Jupyter Notebook environment.
2. Run each cell sequentially to load the required libraries, data, and define the necessary functions.
3. When prompted, enter your name to start the chatbot.
4. Follow the instructions to input your symptoms and the duration you've been experiencing them.
5. The chatbot will provide a diagnosis, condition description, and precaution recommendations based on your input.
6. You can choose to explore additional options, such as speaking with a medical professional or finding nearby healthcare facilities.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project was developed as part of a healthcare-related . 
