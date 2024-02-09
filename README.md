 # SPAM HAM Classification Model with Streamlit

## Overview

This project showcases a machine learning model for classifying emails as either spam or ham using Streamlit, a popular Python framework for creating interactive web applications. The model is trained using a dataset of labeled emails and can accurately predict the category of new emails. The project includes a web interface that allows users to input email content and receive predictions, making it easy to assess the model's performance and explore various email classification scenarios.

## Features

- **Web Interface:**
  - User-friendly interface with an intuitive design for inputting email text and obtaining predictions.

- **Pipeline Architecture:**
  - Modularized pipeline structure for training and prediction, enabling easy integration with other applications or services.
  - Streamlit-based interface for seamless deployment and user interaction.

## Installation

1. Clone the GitHub repository:
```
$ git clone https://github.com/seetharamarao817/spam_ham_classifier
```
2. Install the required Python packages:
```
$ cd spam-ham-classification
$ pip install -r requirements.txt
```

## Running the Project

1. Run the training pipeline to generate the model:
```
$ python pipelines/training_pipeline.py
```
2. Execute the prediction pipeline to evaluate the model's performance:
```
$ python pipelines/prediction_pipeline.py
```
3. Launch the Streamlit web application:
```
$ streamlit run streamlit_app.py
```

## Usage

Once the Streamlit application is running, navigate to the URL provided in the terminal. The interface consists of the following:

- **Email Input:** Provide the email text or subject line you want to classify as spam or ham.
- **Prediction:** The model predicts the email category and displays the result.
- **Data Download:** Download the labeled data used for training the model.

## Contributions

We welcome contributions to improve the project further. You can contribute by:

- Suggesting improvements to the model's architecture or training process.
- Implementing additional features to the web interface.
- Providing additional labeled data for model retraining.
- Creating tutorials or documentation to help users understand and use the project effectively.

For any questions or suggestions, feel free to open an issue in the project's repository.
