# Hand Gesture Recognition Model

This project aims to build a hand gesture recognition model using MediaPipe and a Random Forest classifier. The model is trained on a dataset of hand gesture images and can predict the gesture shown in new images.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Hand gesture recognition is a computer vision task that involves detecting and recognizing hand gestures from images or video streams. This project uses the MediaPipe library for hand landmark detection and a Random Forest classifier to recognize gestures.

## Dataset

The dataset used for this project is assumed to be in a structured format with images of different hand gestures stored in separate directories. Each directory corresponds to a specific gesture.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/hand-gesture-recognition.git
    cd hand-gesture-recognition
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download and prepare your dataset in the specified directory structure.

## Usage

### Preprocess and Train the Model

1. Place your dataset in a folder named `./leapGestRecog`.
2. Run the `trainer.py` script to preprocess the data and train the model:
    ```bash
    python trainer.py
    ```

### Evaluate the Model

1. After training, the model's performance will be evaluated on the test set.
2. The script will output the accuracy, confusion matrix, and classification report.

## Model Training

The training process involves the following steps:

1. **Data Preprocessing**: Images are read, resized, and processed to extract hand landmarks using MediaPipe.
2. **Feature Extraction**: Extracted landmarks are used as features for training the model.
3. **Model Training**: A Random Forest classifier is trained on the extracted features.
4. **Evaluation**: The trained model is evaluated using cross-validation, confusion matrix, and classification report.

## Evaluation

The model is evaluated using:

- Accuracy Score
- Cross-Validation Scores
- Confusion Matrix
- Classification Report

## Results

The model's performance metrics will be displayed in the console after running the `trainer.py` script.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.


