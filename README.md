# Deep_learning_project_classification-
 MNIST Handwritten Digit Clasification using Deep Learning (Neural Network)


# MNIST Handwritten Digit Classification

This project demonstrates a deep learning model for classifying handwritten digits from the MNIST dataset.

## Overview

The model utilizes a neural network with two hidden layers, achieving high accuracy in recognizing handwritten digits.  A prediction system allows users to input images and receive classifications.

## Technologies Used

*   **TensorFlow/Keras:** For model building and training.
*   **OpenCV (cv2):** For image preprocessing.
*   **Matplotlib and Seaborn:** For data visualization and result display.
*   **NumPy:** For numerical operations.

## Model Architecture

A sequential neural network with the following layers:

1.  **Flatten Layer:** Converts the 28x28 image into a 784-dimensional vector.
2.  **Dense Layer (50 neurons, ReLU activation):**  First hidden layer.
3.  **Dense Layer (50 neurons, ReLU activation):** Second hidden layer.
4.  **Dense Layer (10 neurons, Sigmoid activation):** Output layer with 10 neurons (representing digits 0-9).

## Training

The model is trained using the Adam optimizer and sparse categorical crossentropy loss function. Training data accuracy: 98.9%, Test data accuracy: 97.1%

## Prediction System

The user can input an image, and the system performs the following steps:

1.  Reads the image using OpenCV.
2.  Converts it to grayscale.
3.  Resizes it to 28x28 pixels.
4.  Scales pixel values to the range [0, 1].
5.  Reshapes the image for model input.
6.  Predicts the digit using the trained model.
7.  Displays the predicted digit.

## Getting Started

To run this project, you'll need to have the necessary libraries installed. Refer to the code for dependencies and setup instructions.

## Example Usage
