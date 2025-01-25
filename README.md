# Digit Classification Using Neural Networks
This project demonstrates how to train a neural network to classify handwritten digits using the Digits dataset from scikit-learn.
The dataset consists of 1797 8x8 pixel images of handwritten digits, each labeled with the corresponding digit (0-9).
The task is to classify these images into one of 10 classes, representing the digits 0 to 9.

# Procedure
Dataset:
The Digits dataset contains 1797 samples, each representing an 8x8 grayscale image of a digit. The dataset has the following structure:

Features: Each sample has 64 features, representing the pixel intensities of the 8x8 image (flattened into a 1D array).
Labels: There are 10 target classes, corresponding to the digits 0 through 9.

Preprocessing:
Feature Scaling: The features are standardized (zero mean and unit variance) to improve model convergence and speed up training.
Train-Test Split: The dataset is split into a training set (80%) and a testing set (20%).
Model:
A simple feedforward neural network is constructed with the following architecture:

Input Layer: 64 neurons (corresponding to the 64 features of each image).
Hidden Layer 1: 64 neurons with ReLU activation.
Hidden Layer 2: 32 neurons with ReLU activation.
Output Layer: 10 neurons (one for each digit) with softmax activation to perform multi-class classification.

Training:
Optimizer: The model uses the Adam optimizer for training.
Loss Function: Sparse categorical cross-entropy is used since the labels are integers (not one-hot encoded).
Epochs: Training is performed for 50 epochs with a batch size of 16.
Validation Split: 20% of the training data is used as a validation set to monitor the model's performance during training.

Evaluation:
After training, the model is evaluated on the test set to determine its accuracy.

# Output:
Training Output:
During training, the loss and accuracy of both the training and validation sets are displayed after each epoch. 
Here's an example of the output for the first few epochs:

Epoch 1/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - accuracy: 0.1213 - loss: 2.2005 - val_accuracy: 0.2500 - val_loss: 2.0549

Epoch 2/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.4128 - loss: 1.8000 - val_accuracy: 0.5000 - val_loss: 1.5794

...

Epoch 50/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - accuracy: 0.9850 - loss: 0.0778 - val_accuracy: 0.9722 - val_loss: 0.1805

#Test Accuracy:
After training, the model is evaluated on the test dataset. An example of the test accuracy is shown below:


Test Accuracy: 0.9722
This indicates that the model achieved approximately 97.22% accuracy on the test set.

# Predictions:
The model makes predictions on the test set, and the predicted classes are compared with the true classes. An example output is shown below:

Predicted classes: [6 9 3 7 2 1 5 2 5 2 1 9 4 0 4 2 3 7 8 8 4 3 9 7 5 6 
3 5 6 3 4 9 1 4 4 6 9 4 5 8 7 9 8 6 0 6 2 0 7 9 8 9 5 2 7 7 9 8 7 4 3 8 3 5]
True classes: [6 9 3 7 2 1 5 2 5 2 1 9 4 0 4 2 3 7 8 8 4 3 9 7 5 6 3 5 6
3 4 9 1 4 4 6 9 4 5 8 7 9 8 5 0 6 2 0 7 9 8 9 5 2 7 7 1 8 7 4 3 8 3 5]

# Explanation:
Training Output: Displays the loss and accuracy for both the training and validation sets after each epoch.
Test Accuracy: This is the model's accuracy when evaluated on the test dataset.
Predictions: Displays the predicted classes and the true classes for the test dataset,
showing how well the model has learned to classify the digits.

# Notes:
The number of layers, units, epochs, and batch size can be adjusted to further optimize model performance.
The model can also be enhanced by experimenting with different activation functions, optimizers, or regularization techniques.
