**Abstract**：This project aims to deepen the understanding of neural network models through hands-on practice. It is divided into two main parts: MNIST handwritten digit recognition using a Multi-Layer Perceptron (MLP), and CIFAR-10 image classification using a Convolutional Neural Network (CNN).

## Part 1: MNIST Handwritten Digit Recognition with MLP
This section (`code/mlp_xx.py`) implements a fully connected neural network for recognizing MNIST handwritten digits and systematically explores the impact of different hyperparameters on model performance.

**Key Experiments Include:**

1.  **Baseline Model**: A basic MLP model with a `784 -> 128 -> 64 -> 10` architecture was built, using the ReLU activation function.
2.  **Hyperparameter Exploration**:
    *   **Network Depth and Width**: Analyzed the effects of model complexity and overfitting by varying the number of hidden layers and the number of neurons in each layer.
    *   **Activation Functions**: Compared the differences in training speed and final performance among **ReLU**, **Sigmoid**, and **Tanh** activation functions.

## Part 2: CIFAR-10 Image Classification with CNN

This section (`code/cnn_xx.py`) implements a convolutional neural network for CIFAR-10 image classification and investigates the influence of its architecture on performance.

**Key Experiments Include:**

1.  **Baseline Model**: A basic CNN model was implemented and trained to serve as a performance benchmark.
2.  **Architecture Exploration**:
    *   **Kernel Size**: Compared the impact of using **3x3** versus **5x5** convolutional kernels on feature extraction capabilities.
    *   **Network Depth**: Constructed deeper networks by stacking more convolutional/pooling layers and observed the resulting changes in performance.

## Requirements
This project is based on the Python and PyTorch frameworks. Please ensure that all necessary dependencies are installed.

## Note
Due to GitHub's file size limits, the MNIST and CIFAR-10 datasets have not been uploaded to this repository. The PyTorch scripts are configured to automatically download them upon the first run.


