#!/usr/bin/python3

""" Consider the following two networks.
    W is a matrix, x is a vector, z is a vector, and a is a vector.
    y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
    Initialize X and W randomly. Every neuron has a bias term.

    Implement forward pass for the above two networks.
    Print activation values for each neuron at each layer.
    Print the loss value (y^).

    Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z and a are vectors.
    The implementation should not contain any loops."""

import numpy as np

""" Forward Pass Implementation for Two Neural Networks """

np.random.seed(42)

""" ReLU Activation Function """
def relu(z):
    return np.maximum(0, z)

""" Network 1: 4 inputs and 1 output """
def network1():

    x = np.random.randn(4, 1)

    #Weights and Bias
    W = np.random.randn(1, 4)
    b = np.random.randn(1, 1)

    #Forward Pass
    z = np.dot(W, x) + b
    a = relu(z)

    y_hat = a

    y = np.array([[1]])

    loss = np.mean((y_hat - y) ** 2)

    print("=" * 50)
    print("NETWORK 1")
    print("=" * 50)

    print("\nInput (x):")
    print(x)

    print("\nWeights (W):")
    print(W)

    print("\nBias (b):")
    print(b)

    print("\nWeighted Sum (z):")
    print(z)

    print("\nActivation (a):")
    print(a)

    print("\nPrediction (y_hat):")
    print(y_hat)

    print("\nLoss:")
    print(loss)

""" Network 2: 4 inputs and 1 output with 2 hidden layers """
def network2():

    x = np.random.randn(4, 1)

    # Hidden Layer 1
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)

    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)

    # Hidden Layer 2
    W2 = np.random.randn(2, 3)
    b2 = np.random.randn(2, 1)

    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)

    # Output Layer
    W3 = np.random.randn(1, 2)
    b3 = np.random.randn(1, 1)

    z3 = np.dot(W3, a2) + b3
    a3 = relu(z3)

    y_hat = a3

    # Target
    y = np.array([[1]])

    # Loss
    loss = np.mean((y_hat - y) ** 2)

    print("\n\n" + "=" * 50)
    print("NETWORK 2")
    print("=" * 50)

    print("\nInput (x):")
    print(x)

    print("\n------ Hidden Layer 1 ------")
    print("Weights W1:")
    print(W1)

    print("\nBias b1:")
    print(b1)

    print("\nz1:")
    print(z1)

    print("\na1:")
    print(a1)

    print("\n------ Hidden Layer 2 ------")
    print("Weights W2:")
    print(W2)

    print("\nBias b2:")
    print(b2)

    print("\nz2:")
    print(z2)

    print("\na2:")
    print(a2)

    print("\n------ Output Layer ------")
    print("Weights W3:")
    print(W3)

    print("\nBias b3:")
    print(b3)

    print("\nz3:")
    print(z3)

    print("\na3:")
    print(a3)

    print("\nPrediction (y_hat):")
    print(y_hat)

    print("\nLoss:")
    print(loss)


def main():
    network1()
    network2()

if __name__ == "__main__":
    main()