#!usr/bin/env python3

""" Implement a 2-layer (input layer, hidden layer and output layer) neural network from scratch for
    the XOR operation. This includes implementing forward and backward passes from scratch.
    The truth table for XOR is given by

    x1          x2          y
    0           0           0
    0           1           1
    1           0           1
    1           1           0

"""

import numpy as np

""" XOR Dataset """
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)                  #float is required as during calculation sometimes become decimal and python will treat them as decimals not integers

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

""" Activation Function """
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

""" Initialize weights and biases """
np.random.seed(7)

#input layer to hidden layer
w1 = np.random.randn(2,2) * 0.5                       # 2 input neurons and 2 hidden layer neurons
b1 = np.zeros((1, 2))                                 # 2 hidden neurons, so we need 2 biases: b1 =[[0, 0]]

#hidden layer to output layer
w2 = np.random.randn(2,1) * 0.5                       # 2 hidden neurons and 1 output neuron
b2 = np.zeros((1,1))                                  # 1 output neuron, so we need only 1 bias:

#If we initialize all weights to zero, the hidden neurons would behave identically and the network would have difficulty learning different features.
#So we start with small/random values.
#0.5  gives smaller initial weights and makes training more stable.
""" Learning Parameters """
alpha = 0.5
epochs = 10000

""" Binary Cross Entropy Loss """
def binary_crossentropy(y, y_hat):

    epsilon = 1e-8

    loss = -np.mean(
       y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon)
       )

    return loss

""" Training """
for epoch in range(epochs):
    z1 = x @ w1 + b1                #forward pass (intput to hidden)
    a1 = sigmoid(z1)

    z2 = a1 @ w2 + b2               #(hidden to output)
    y_hat = sigmoid(z2)

    loss = binary_crossentropy(y, y_hat)              #calculating loss

    dz2 = y_hat - y                 #backward pass (output layer error)
    dw2 = (a1.T @ dz2) / len(x)     #gradient of w2
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(x)       #gradient of b2

    da1 = dz2 @ w2.T                #hidden layer error
    dz1 = da1 * sigmoid_derivative(a1)

    dw1 = (x.T @ dz1) / len(x)      #gradient of w1
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(x)       #gradient of b1


    #update weights and biases
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1

    if epoch % 1000 == 0:           #printing loss
        print("Epoch:", epoch, "Loss:", loss)


""" Testing """
z1 = x @ w1 + b1
a1 = sigmoid(z1)

z2 = a1 @ w2 + b2
y_hat = sigmoid(z2)

predictions = (y_hat >= 0.5).astype(int)

""" Display Results """
print("\nFinal predictions:")
print(predictions)

print("\nActual values:")
print(y.astype(int))

print("\nOutput probabilities:")
print(y_hat)

accuracy = np.mean(predictions == y) * 100

print("\nAccuracy:", accuracy, "%")