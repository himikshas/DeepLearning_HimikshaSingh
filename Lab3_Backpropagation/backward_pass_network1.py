#!usr/bin/env python3

""" Implement backward pass for the above two networks. Print the gradient values for each neuron in each layer.
"""

import numpy as np

np.random.seed(10)


""" Activation Functions """
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


""" Forward Pass """
def forward(x, W, b):
    z = np.dot(W, x) + b                #computes z = Wx + b
    a = relu(z)                         #applying activation function which here I am using ReLU
    y_hat = a                           #prediction computed as only one output neuron so prediction will be equal to activation
    return z, a, y_hat


""" Backward Pass """
def backward(x, y, W, b):

    z, a, y_hat = forward(x, W, b)              #running forward propagation first

    # Loss = (y_hat-y)^2 /2
    dL_dy = y_hat - y                   #derivative of loss function

    delta_output = dL_dy * relu_derivative(z)               #output neuron gradient which actually will be 1
                                                            #if z is negative then delta = 0 so learning stops called dying ReLU problem
    dW = delta_output.reshape(-1,1) @ x.reshape(1,-1)           #computes the gradient of every weight.
    db = delta_output

    print("\n SINGLE LAYER NETWORK :")
    print("Prediction =", y_hat)

    print("\nGradient at Output Neuron")
    print(delta_output)

    print("\nGradient of Weights")
    print(dW)

    print("\nGradient of Bias")
    print(db)


""" Random Initialization """
x = np.random.randn(4)

W = np.random.randn(1,4)

b = np.random.randn(1)

y = np.array([1.0])

backward(x,y,W,b)