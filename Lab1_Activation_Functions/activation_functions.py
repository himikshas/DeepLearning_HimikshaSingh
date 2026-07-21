#!/usr/bin/python3

""" Generate 100 equally spaced values between -10 and 10. Call this list as  z.
    Implement the following functions and its derivative.Use z as input and plot both the function
    outputs and its derivative outputs.
    a.  Sigmoid
    b.  Tanh
    c.  ReLU (Rectified Linear Unit)
    d.  Leaky ReLU
    e.  Softmax (no need for visualization)
"""

import numpy as np
import matplotlib.pyplot as plt


z = np.linspace(-10, 10, 100)


""" a. Sigmoid """
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

""" b. tanh """
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

""" c. ReLU """
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

""" d. Leaky ReLU """
def leaky_relu(x):
    alpha = 0.01
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x):
    alpha = 0.01
    return np.where(x > 0, 1, alpha)

""" e. Softmax """
def softmax(x):
    exp_values = np.exp(x - np.max(x))                  #numerical stability - as if the x is huge the exponential number would be very high so we subtract it from max input value
    return exp_values / np.sum(exp_values)

softmax_output = softmax(z)


""" Plot the functions """
def plot_activation(x, y, dy, title):
    plt.figure(figsize=(8,5))
    plt.plot(x, y, label=title, linewidth=2)
    plt.plot(x, dy, label="Derivative", linewidth=2)
    plt.title(title)
    plt.xlabel("z")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()


""" Plot the activation functions """
plot_activation(z, sigmoid(z), sigmoid_derivative(z), "Sigmoid")

plot_activation(z, tanh(z), tanh_derivative(z), "tanh")

plot_activation(z, relu(z), relu_derivative(z), "ReLU")

plot_activation(z, leaky_relu(z), leaky_relu_derivative(z), "Leaky ReLU")


""" Output of Softmax """
print("Softmax Output:")
print(softmax_output)
print("\nSum of Softmax values =", np.sum(softmax_output))


""" Observations """
print("\nObservations:")
print("\na. Sigmoid Function\nThe sigmoid function has an S-shaped curve.\nOutput ranges from 0 to 1.\nIt is NOT zero-centered.\nThe derivative is maximum at z = 0 and approaches 0 for very large positive and negative inputs.\nSuffers from the vanishing gradient problem means gradient or derivative become extremely small almost close to zero.\nMin Value : 0 (approaches 0)\nMax Value : 1 (approaches 1)")
print("\nb. tanh Function\nThe tanh function is also S-shaped.\nOutput ranges from -1 to 1.\nIt IS zero-centered.\nThe derivative is maximum at z = 0 and becomes nearly 0 for  very large positive and negative inputs.\nAlso suffers from the vanishing gradient problem, but performs better than sigmoid because it is zero-centered.\nMin Value : -1\nMax Value : 1 ")
print("\nc. ReLU Function\n# - Output is 0 for all negative inputs.\nOutput increases linearly for positive inputs.\nNot zero-centered.\nGradient is 0 for negative inputs and 1 for positive inputs.\nDoes not suffer from vanishing gradients for positive inputs.\nCan suffer from the Dead ReLU problem.\nMin Value : 0\nMax Value : Infinity")
print("\nd. Leaky ReLU\nSimilar to ReLU but allows a small negative output.\nOutput is alpha*x for negative values and x for positive values.\nNearly zero-centered since outputs can be negative.\nGradient is alpha (0.01) for negative inputs and 1 for positive inputs.\nPrevents the Dead ReLU problem.\nMin Value : -Infinity\nMax Value : Infinity")