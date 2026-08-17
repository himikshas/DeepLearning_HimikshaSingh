#!usr/bin/env python3

""" NETWORK 2 """

import numpy as np

np.random.seed(20)


""" Activation """
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)


""" Forward Pass """
def forward(x, W1,b1, W2,b2, W3,b3):

    #input = 4 neurons
    #Hidden Layer 1 = 3 neurons

    z1 = np.dot(W1,x)+b1                        #w1 => 3 * 4    x => (x1, x2, x3, x4)
    a1 = relu(z1)                               #These become the outputs of Hidden Layer 1.

    z2 = np.dot(W2,a1)+b2                       #w2 => 2 * 3
    a2 = relu(z2)

    z3 = np.dot(W3,a2)+b3                       #w3 => 1 * 2
    a3 = relu(z3)

    return z1,a1,z2,a2,z3,a3


""" Backward Pass """
def backward(x,y, W1,b1, W2,b2, W3,b3):

    z1,a1,z2,a2,z3,a3 = forward(x,W1,b1,W2,b2,W3,b3)                #first running forward propagation

    # Loss
    dL = a3-y                           #(prediction - true value)

    # Output Layer
    delta3 = dL*relu_derivative(z3)                 #gradient of output neuron (loss gradient * activation function gradient) either 1 or 0

    dW3 = delta3.reshape(-1,1) @ a2.reshape(1,-1)
    db3 = delta3

    # Hidden Layer 2
    delta2 = (W3.T @ delta3) * relu_derivative(z2)

    dW2 = delta2.reshape(-1,1) @ a1.reshape(1,-1)
    db2 = delta2

    # Hidden Layer 1
    delta1 = (W2.T @ delta2) * relu_derivative(z1)

    dW1 = delta1.reshape(-1,1) @ x.reshape(1,-1)
    db1 = delta1

    # Printing Gradients
    print("\n===== DEEP NETWORK =====")

    print("\nPrediction")
    print(a3)

    print("\nOutput Layer Gradient")
    print(delta3)

    print("\nHidden Layer 2 Gradients")
    for i,g in enumerate(delta2):
        print("Neuron",i+1,"=",g)

    print("\nHidden Layer 1 Gradients")
    for i,g in enumerate(delta1):
        print("Neuron",i+1,"=",g)

    print("\nWeight Gradient W3")
    print(dW3)

    print("\nWeight Gradient W2")
    print(dW2)

    print("\nWeight Gradient W1")
    print(dW1)

    print("\nBias Gradient Layer3")
    print(db3)

    print("\nBias Gradient Layer2")
    print(db2)

    print("\nBias Gradient Layer1")
    print(db1)


""" Random Initialization """

x = np.random.randn(4)

y = np.array([1.0])

# Input → Hidden1
W1 = np.random.randn(3,4)
b1 = np.random.randn(3)

# Hidden1 → Hidden2
W2 = np.random.randn(2,3)
b2 = np.random.randn(2)

# Hidden2 → Output
W3 = np.random.randn(1,2)
b3 = np.random.randn(1)

backward(x,y,W1,b1,W2,b2,W3,b3)