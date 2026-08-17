#!usr/bin/env python3

import numpy as np

np.random.seed(42)

def relu(z):
    return max(0, z)

def network1():

    x = np.random.randn(4)
    W = np.random.randn(1,4)
    b = np.random.randn(1)

    z = 0

    for i in range(4):
        z += W[0][i] * x[i]

    z += b[0]

    a = relu(z)

    y_hat = a
    y = 1

    loss = (y_hat - y) ** 2

    print("Input:", x)
    print("Weights:", W)
    print("Bias:", b)
    print("z =", z)
    print("Activation =", a)
    print("Prediction =", y_hat)
    print("Loss =", loss)

network1()