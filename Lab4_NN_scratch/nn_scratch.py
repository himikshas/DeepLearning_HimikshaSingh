#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

#Dataset given
x = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

y = np.array([
    [0], [1], [1], [0]
])

#Initializing weights
np.random.seed(42)

w = np.random.randn(3, 1) * 0.1

alpha = 0.01
iterations = 1000

loss_history = []

#Activation function using sigmoid here
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Binary cross entropy loss mostly done for 0 and 1 dataset
def binary_entropy(y, y_hat):

    epsilon = 1e-9

    loss = -np.mean(
        y * np.log(y_hat + epsilon)
        + (1 - y) * np.log(1 - y_hat + epsilon)
    )

    return loss


# Training
for i in range(iterations):

    # Forward Pass

    # Weighted sum
    z = x @ w

    # Activation
    y_hat = sigmoid(z)


    # Calculate Loss

    loss = binary_entropy(y, y_hat)

    loss_history.append(loss)


    # Backward Pass

    # Gradient of loss with respect to z
    dz = y_hat - y

    # Gradient of weights
    dW = (x.T @ dz) / len(x)


    # Update weights
    w = w - alpha * dW


    # Print loss
    if i % 100 == 0:
        print("Iteration:", i, "Loss:", loss)



z = x @ w
y_hat = sigmoid(z)

predictions = (y_hat >= 0.5).astype(int)



print("\nFinal Weights:")
print(w)

print("\nPredicted probabilities:")
print(y_hat)

print("\nPredicted classes:")
print(predictions)

print("\nActual classes:")
print(y.astype(int))


# Accuracy
accuracy = np.mean(predictions == y) * 100

print("\nAccuracy:", accuracy, "%")


# Plot Loss
plt.plot(loss_history)

plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training Loss over 1000 Iterations")

plt.grid(True)
plt.show()
