#!usr/bin/env python3

import numpy as np


def batch_normalization(X, gamma=1.0, beta=0.0, eps=1e-5):
    # Mean and variance across the batch
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)

    # Normalize
    X_normalized = (X - mean) / np.sqrt(variance + eps)

    # Scale and shift
    output = gamma * X_normalized + beta

    return output


def layer_normalization(X, gamma=1.0, beta=0.0, eps=1e-5):
    # Mean and variance for each sample
    mean = np.mean(X, axis=1, keepdims=True)
    variance = np.var(X, axis=1, keepdims=True)

    # Normalize
    X_normalized = (X - mean) / np.sqrt(variance + eps)

    # Scale and shift
    output = gamma * X_normalized + beta

    return output


# Example input
X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 4.0, 6.0, 8.0],
    [3.0, 6.0, 9.0, 12.0]
])

print("Original Input:")
print(X)

print("\nBatch Normalization:")
print(batch_normalization(X))

print("\nLayer Normalization:")
print(layer_normalization(X))