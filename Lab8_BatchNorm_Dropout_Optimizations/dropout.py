#!usr/bin/env python3

import numpy as np


def dropout(X, dropout_rate=0.5, training=True):
    if not training:
        return X

    # Probability of keeping a neuron
    keep_probability = 1 - dropout_rate

    # Generate random mask
    mask = np.random.binomial(
        1,
        keep_probability,
        size=X.shape
    )

    # Inverted dropout
    output = X * mask / keep_probability

    return output


# Example input
X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
])

print("Original Input:")
print(X)

print("\nAfter Dropout:")
print(dropout(X, dropout_rate=0.5))

print("\nDuring Testing:")
print(dropout(X, dropout_rate=0.5, training=False))