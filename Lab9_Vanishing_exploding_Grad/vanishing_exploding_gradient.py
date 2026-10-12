#usr/bin/env python3

""" Write a program to simulate vanishing & exploding gradient problems.
    https://machinelearningmastery.com/visualizing-the-vanishing-gradient-problem/
    Plot the gradient values to demonstrate the issues with training a very deep network.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Number of layers
num_layers = 20



""" Create Deep Network """
def create_network(weight_value):

    layers = []

    for i in range(num_layers):

        layer = nn.Linear(10, 10)               #creating a linear layer

        # Set all weights to the specified value
        with torch.no_grad():                               #dont want pytorch to assign that's why .no_grad so no gradients will calculate
            layer.weight.fill_(weight_value)
            layer.bias.zero_()

        layers.append(layer)
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)                   #connects all the layers sequentially so one layer becomes input to next


""" Calculating Gradients """
def calculate_gradients(weight_value):

    model = create_network(weight_value)

    # Input
    x = torch.ones(1, 10)               #creates x=[1,1,1,1,1,1,1,1,1,1]  (1 sample, 10 input features)

    # Forward pass
    output = model(x)

    # Simple loss
    loss = output.sum()

    # Backpropagation
    loss.backward()

    gradients = []

    # Store gradient magnitude of each layer
    for layer in model:

        if isinstance(layer, nn.Linear):

            gradient = layer.weight.grad.abs().mean().item()

            gradients.append(gradient)

    return gradients


# Calculate gradients

# Small weights -> vanishing gradient
vanishing_gradients = calculate_gradients(0.1)

# Large weights -> exploding gradient
exploding_gradients = calculate_gradients(2.0)


""" Printing Gradient Values """
print("Vanishing gradients:")

for i, value in enumerate(vanishing_gradients):
    print(
        "Layer",
        i + 1,
        ":",
        value
    )


print("\nExploding gradients:")

for i, value in enumerate(exploding_gradients):
    print(
        "Layer",
        i + 1,
        ":",
        value
    )


""" Plot the Gradients """
layers = range(1, num_layers + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    layers,
    vanishing_gradients,
    marker="o",
    label="Vanishing Gradient"
)

plt.plot(
    layers,
    exploding_gradients,
    marker="o",
    label="Exploding Gradient"
)

plt.yscale("log")

plt.xlabel("Layer")
plt.ylabel("Average Absolute Gradient")
plt.title("Vanishing and Exploding Gradients")

plt.legend()
plt.grid(True)

plt.show()