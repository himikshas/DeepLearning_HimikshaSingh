#!usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim


# Create simple dataset
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0]
])


# Neural network
class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Different optimizers
optimizers = {
    "SGD": lambda model: optim.SGD(
        model.parameters(),
        lr=0.01
    ),

    "Momentum": lambda model: optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    ),

    "AdaGrad": lambda model: optim.Adagrad(
        model.parameters(),
        lr=0.01
    ),

    "RMSprop": lambda model: optim.RMSprop(
        model.parameters(),
        lr=0.01
    ),

    "Adam": lambda model: optim.Adam(
        model.parameters(),
        lr=0.01
    )
}


# Train using each optimizer
for name, optimizer_function in optimizers.items():

    model = SimpleNN()

    criterion = nn.MSELoss()

    optimizer = optimizer_function(model)

    print("\nOptimizer:", name)

    for epoch in range(1000):

        # Forward pass
        output = model(X)

        # Calculate loss
        loss = criterion(output, y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

    print("Final Loss:", loss.item())

    print(
        "Weight:",
        model.linear.weight.item()
    )

    print(
        "Bias:",
        model.linear.bias.item()
    )