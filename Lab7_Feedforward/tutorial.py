#!usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

""" Create DataLoaders """
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


""" Define Device """
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using:", device)

""" Define Neural Network"""
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)

        return logits


# Create model
model = NeuralNetwork().to(device)

print(model)


""" Define loss function and optimizer """
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3
)


""" Training Function """
def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        # Move data to CPU/GPU
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        pred = model(X)

        # Calculate loss
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Reset gradients
        optimizer.zero_grad()

        if batch % 100 == 0:

            loss_value = loss.item()

            current = batch * len(X)

            print(
                f"loss: {loss_value:>7f}"
                f"  [{current:>5d}/{size:>5d}]"
            )


""" Testing Function """
def test(dataloader, model, loss_fn):

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            # Prediction
            pred = model(X)

            # Calculate loss
            test_loss += loss_fn(
                pred,
                y
            ).item()

            # Calculate correct predictions
            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches

    accuracy = correct / size

    print(
        f"Test Error: "
        f"Accuracy: {(100 * accuracy):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}"
    )


""" Training """
epochs = 5

for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}\n-------------------------------")

    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    test(
        test_dataloader,
        model,
        loss_fn
    )


print("Done!")


""" Prediction """

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

model.eval()

x, y = test_data[0]

with torch.no_grad():

    x = x.to(device)

    pred = model(x)

    predicted_class = pred.argmax(1).item()

print(
    f"Predicted: {classes[predicted_class]}"
)

print(
    f"Actual: {classes[y]}"
)