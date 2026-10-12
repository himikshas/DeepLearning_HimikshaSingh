#!usr/bin/env python3

""" Implement CNN using PyTorch for image classification using cifar10 dataset -
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html .
    Plot train error vs increasing number of layers. After some point, the training error increases
    with the number of layers.
"""


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)


""" CIFAR-10 dataset """
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


""" Data loaders """
train_loader = DataLoader(
    training_data,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False
)


""" CNN with variable number of convolution layers """
class CNN(nn.Module):

    def __init__(self, num_layers):
        super().__init__()                      #using numbers as argument for number of cnn layers (num_layers)

        layers = []

        in_channels = 3         #CIFAR-10 has RGB images

        channels = [32, 64, 128, 256, 256]

        for i in range(num_layers):                     #loop runs according to the number of layers requested

            out_channels = channels[i]                  #for each layer filters are decided

            layers.append(                              #adding the conv layer of 3 * 3
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(nn.ReLU())                #add ReLu

            # Pool after every 2 convolution layers(reduces spatial dimensions by half)
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(2, 2))

            in_channels = out_channels                      #updating input channels

        self.features = nn.Sequential(*layers)              #convert list into sequence of different con layers

        # Determine feature size automatically(as output size changes and depends on conv layers)
        with torch.no_grad():

            dummy = torch.zeros(1, 3, 32, 32)               #dummy image to determine shape and gradients are not reqd

            output = self.features(dummy)

            feature_size = output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(                    #After the convolution layers, we flatten the feature maps.
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )


    def forward(self, x):
                                                #forward function tells how input data should flow through NN
        x = self.features(x)                    #different from forward pass

        x = self.classifier(x)

        return x


# Function to train a model(for a single cnn layer)
def train_model(num_layers, epochs=5):

    model = CNN(num_layers).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Loss
            loss = loss_fn(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # Calculate training error
    model.eval()

    correct = 0             #counting correct prediction
    total = 0               #total prediction

    with torch.no_grad():

        for images, labels in train_loader:             #only trainloader not test as just want training error

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)            #for every image model produces 10 scores and torch.max finds largest

            total += labels.size(0)

            correct += (                                  #count correct predictions
                predicted == labels
            ).sum().item()

    train_accuracy = 100 * correct / total

    train_error = 100 - train_accuracy

    print(
        f"Layers: {num_layers}, "
        f"Training Accuracy: {train_accuracy:.2f}%, "
        f"Training Error: {train_error:.2f}%"
    )

    return train_error


# Test different numbers of layers
layer_numbers = [1, 2, 3, 4, 5]

train_errors = []

for layers in layer_numbers:

    error = train_model(
        num_layers=layers,
        epochs=5
    )

    train_errors.append(error)


# Plot training error vs number of layers
plt.figure(figsize=(8, 5))

plt.plot(
    layer_numbers,
    train_errors,
    marker="o"
)

plt.xlabel("Number of Convolutional Layers")

plt.ylabel("Training Error (%)")

plt.title(
    "Training Error vs Number of CNN Layers"
)

plt.xticks(layer_numbers)

plt.grid()

plt.show()