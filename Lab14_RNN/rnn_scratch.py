import numpy as np

""" Data """
text = """
Apple apple apple
apple is a fruit
apple is red
apple is sweet
"""

# Create vocabulary
chars = sorted(list(set(text)))

vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print("Vocabulary size:", vocab_size)
print("Vocabulary:", chars)


""" Hyperparameters """
input_size = vocab_size
hidden_size = 64
output_size = vocab_size

learning_rate = 0.1
epochs = 10000


""" Initialize Parameters """
np.random.seed(42)

# Input -> Hidden
U = np.random.randn(hidden_size, input_size) * 0.01

# Hidden -> Hidden
W = np.random.randn(hidden_size, hidden_size) * 0.01

# Hidden -> Output
V = np.random.randn(output_size, hidden_size) * 0.01

# Hidden bias
b = np.zeros((hidden_size, 1))

# Output bias
c = np.zeros((output_size, 1))


""" Activation Functions """
def softmax(x):

    # Subtract max for numerical stability
    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


""" One-Hot Encoding """
def one_hot(index, size):

    vector = np.zeros((size, 1))

    vector[index] = 1

    return vector


""" Forward Pass """
def forward(inputs, targets):

    # Store hidden states
    h = {}

    # Store outputs
    o = {}

    # Store predictions
    y_hat = {}

    # Initial hidden state
    h[-1] = np.zeros((hidden_size, 1))
    loss = 0

    # Process every timestep
    for t in range(len(inputs)):

        # One-hot encode input
        x_t = one_hot(inputs[t], input_size)

        # ----------------------------------------------------
        # RNN equation
        #
        # h_t = tanh(Ux_t + Wh_(t-1) + b)
        # ----------------------------------------------------

        h[t] = np.tanh(np.dot(U, x_t)+ np.dot(W, h[t - 1])+ b)

        # ----------------------------------------------------
        # Output
        #
        # o_t = Vh_t + c
        # ----------------------------------------------------
        o[t] = np.dot(V, h[t]) + c

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------
        y_hat[t] = softmax(o[t])

        # ----------------------------------------------------
        # Cross entropy loss
        # ----------------------------------------------------
        loss -= np.log(y_hat[t][targets[t], 0] + 1e-12)

    return loss, h, o, y_hat


""" Backpropagation """
def backward(inputs, targets, h, o, y_hat):

    global U, W, V, b, c

    # Initialize gradients

    dU = np.zeros_like(U)
    dW = np.zeros_like(W)
    dV = np.zeros_like(V)

    db = np.zeros_like(b)
    dc = np.zeros_like(c)

    # Gradient flowing backward through hidden state
    dh_next = np.zeros((hidden_size, 1))


    # --------------------------------------------------------
    # Go backwards through time
    # --------------------------------------------------------

    for t in reversed(range(len(inputs))):

        # Gradient of cross entropy + softmax
        #
        # dL/do = y_hat - y
        # ----------------------------------------------------

        do = y_hat[t].copy()
        do[targets[t]] -= 1


        # Output layer gradients
        dV += np.dot(do, h[t].T)
        dc += do


        # Gradient flowing into hidden state
        dh = np.dot(V.T, do) + dh_next


        # tanh derivative
        # d tanh(x) / dx = 1 - tanh(x)^2
        ds = (1 - h[t] * h[t]) * dh

        # Hidden layer gradients
        x_t = one_hot(inputs[t], input_size)

        dU += np.dot(ds, x_t.T)
        dW += np.dot(ds, h[t - 1].T)
        db += ds


        # Pass gradient to previous timestep
        dh_next = np.dot(W.T, ds)


    # --------------------------------------------------------
    # Gradient clipping
    # Prevent exploding gradients
    # --------------------------------------------------------

    for gradient in [dU, dW, dV, db, dc]:

        np.clip(gradient, -5,5, out=gradient)


    return dU, dW, dV, db, dc


""" Training """
for epoch in range(epochs):

    # Create training sequence
    inputs = [
        char_to_idx[ch]
        for ch in text[:-1]
    ]

    targets = [
        char_to_idx[ch]
        for ch in text[1:]
    ]

    # Forward pass
    loss, h, o, y_hat = forward(
        inputs,
        targets
    )

    # Backward pass
    dU, dW, dV, db, dc = backward(inputs, targets, h, o, y_hat)

    # Update parameters using gradient descent
    U -= learning_rate * dU
    W -= learning_rate * dW
    V -= learning_rate * dV

    b -= learning_rate * db
    c -= learning_rate * dc


    # Display loss
    if epoch % 100 == 0:

        print(
            f"Epoch {epoch}, Loss = {loss:.4f}"
        )


""" Test the RNN """
def predict(start_char, length=20):

    current_char = start_char

    hidden = np.zeros((hidden_size, 1))

    result = current_char

    for _ in range(length):

        idx = char_to_idx[current_char]

        x = one_hot(idx, input_size)

        hidden = np.tanh(
            np.dot(U, x)
            + np.dot(W, hidden)
            + b
        )

        output = np.dot(V, hidden) + c

        probabilities = softmax(output)

        next_idx = np.argmax(probabilities)

        current_char = idx_to_char[next_idx]

        result += current_char

    return result


print("\nPrediction:")
print(predict("A", length=20))