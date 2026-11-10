"""Backpropagation from scratch: a tiny 2-layer network learns XOR (NumPy only)."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_xor(hidden=4, lr=1.0, epochs=5000, seed=0):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, hidden)); b1 = np.zeros((1, hidden))
    W2 = rng.normal(0, 1, (hidden, 1)); b2 = np.zeros((1, 1))

    for _ in range(epochs):
        # Forward pass
        h = sigmoid(X @ W1 + b1)
        out = sigmoid(h @ W2 + b2)

        # Backward pass (chain rule, layer by layer)
        d_out = (out - y) * out * (1 - out)      # error at output layer
        dW2 = h.T @ d_out
        db2 = d_out.sum(axis=0, keepdims=True)
        d_h = (d_out @ W2.T) * h * (1 - h)       # error sent back to hidden layer
        dW1 = X.T @ d_h
        db1 = d_h.sum(axis=0, keepdims=True)

        # Gradient descent update
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

    h = sigmoid(X @ W1 + b1)
    return sigmoid(h @ W2 + b2)


if __name__ == "__main__":
    preds = train_xor()
    print("XOR predictions (target 0, 1, 1, 0):")
    print(np.round(preds.ravel(), 3))
