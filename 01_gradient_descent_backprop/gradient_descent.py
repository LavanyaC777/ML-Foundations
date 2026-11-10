"""Gradient descent from scratch: fit a line y = w*x + b to noisy data."""
import numpy as np


def make_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.1, n)  # true w=3, b=2
    return x, y


def gradient_descent(x, y, lr=0.1, epochs=200):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        y_pred = w * x + b
        error = y_pred - y
        loss = np.mean(error ** 2)          # MSE
        dw = 2 * np.mean(error * x)         # d(loss)/dw
        db = 2 * np.mean(error)             # d(loss)/db
        w -= lr * dw                        # step against the gradient
        b -= lr * db
        losses.append(loss)
    return w, b, losses


if __name__ == "__main__":
    x, y = make_data()
    w, b, losses = gradient_descent(x, y)
    print(f"Learned: w={w:.2f}, b={b:.2f} (true: w=3.00, b=2.00)")
    print(f"Loss: start={losses[0]:.3f}, end={losses[-1]:.4f}")
