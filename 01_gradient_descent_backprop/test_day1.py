import numpy as np
from gradient_descent import make_data, gradient_descent
from backprop_xor import train_xor


def test_gradient_descent_recovers_line():
    x, y = make_data()
    w, b, losses = gradient_descent(x, y)
    assert abs(w - 3.0) < 0.2
    assert abs(b - 2.0) < 0.2
    assert losses[-1] < losses[0]


def test_backprop_learns_xor():
    preds = train_xor().ravel()
    assert np.round(preds).tolist() == [0.0, 1.0, 1.0, 0.0]
