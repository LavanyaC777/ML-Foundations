# 01 – Gradient Descent & Backpropagation (from scratch)

Two small NumPy-only scripts that show how neural networks learn.

| File | What it shows |
|------|---------------|
| `gradient_descent.py` | Fits a line `y = w*x + b` by following the gradient of the MSE loss |
| `backprop_xor.py` | A 2-layer network learns XOR; gradients are computed by hand with the chain rule |
| `test_day1.py` | Tests that both scripts really learn |

## Idea in 3 sentences
- The **loss** tells us how wrong the model is.
- **Gradient descent** changes the weights a small step (learning rate) in the direction that lowers the loss.
- **Backpropagation** uses the chain rule to get the gradient for every weight, layer by layer, from output back to input.

## Run
```bash
pip install -r requirements.txt
python gradient_descent.py
python backprop_xor.py
pytest
```

## Expected output
```
Learned: w=3.00, b=2.00 (true: w=3.00, b=2.00)
XOR predictions (target 0, 1, 1, 0): [0.007 0.981 0.981 0.024]
```

## What I learned
- Too large a learning rate: training jumps around or diverges. Too small: training is very slow.
- XOR cannot be solved by a single line, so a hidden layer is needed.
