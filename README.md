# Neural Network from Scratch – Optimized Version

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-latest-orange.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains an **optimized neural network implementation from scratch** in Python using only NumPy. The model is designed to classify handwritten digits from the **MNIST dataset**.

It builds on a previous baseline implementation that achieved low performance (~20% accuracy). This **optimized version** integrates modern techniques to achieve **97–98% test accuracy** on MNIST.

## Project Overview

This project demonstrates how to implement and train a neural network without relying on deep learning frameworks such as TensorFlow or PyTorch. It highlights both the **mathematical foundations** and **engineering practices** required for stable training.

### Main Features
- Fully connected (multi-layer perceptron) network
- Configurable hidden layers and architecture
- Multiple activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU)
- Softmax output for multi-class classification
- Cross-entropy loss function

### Key Optimizations
- Mini-batch training
- Adam optimizer with bias correction
- He/Xavier weight initialization
- L2 regularization (weight decay)
- Optional dropout
- Gradient clipping
- Learning rate scheduling
-  Early stopping
- Numerically stable softmax
- Per-feature min–max normalization

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn (for loading MNIST)
- tqdm (for training progress visualization)

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn tqdm
```

## Usage

Example script for training the model on MNIST:

```python
from sklearn.datasets import fetch_openml
from neural_network import NeuralNetwork, one_hot_encode

# Load MNIST dataset
mnist = fetch_openml(name="mnist_784", version=1, as_frame=False)
X_all = mnist.data.T.astype("float32")
y_all = mnist.target.astype(int)

# Train/test split
train_size = 60000
X_train = X_all[:, :train_size]
X_test  = X_all[:, train_size:]
y_train = one_hot_encode(y_all[:train_size], 10)
y_test  = one_hot_encode(y_all[train_size:], 10)

# Initialize model
nn = NeuralNetwork(
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    activation="relu",
    num_labels=10,
    architecture=[512, 256, 128],
    seed=42
)

# Train the model
nn.fit(
    lr=1e-3,
    epochs=30,
    batch_size=128,
    lam=1e-4,
    dropout_keep=1.0,
    grad_clip=5.0,
    early_stopping_patience=8
)

# Visualize results
nn.plot_cost(lr_used=1e-3)
nn.plot_accuracies(lr_used=1e-3)
```

## Network Architecture

Example configuration:

- **Input layer**: 784 neurons (28×28 pixels)
- **Hidden layer 1**: 512 neurons (ReLU)
- **Hidden layer 2**: 256 neurons (ReLU)
- **Hidden layer 3**: 128 neurons (ReLU)
- **Output layer**: 10 neurons (softmax)

## Training Process

### Forward Propagation
1. Linear combination: `Z = WX + b`
2. Non-linear activation (ReLU, Tanh, etc.)
3. Softmax at output layer

### Loss Calculation
- Cross-entropy loss with optional L2 regularization

### Backward Propagation
1. Gradient computation with chain rule
2. Derivatives of activation functions
3. Parameter gradients for each layer

### Parameter Update
1. Adam optimizer with bias correction
2. Gradient clipping if norms exceed threshold

### Training Control
- Mini-batch updates
- Learning rate decay every 30 epochs
- Early stopping when validation accuracy stalls

## Results

- **Training accuracy**: ~99%
- **Test accuracy**: ~97.5–98.2%
- **Convergence**: Stable after 20–30 epochs

Example training log:
```
Epoch   5 | Cost 0.0910 | Train 99.46% | Test 97.80% | LR 1.00e-03
Epoch  10 | Cost 0.0523 | Train 99.72% | Test 98.05% | LR 1.00e-03
Epoch  15 | Cost 0.0401 | Train 99.85% | Test 98.15% | LR 1.00e-03
```

## Learning Notes

- **Baseline version**: Basic gradient descent, unstable training, ~20% accuracy
- **Optimized version**: Added Adam, dropout, weight decay, gradient clipping → reliable convergence and high accuracy
- **Key insight**: Simple architectural and optimization improvements can drastically improve results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and distributed under the MIT License.

## Contact

If you have any questions or suggestions, feel free to open an issue or reach out!

---

**Star this repo if you found it helpful!**