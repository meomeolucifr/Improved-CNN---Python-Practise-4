"""
Neural Network (from scratch) - Optimized, Fully Commented

Features:
- Mini-batch training
- Adam optimizer
- He/Xavier initialization (activation-aware)
- L2 regularization (weight decay)
- Optional Dropout
- Gradient clipping
- Learning rate schedule + Early stopping
- Numerically stable softmax
- Per-feature min-max normalization

Data shape conventions (match your original):
    X: (n_features, m)   -> input data, columns = examples
    y: (n_classes,  m)   -> labels (one-hot encoded), columns = examples
"""


# Imports

import numpy as np                     # Linear algebra, arrays, vectors, matrices
import matplotlib.pyplot as plt        # Plot loss and accuracy curves
from tqdm import tqdm                  # Progress bar for training loop
from typing import List, Optional      # For type hints



# Utility functions


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Stable Softmax activation function (column-wise):
    - Subtract max per column to avoid overflow
    - Convert raw scores into probabilities that sum to 1 (per example)
    Input:  z (n_classes, m)
    Output: probs (n_classes, m)
    """
    z_shift = z - np.max(z, axis=0, keepdims=True)           # Stability: shift logits by max per column
    e = np.exp(z_shift)                                      # Exponentiate shifted logits
    return e / (np.sum(e, axis=0, keepdims=True) + 1e-12)    # Normalize per column; add epsilon to avoid /0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: f(x) = 1 / (1 + e^(-x)); range (0,1)"""
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: f(x) = max(0, x)"""
    return np.maximum(0.0, z)


def tanh_act(z: np.ndarray) -> np.ndarray:
    """Tanh activation: f(x) = tanh(x); range (-1,1)"""
    return np.tanh(z)


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: f(x) = x if x>0 else alpha*x"""
    return np.where(z > 0, z, alpha * z)


def d_activation(name: str, z: np.ndarray) -> np.ndarray:
    """
    Derivative of activation functions (for backprop).
    Input: activation name and pre-activation values z
    Output: elementwise derivative (same shape as z)
    """
    if name == "sigmoid":
        s = sigmoid(z)                          # Reuse forward sigmoid
        return s * (1.0 - s)                    # f'(x) = f(x)(1-f(x))
    if name == "tanh":
        t = np.tanh(z)                          # Reuse forward tanh
        return 1.0 - t * t                      # f'(x) = 1 - tanh^2(x)
    if name == "relu":
        return (z > 0).astype(z.dtype)          # 1 for z>0, else 0
    if name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)       # slope alpha on negative side
    raise ValueError(f"Unknown activation: {name}")


def normalize_per_feature(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalization per feature (row/feature-wise).
    Ensures each feature lies roughly in [0,1].
    """
    x_min = np.min(x, axis=1, keepdims=True)               # Min per feature (row)
    x_max = np.max(x, axis=1, keepdims=True)               # Max per feature (row)
    return (x - x_min) / (x_max - x_min + 1e-9)            # Normalize, add epsilon to avoid /0


def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Convert integer labels into one-hot matrix (n_classes, m).
    Example: num_labels=3, [1,2] -> [[0,0],[1,0],[0,1]]
    """
    return np.eye(num_labels)[x.astype(int)].T              # Build identity; index rows; transpose to (C,m)


def iterate_minibatches(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Generator that yields mini-batches (Xb, Yb) from (X,Y).
    X shape: (n_features, m); Y shape: (n_classes, m)
    """
    m = X.shape[1]                                          # Number of examples (columns)
    idx = np.arange(m)                                      # Indices 0..m-1
    if shuffle:                                             # Shuffle examples if requested
        np.random.shuffle(idx)
    for start in range(0, m, batch_size):                   # Iterate in steps of batch_size
        end = min(start + batch_size, m)                    # Compute end index (handle last partial batch)
        batch_idx = idx[start:end]                          # Slice indices for this batch
        yield X[:, batch_idx], Y[:, batch_idx]              # Yield corresponding data slices



# Neural Network class

class NeuralNetwork:
    def __init__(
        self,
        X: np.ndarray,                   # training inputs, shape (n_features, m_train)
        y: np.ndarray,                   # training labels (one-hot), shape (n_classes, m_train)
        X_test: np.ndarray,              # test inputs, shape (n_features, m_test)
        y_test: np.ndarray,              # test labels (one-hot), shape (n_classes, m_test)
        activation: str,                 # activation for hidden layers: "relu" | "tanh" | "sigmoid" | "leaky_relu"
        num_labels: int,                 # number of output classes
        architecture: List[int],         # list of hidden layer sizes, e.g., [512, 256, 128]
        seed: Optional[int] = 42         # RNG seed for reproducibility
    ):
        """
        Constructor sets data, normalizes features, stores config, prepares containers.
        """
        #  Data copies and normalization 
        self.X_raw, self.X_test_raw = X.copy(), X_test.copy()    # Keep raw copies (optional)
        self.X = normalize_per_feature(self.X_raw)               # Normalize training per feature
        self.X_test = normalize_per_feature(self.X_test_raw)     # Normalize test per feature
        self.y = y.copy()                                        # Copy training one-hot labels
        self.y_test = y_test.copy()                              # Copy test one-hot labels

        #  Store config 
        self.activation = activation                              # Activation name for hidden layers
        assert self.activation in ["relu", "tanh", "sigmoid", "leaky_relu"]  # Validate option
        self.num_labels = num_labels                              # Number of classes (e.g., 10 for MNIST)
        self.m = self.X.shape[1]                                  # Number of training examples (columns)
        self.num_input_features = self.X.shape[0]                 # Input dimensionality (e.g., 784 for MNIST)

        #  Build full layer sizes 
        self.architecture = architecture.copy()                   # Copy hidden sizes
        self.architecture.insert(0, self.num_input_features)      # Prepend input size
        self.architecture.append(self.num_labels)                 # Append output size
        self.L = len(self.architecture)                           # Total number of layers (including input & output)

        #  Sanity checks on shapes 
        assert self.X.shape == (self.num_input_features, self.m)  # Ensure normalized shape
        assert self.y.shape[0] == self.num_labels                 # One-hot rows == classes

        #  Containers for params/state 
        self.parameters = {}                                       # Dict for weights/biases: w1,b1,...,wL-1,bL-1
        self.layers = {}                                           # Dict to store forward pass intermediates
        self.derivatives = {}                                      # Dict to store gradients
        self.accuracies = {"train": [], "test": []}                # Track train/test accuracy per epoch
        self.costs = []                                            # Track cost per epoch

        #  Reproducibility 
        if seed is not None:                                       # If a seed is provided
            np.random.seed(seed)                                   # Fix RNG for reproducibility


    #  Parameter initialization (He for ReLU/Leaky, Xavier otherwise) 
    def initialize_parameters(self):
        """
        Initialize weights and biases for all layers (1..L-1).
        Uses He init for ReLU/Leaky ReLU; Xavier for sigmoid/tanh.
        Shapes:
            w_l: (n_l, n_{l-1})
            b_l: (n_l, 1)
        """
        for i in range(1, self.L):                                           # Loop over layers 1..L-1
            fan_in = self.architecture[i - 1]                                # n_{l-1}
            fan_out = self.architecture[i]                                   # n_l

            # Choose scale based on activation (He vs Xavier)
            if self.activation in ("relu", "leaky_relu"):
                scale = np.sqrt(2.0 / fan_in)                                # He initialization
            else:
                scale = np.sqrt(1.0 / fan_in)                                # Xavier/Glorot (simple variant)

            # Initialize weights from N(0, scale^2)
            self.parameters[f"w{i}"] = np.random.randn(fan_out, fan_in) * scale
            # Initialize biases to zeros
            self.parameters[f"b{i}"] = np.zeros((fan_out, 1))


    #  Forward propagation 
    def _activate(self, z: np.ndarray) -> np.ndarray:
        """Apply the configured hidden activation to pre-activations z."""
        if self.activation == "relu":
            return relu(z)
        if self.activation == "sigmoid":
            return sigmoid(z)
        if self.activation == "tanh":
            return tanh_act(z)
        if self.activation == "leaky_relu":
            return leaky_relu(z)
        raise ValueError("Unknown activation")

    def forward(self, Xb: np.ndarray, Yb: Optional[np.ndarray] = None, training: bool = False, dropout_keep: float = 1.0):
        """
        Perform forward propagation on a batch.
        Inputs:
            Xb: mini-batch inputs (n_features, m_b)
            Yb: mini-batch labels (one-hot) (n_classes, m_b); can be None for inference
            training: whether we are in training mode (affects dropout)
            dropout_keep: probability to keep a unit (e.g., 0.9). Use 1.0 to disable dropout.
        Returns:
            output probabilities (n_classes, m_b), and cross-entropy cost if Yb is provided.
        Side effects:
            Stores intermediates in self.layers for backprop.
        """
        self.layers = {}                                                     # Reset layer cache for this pass
        self.layers["a0"] = Xb                                              # a0 is input (activation of layer 0)

        # Forward through hidden layers 1..L-2
        for l in range(1, self.L - 1):
            W = self.parameters[f"w{l}"]                                    # Weight matrix of layer l
            b = self.parameters[f"b{l}"]                                    # Bias vector of layer l
            Z = np.dot(W, self.layers[f"a{l-1}"]) + b                       # Pre-activation: z_l = W_l a_{l-1} + b_l
            A = self._activate(Z)                                           # Activation: a_l = g(z_l)

            # Optional dropout (training only, hidden layers only)
            if training and 0.0 < dropout_keep < 1.0:
                mask = (np.random.rand(*A.shape) < dropout_keep) / dropout_keep  # Inverted dropout mask
                A = A * mask                                                # Apply mask to activations
                self.layers[f"mask{l}"] = mask                              # Store mask for backprop

            # Cache for backprop and shape checks
            self.layers[f"z{l}"] = Z                                        # Store pre-activation
            self.layers[f"a{l}"] = A                                        # Store activation
            # (Optional) shape assertion:
            # assert A.shape == (self.architecture[l], Xb.shape[1])

        # Output layer (L-1): linear + softmax
        l_out = self.L - 1                                                  # Index of output layer
        W_out = self.parameters[f"w{l_out}"]                                 # Output weights (n_classes, last_hidden)
        b_out = self.parameters[f"b{l_out}"]                                 # Output biases (n_classes, 1)
        Z_out = np.dot(W_out, self.layers[f"a{l_out-1}"]) + b_out            # Pre-activation at output
        A_out = softmax(Z_out)                                               # Softmax to get probabilities

        # Cache output layer intermediates
        self.layers[f"z{l_out}"] = Z_out                                     # Store output pre-activation
        self.layers[f"a{l_out}"] = A_out                                     # Store output probabilities

        # If labels provided, compute cross-entropy cost (+ optional L2)
        cost = None
        if Yb is not None:
            eps = 1e-9                                                      # Small epsilon to avoid log(0)
            m_b = Xb.shape[1]                                               # Batch size
            cost = -np.sum(Yb * np.log(A_out + eps)) / m_b                  # Cross-entropy mean over batch

            # Add L2 penalty if enabled (lam stored on self or default 0.0)
            lam = getattr(self, "lam", 0.0)                                 # Regularization strength
            if lam > 0.0:
                l2_sum = 0.0                                                # Accumulator for ||W||^2 sum
                for i in range(1, self.L):                                  # Sum over all layers
                    l2_sum += np.sum(self.parameters[f"w{i}"] ** 2)         # Frobenius norm squared
                cost += (lam / (2.0 * m_b)) * l2_sum                        # Add weight decay term

        return A_out, cost                                                   # Return probs and (optional) cost


    #  Backpropagation 
    def backpropagate(self, Yb: np.ndarray, dropout_keep: float = 1.0):
        """
        Backward pass to compute gradients on the last forward() batch.
        Inputs:
            Yb: mini-batch one-hot labels (n_classes, m_b)
            dropout_keep: same keep prob used in forward (to backprop through masks)
        Returns:
            Dictionary of gradients {dW1, db1, ..., dW_{L-1}, db_{L-1}}
        Side effects:
            Stores gradients in self.derivatives.
        """
        grads = {}                                                           # Dict to store gradients
        m_b = Yb.shape[1]                                                    # Batch size
        lam = getattr(self, "lam", 0.0)                                      # L2 regularization strength

        # Output layer derivatives (softmax + cross-entropy): dZ = A - Y
        l_out = self.L - 1                                                   # Output layer index
        A_out = self.layers[f"a{l_out}"]                                     # Output probs
        dZ = A_out - Yb                                                      # Gradient at output pre-activation
        assert dZ.shape == (self.num_labels, m_b)                            # Shape check for safety

        # Gradients for W_out, b_out
        A_prev = self.layers[f"a{l_out-1}"]                                  # Activation from last hidden layer
        dW = (np.dot(dZ, A_prev.T) / m_b) + (lam / m_b) * self.parameters[f"w{l_out}"]  # Add L2 on weights
        db = np.sum(dZ, axis=1, keepdims=True) / m_b                         # Bias gradient is average over batch
        grads[f"dW{l_out}"] = dW                                             # Store dW_out
        grads[f"db{l_out}"] = db                                             # Store db_out

        # Gradient flowing to previous activation
        dA_prev = np.dot(self.parameters[f"w{l_out}"].T, dZ)                 # dA_{L-2} = W_{L-1}^T * dZ_{L-1}

        # Hidden layers: loop backward from L-2 down to 1
        for l in range(self.L - 2, 0, -1):
            Z = self.layers[f"z{l}"]                                         # Pre-activation at layer l
            dG = d_activation(self.activation, Z)                            # g'(Z_l)
            dZ = dA_prev * dG                                                # dZ_l = dA_l * g'(Z_l)

            # If dropout was used, backprop through mask (same layer l)
            if 0.0 < dropout_keep < 1.0:
                dZ *= self.layers.get(f"mask{l}", 1.0)                       # Apply stored mask

            A_prev = self.layers[f"a{l-1}"]                                  # Activation from previous layer
            dW = (np.dot(dZ, A_prev.T) / m_b) + (lam / m_b) * self.parameters[f"w{l}"]  # L2 term
            db = np.sum(dZ, axis=1, keepdims=True) / m_b                     # Bias gradient

            grads[f"dW{l}"] = dW                                             # Store dW_l
            grads[f"db{l}"] = db                                             # Store db_l

            if l > 1:                                                        # If there is a previous hidden/input layer
                dA_prev = np.dot(self.parameters[f"w{l}"].T, dZ)             # Backprop to dA_{l-1}

        self.derivatives = grads                                             # Cache for optimizer steps
        return grads                                                         # Return gradients


    #  Gradient clipping 
    def _clip_grads(self, grads, max_norm: float = 5.0):
        """
        Clip gradients to have global L2 norm <= max_norm (per tensor).
        Helps prevent exploding gradients.
        """
        for l in range(1, self.L):                                           # For each layer
            for p in ("W", "b"):                                             # For weights and biases
                g = grads[f"d{p}{l}"]                                        # Gradient tensor
                norm = np.linalg.norm(g)                                     # L2 norm
                if norm > max_norm and norm > 0:                             # If above threshold
                    grads[f"d{p}{l}"] = g * (max_norm / (norm + 1e-9))       # Scale down to max_norm


    #  Adam optimizer (state + step) 
    def _adam_init(self, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Initialize Adam optimizer state: moment and velocity tensors per parameter.
        """
        self.opt = {"t": 0, "beta1": beta1, "beta2": beta2, "eps": eps, "m": {}, "v": {}}
        for l in range(1, self.L):                                           # For each layer
            self.opt["m"][f"w{l}"] = np.zeros_like(self.parameters[f"w{l}"]) # First moment (mean) for weights
            self.opt["m"][f"b{l}"] = np.zeros_like(self.parameters[f"b{l}"]) # First moment for biases
            self.opt["v"][f"w{l}"] = np.zeros_like(self.parameters[f"w{l}"]) # Second moment (var) for weights
            self.opt["v"][f"b{l}"] = np.zeros_like(self.parameters[f"b{l}"]) # Second moment for biases

    def _adam_step(self, grads, lr: float):
        """
        Apply one Adam update step using gradients computed on the current batch.
        Implements bias-corrected moments for both weights and biases.
        Uses the same key names as produced by backpropagate(): dW* and db*.
        """
        # ---------------- Time step update ----------------
        self.opt["t"] += 1                             # Increment global Adam step counter
        t = self.opt["t"]                              # Current timestep (used for bias correction)

        # ---------------- Adam hyperparameters ----------------
        b1, b2, eps = self.opt["beta1"], self.opt["beta2"], self.opt["eps"]

        # ---------------- Loop over each layer ----------------
        for l in range(1, self.L):
            # =================================================
            # ---- Update WEIGHTS (w_l) ----
            # =================================================
            gW = grads[f"dW{l}"]                       # Gradient of weights for this layer
            # Update biased first moment estimate (mean of grads)
            self.opt["m"][f"w{l}"] = b1 * self.opt["m"][f"w{l}"] + (1 - b1) * gW
            # Update biased second moment estimate (uncentered variance of grads)
            self.opt["v"][f"w{l}"] = b2 * self.opt["v"][f"w{l}"] + (1 - b2) * (gW * gW)
            # Bias correction for first moment
            mW_hat = self.opt["m"][f"w{l}"] / (1 - b1 ** t)
            # Bias correction for second moment
            vW_hat = self.opt["v"][f"w{l}"] / (1 - b2 ** t)
            # Parameter update rule for weights
            self.parameters[f"w{l}"] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            # =================================================
            # ---- Update BIASES (b_l) ----
            # =================================================
            gb = grads[f"db{l}"]                       # Gradient of biases for this layer
            # Update biased first moment estimate
            self.opt["m"][f"b{l}"] = b1 * self.opt["m"][f"b{l}"] + (1 - b1) * gb
            # Update biased second moment estimate
            self.opt["v"][f"b{l}"] = b2 * self.opt["v"][f"b{l}"] + (1 - b2) * (gb * gb)
            # Bias correction for first moment
            mb_hat = self.opt["m"][f"b{l}"] / (1 - b1 ** t)
            # Bias correction for second moment
            vb_hat = self.opt["v"][f"b{l}"] / (1 - b2 ** t)
            # Parameter update rule for biases
            self.parameters[f"b{l}"] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


    #  Prediction & metrics 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run a forward pass (no dropout, no cost) and return predicted class ids.
        Input: X (n_features, m)
        Output: class ids (m,)
        """
        probs, _ = self.forward(X, None, training=False, dropout_keep=1.0)   # Inference mode
        return np.argmax(probs, axis=0)                                      # Pick argmax per column (example)

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute accuracy (%) on inputs X against one-hot labels Y.
        """
        preds = self.predict(X)                                              # Predicted class ids
        true = np.argmax(Y, axis=0)                                          # True class ids
        return np.mean(preds == true) * 100.0                                # Percentage correct


    #  Learning rate schedule (simple step decay) 
    def _lr_schedule(self, epoch: int, base_lr: float) -> float:
        """
        Example LR schedule: halve lr every 30 epochs.
        Replace with cosine/plateau schedule if needed.
        """
        if epoch > 0 and (epoch % 30 == 0):                                  # Every 30 epochs
            return base_lr * 0.5                                             # Halve learning rate
        return base_lr                                                       # Otherwise keep same


    #  Training loop 
    def fit(
        self,
        lr: float = 1e-3,                 # Base learning rate for Adam
        epochs: int = 30,                 # Number of passes over the training set
        batch_size: int = 128,            # Mini-batch size
        lam: float = 1e-4,                # L2 regularization strength (weight decay); 0 to disable
        dropout_keep: float = 1.0,        # Dropout keep probability (1.0 disables)
        grad_clip: float = 5.0,           # Clip gradients to this L2 norm; None to disable
        early_stopping_patience: int = 10 # Stop if no test accuracy improvement for N epochs
    ):
        """
        Train the network with mini-batch Adam.
        Tracks cost and train/test accuracy by epoch.
        Uses simple LR schedule and early stopping on test accuracy.
        """
        # Store regularization strength on self so forward/backprop can access it
        self.lam = lam

        # Initialize parameters and optimizer state
        self.initialize_parameters()
        self._adam_init()

        # Reset history trackers
        self.costs = []
        self.accuracies = {"train": [], "test": []}

        # Early stopping bookkeeping
        best = {"acc": -np.inf, "params": None, "wait": 0}                   # Track best test acc and patience

        # Training loop over epochs
        for epoch in tqdm(range(epochs), desc="Training"):
            # Optionally update lr via schedule (epoch-wise)
            cur_lr = self._lr_schedule(epoch, lr)

            # Loop over mini-batches
            last_cost = None                                                 # Will store the last batch cost
            for Xb, Yb in iterate_minibatches(self.X, self.y, batch_size, shuffle=True):
                # Forward pass on the batch (training=True for dropout)
                probs, cost = self.forward(Xb, Yb, training=True, dropout_keep=dropout_keep)
                last_cost = cost                                             # Remember last computed cost for logging

                # Backprop on the batch to get gradients
                grads = self.backpropagate(Yb, dropout_keep=dropout_keep)

                # Optional: gradient clipping to avoid spikes
                if grad_clip is not None:
                    self._clip_grads(grads, max_norm=grad_clip)

                # Optimizer step (Adam) using current gradients
                self._adam_step(grads, lr=cur_lr)

            # After finishing all mini-batches: log epoch metrics
            self.costs.append(float(last_cost if last_cost is not None else np.nan))  # Store last batch cost

            # Compute accuracy on full train and test sets (no dropout)
            train_acc = self.accuracy(self.X, self.y)
            test_acc = self.accuracy(self.X_test, self.y_test)
            self.accuracies["train"].append(train_acc)
            self.accuracies["test"].append(test_acc)

            # Print periodic progress (every 5 epochs)
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Cost {self.costs[-1]:.4f} | "
                      f"Train {train_acc:.2f}% | Test {test_acc:.2f}% | LR {cur_lr:.2e}")

            # Early stopping logic: monitor test accuracy
            if test_acc > best["acc"] + 1e-8:                                 # Significant improvement
                best["acc"] = test_acc                                        # Update best score
                # Deep copy parameters (weights/biases) for restoration
                best["params"] = {k: v.copy() for k, v in self.parameters.items()}
                best["wait"] = 0                                              # Reset patience counter
            else:
                best["wait"] += 1                                             # No improvement: increase wait
                if best["wait"] >= early_stopping_patience:                   # Patience exceeded
                    print("Early stopping triggered. Restoring best parameters.")
                    # Restore best parameters
                    if best["params"] is not None:
                        self.parameters = {k: v.copy() for k, v in best["params"].items()}
                    break                                                     # Exit training loop


    #  Plotting helpers 
    def plot_cost(self, lr_used: float):
        """
        Plot the training cost (last batch per epoch) over epochs.
        """
        plt.figure(figsize=(8, 4))                                           # Create a horizontal plot
        plt.plot(np.arange(len(self.costs)), self.costs, lw=1)               # Plot cost curve
        plt.title(f"Training Cost\nLearning rate (base): {lr_used}")         # Title with LR info
        plt.xlabel("Epoch")                                                  # X label
        plt.ylabel("Cost (Cross-Entropy + L2)")                              # Y label
        plt.grid(True, alpha=0.3)                                            # Light grid
        plt.tight_layout()                                                   # Nicely fit
        plt.show()                                                           # Display

    def plot_accuracies(self, lr_used: float):
        """
        Plot training and test accuracies over epochs.
        """
        acc = self.accuracies                                                # Shortcut
        plt.figure(figsize=(8, 4))                                           # Create a horizontal plot
        plt.plot(acc["train"], label="Train", linewidth=1.5)                 # Train accuracy curve
        plt.plot(acc["test"], label="Test", linewidth=1.5)                   # Test accuracy curve
        plt.legend(loc="lower right")                                        # Legend placement
        plt.title(f"Accuracy Curves\nLearning rate (base): {lr_used}")       # Title with LR info
        plt.xlabel("Epoch")                                                  # X label
        plt.ylabel("Accuracy (%)")                                           # Y label
        # Annotate final values
        if acc["train"]:
            plt.annotate(f"{acc['train'][-1]:.2f}%", (len(acc['train'])-1, acc['train'][-1]))
        if acc["test"]:
            plt.annotate(f"{acc['test'][-1]:.2f}%", (len(acc['test'])-1, acc['test'][-1]))
        plt.grid(True, alpha=0.3)                                            # Light grid
        plt.tight_layout()                                                   # Fit
        plt.show()                                                           # Display



# Example usage (MNIST)

if __name__ == "__main__":
    # NOTE: fetch_openml will download MNIST the first time (cached afterwards).
    from sklearn.datasets import fetch_openml           # Import inside main to avoid hard dependency on import

    print("Loading MNIST dataset...")                   # Status print

    # Fetch the MNIST dataset by name; cached=True stores it locally for reuse
    mnist = fetch_openml(name="mnist_784", version=1, as_frame=False, parser="auto", cache=True)

    # Extract data and labels
    X_all = mnist.data.T.astype(np.float32)             # Transpose to (features, m); cast to float32
    y_all = mnist.target.astype(int)                    # Convert labels from strings to ints

    # Normalize pixel values to [0,1] at source (still normalize per-feature inside class)
    # X_all /= 255.0

    # Split into train/test using the classic 60k/10k split (MNIST ordering)
    train_size = 60000                                  # First 60k as training
    X_train = X_all[:, :train_size]                     # Training inputs
    X_test  = X_all[:, train_size:]                     # Test inputs
    y_train = one_hot_encode(y_all[:train_size], 10)    # One-hot train labels -> (10, 60000)
    y_test  = one_hot_encode(y_all[train_size:], 10)    # One-hot test labels  -> (10, 10000)

    # Create the neural network instance
    nn = NeuralNetwork(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        activation="relu",                              # Hidden activation: "relu", "tanh", "sigmoid", "leaky_relu"
        num_labels=10,                                  # 10 classes (digits 0..9)
        architecture=[512, 256, 128],                   # Three hidden layers
        seed=42                                         # RNG seed for reproducibility
    )

    # Train the model with recommended defaults
    nn.fit(
        lr=1e-3,                      # Adam base learning rate
        epochs=30,                    # 20–30 is usually enough with Adam on MNIST MLP
        batch_size=128,               # Standard mini-batch size
        lam=1e-4,                     # L2 regularization (set 0 to disable)
        dropout_keep=1.0,             # 1.0 disables dropout; try 0.9 for light regularization
        grad_clip=5.0,                # Clip gradients to avoid spikes
        early_stopping_patience=8     # Stop early if test acc plateaus
    )

    # Plot the learning curves
    nn.plot_cost(lr_used=1e-3)        # Cost over epochs
    nn.plot_accuracies(lr_used=1e-3)  # Accuracy curves over epochs