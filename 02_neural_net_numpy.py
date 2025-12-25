"""
02_neural_net_numpy.py - Neural Network from Scratch

This implements a complete neural network for MNIST digit classification
using only NumPy. No PyTorch, no TensorFlow - just matrix math!

Architecture:
    Input (784) → Hidden (128, ReLU) → Output (10, Softmax)

What you'll learn:
    1. How to load and preprocess MNIST
    2. Weight initialization (Xavier)
    3. Forward propagation
    4. Backpropagation (computing gradients)
    5. Gradient descent (updating weights)
    6. Mini-batch training
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import struct
from urllib.request import urlretrieve

np.random.seed(42)

# =============================================================================
# PART 1: MNIST DATA LOADING
# =============================================================================

def download_mnist():
    """Download MNIST dataset if not already present."""
    # Using a reliable mirror since original LeCun site can be unreliable
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urlretrieve(base_url + filename, filepath)

    return data_dir

def load_mnist_images(filepath):
    """Load MNIST images from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        # Read header: magic number, num images, rows, cols
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read image data as unsigned bytes
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, 784) and normalize to [0, 1]
        images = images.reshape(num, rows * cols).astype(np.float32) / 255.0
    return images

def load_mnist_labels(filepath):
    """Load MNIST labels from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        # Read header: magic number, num labels
        magic, num = struct.unpack(">II", f.read(8))
        # Read labels as unsigned bytes
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist():
    """Load full MNIST dataset."""
    data_dir = download_mnist()

    X_train = load_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Image shape: {X_train.shape[1]} pixels (28x28 flattened)")

    return X_train, y_train, X_test, y_test

# =============================================================================
# PART 2: NEURAL NETWORK CLASS
# =============================================================================

class NeuralNetwork:
    """
    A simple 2-layer neural network.

    Architecture:
        Input (784) → Hidden (128, ReLU) → Output (10, Softmax)

    The network learns to classify 28x28 grayscale images of digits (0-9).
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize the network with random weights.

        We use Xavier initialization: weights ~ N(0, sqrt(2 / (fan_in + fan_out)))
        This helps prevent vanishing/exploding gradients at the start.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layer 1: Input → Hidden
        # Xavier initialization: scale by sqrt(2 / (input + output))
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros(hidden_size)

        # Layer 2: Hidden → Output
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros(output_size)

        print(f"Network initialized:")
        print(f"  W1: {self.W1.shape}  (input → hidden)")
        print(f"  b1: {self.b1.shape}")
        print(f"  W2: {self.W2.shape}  (hidden → output)")
        print(f"  b2: {self.b2.shape}")
        print(f"  Total parameters: {self.W1.size + self.b1.size + self.W2.size + self.b2.size:,}")

    # -------------------------------------------------------------------------
    # ACTIVATION FUNCTIONS
    # -------------------------------------------------------------------------

    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Softmax activation: converts logits to probabilities.

        softmax(z_i) = exp(z_i) / sum(exp(z_j))

        We subtract max(z) for numerical stability (prevents overflow).
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # -------------------------------------------------------------------------
    # FORWARD PROPAGATION
    # -------------------------------------------------------------------------

    def forward(self, X):
        """
        Forward pass: compute predictions from input.

        X: (batch_size, 784) - input images

        Returns:
            probabilities: (batch_size, 10) - probability of each digit

        Also stores intermediate values needed for backpropagation.
        """
        # Store input for backprop
        self.X = X

        # Layer 1: Linear transformation
        # z1 = np.dot(X, W1) + b1
        # Shape: np.dot((batch, 784), (784, 128)) + (128,) = (batch, 128)
        self.z1 = np.dot(X, self.W1) + self.b1

        # Layer 1: Activation (ReLU)
        # a1 = relu(z1)
        self.a1 = self.relu(self.z1)

        # Layer 2: Linear transformation
        # z2 = np.dot(a1, W2) + b2
        # Shape: np.dot((batch, 128), (128, 10)) + (10,) = (batch, 10)
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        # Layer 2: Activation (Softmax)
        # a2 = softmax(z2)
        self.a2 = self.softmax(self.z2)

        return self.a2

    # -------------------------------------------------------------------------
    # LOSS FUNCTION
    # -------------------------------------------------------------------------

    def cross_entropy_loss(self, predictions, targets):
        """
        Cross-entropy loss for classification.

        Loss = -mean(log(predicted_probability_of_true_class))

        predictions: (batch_size, 10) - probabilities from softmax
        targets: (batch_size,) - true class labels (0-9)

        Returns: scalar loss value
        """
        batch_size = predictions.shape[0]
        # Get probability assigned to the correct class for each sample
        correct_probs = predictions[np.arange(batch_size), targets]
        # Cross-entropy: -log(correct_probability)
        # Add small epsilon to prevent log(0)
        loss = -np.mean(np.log(correct_probs + 1e-10))
        return loss

    # -------------------------------------------------------------------------
    # BACKPROPAGATION - The Heart of Learning!
    # -------------------------------------------------------------------------

    def backward(self, y):
        """
        Backpropagation: compute gradients of loss with respect to all parameters.

        This is where the magic happens! We use the chain rule to compute
        how much each weight contributed to the error.

        y: (batch_size,) - true class labels

        The key insight:
        - We propagate the error backward through the network
        - At each layer, we compute: how much did this weight affect the error?
        - We use these gradients to update weights in the direction that reduces error
        """
        batch_size = y.shape[0]

        # =====================================================================
        # Step 1: Output layer gradient (Softmax + Cross-Entropy)
        # =====================================================================
        #
        # For softmax + cross-entropy, the gradient has a beautiful simple form:
        # dL/dz2 = predictions - one_hot(targets)
        #
        # If true label is 3 and we predicted [0.1, 0.1, 0.1, 0.6, 0.1, ...]:
        # Gradient = [0.1, 0.1, 0.1, 0.6-1, 0.1, ...] = [0.1, 0.1, 0.1, -0.4, 0.1, ...]
        #
        # This means: decrease probability of wrong classes, increase probability of class 3

        # Create one-hot encoding of labels
        one_hot_y = np.zeros_like(self.a2)
        one_hot_y[np.arange(batch_size), y] = 1

        # Gradient of loss with respect to z2 (output layer pre-activation)
        # dL/dz2 = a2 - y (predictions minus true one-hot labels)
        dz2 = (self.a2 - one_hot_y) / batch_size  # Divide by batch_size for mean

        # =====================================================================
        # Step 2: Gradients for W2 and b2
        # =====================================================================
        #
        # z2 = np.dot(a1, W2) + b2
        #
        # dL/dW2 = np.dot(a1.T, dz2)  (chain rule: upstream gradient times local gradient)
        # dL/db2 = sum(dz2)           (bias gradient is sum over batch)

        self.dW2 = np.dot(self.a1.T, dz2)   # Shape: (128, 10)
        self.db2 = np.sum(dz2, axis=0)       # Shape: (10,)

        # =====================================================================
        # Step 3: Backpropagate through Layer 2
        # =====================================================================
        #
        # We need dL/da1 to continue backpropagating
        # dL/da1 = np.dot(dz2, W2.T)

        da1 = np.dot(dz2, self.W2.T)  # Shape: (batch, 128)

        # =====================================================================
        # Step 4: Backpropagate through ReLU
        # =====================================================================
        #
        # ReLU(z) = max(0, z)
        # ReLU'(z) = 1 if z > 0, else 0
        #
        # dL/dz1 = dL/da1 * relu'(z1)
        #
        # This is why ReLU can cause "dead neurons" - if z1 < 0, gradient is 0

        dz1 = da1 * self.relu_derivative(self.z1)  # Shape: (batch, 128)

        # =====================================================================
        # Step 5: Gradients for W1 and b1
        # =====================================================================
        #
        # z1 = np.dot(X, W1) + b1
        #
        # dL/dW1 = np.dot(X.T, dz1)
        # dL/db1 = sum(dz1)

        self.dW1 = np.dot(self.X.T, dz1)    # Shape: (784, 128)
        self.db1 = np.sum(dz1, axis=0)       # Shape: (128,)

    # -------------------------------------------------------------------------
    # GRADIENT DESCENT - Update Weights
    # -------------------------------------------------------------------------

    def update_weights(self, learning_rate):
        """
        Gradient descent: update weights in the direction that reduces loss.

        new_weight = old_weight - learning_rate * gradient

        The learning rate controls how big of a step we take.
        - Too big: we might overshoot the minimum
        - Too small: training takes forever
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    def predict(self, X):
        """Get predicted class labels."""
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y):
        """Calculate classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# =============================================================================
# PART 3: TRAINING
# =============================================================================

def train(model, X_train, y_train, X_test, y_test,
          epochs=10, batch_size=64, learning_rate=0.1):
    """
    Train the neural network using mini-batch gradient descent.

    Mini-batches are better than:
    - Full batch: faster updates, can escape local minima
    - Single sample: too noisy, doesn't utilize vectorization

    Args:
        model: NeuralNetwork instance
        X_train, y_train: training data
        X_test, y_test: test data for evaluation
        epochs: number of complete passes through training data
        batch_size: number of samples per gradient update
        learning_rate: step size for gradient descent
    """
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    history = {'loss': [], 'train_acc': [], 'test_acc': []}

    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Batches per epoch: {n_batches}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 60)

    for epoch in range(epochs):
        # Shuffle training data at the start of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0

        # Process mini-batches
        for batch in range(n_batches):
            # Get batch data
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass: compute predictions
            predictions = model.forward(X_batch)

            # Compute loss
            loss = model.cross_entropy_loss(predictions, y_batch)
            epoch_loss += loss

            # Backward pass: compute gradients
            model.backward(y_batch)

            # Update weights
            model.update_weights(learning_rate)

        # Calculate metrics
        avg_loss = epoch_loss / n_batches
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)

        history['loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

    return history

# =============================================================================
# PART 4: VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """Plot loss and accuracy over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(history['test_acc'], 'r-', label='Test', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    plt.show()
    print("Saved training_history.png")

def visualize_predictions(model, X_test, y_test, n_samples=10):
    """Show some predictions with the actual images."""
    # Get random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # Get image and reshape to 28x28
        image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]

        # Get prediction
        probs = model.forward(X_test[idx:idx+1])
        pred_label = np.argmax(probs)
        confidence = probs[0, pred_label]

        # Plot
        axes[i].imshow(image, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'Pred: {pred_label} ({confidence:.1%})\nTrue: {true_label}',
                         color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=100)
    plt.show()
    print("Saved predictions.png")

def visualize_weights(model):
    """Visualize what the first layer has learned."""
    # Each column of W1 is the weights for one hidden neuron
    # Reshape to 28x28 to see what patterns each neuron detects

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(32):  # Show first 32 hidden neurons
        weights = model.W1[:, i].reshape(28, 28)
        axes[i].imshow(weights, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[i].axis('off')

    plt.suptitle('First 32 Hidden Neuron Weights\n(What patterns does each neuron detect?)')
    plt.tight_layout()
    plt.savefig('weights.png', dpi=100)
    plt.show()
    print("Saved weights.png")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MNIST Neural Network from Scratch")
    print("=" * 60)

    # Load data
    print("\n1. Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Create network
    print("\n2. Creating neural network...")
    model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

    # Check initial accuracy (should be ~10% - random guessing)
    initial_acc = model.accuracy(X_test, y_test)
    print(f"\nInitial accuracy (random weights): {initial_acc:.2%}")

    # Train
    print("\n3. Training...")
    history = train(
        model, X_train, y_train, X_test, y_test,
        epochs=10,
        batch_size=64,
        learning_rate=0.1
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_acc = model.accuracy(X_test, y_test)
    print(f"Test Accuracy: {final_acc:.2%}")

    # Visualizations
    print("\n4. Generating visualizations...")
    plot_training_history(history)
    visualize_predictions(model, X_test, y_test)
    visualize_weights(model)

    print("\n" + "=" * 60)
    print("SUMMARY - What the Network Learned")
    print("=" * 60)
    print(f"""
The network went from {initial_acc:.1%} (random guessing) to {final_acc:.1%} accuracy!

What happened during training:
1. FORWARD PASS: Images flow through layers, producing predictions
2. LOSS: Cross-entropy measures how wrong predictions are
3. BACKWARD PASS: Gradients show which weights caused errors
4. UPDATE: Weights adjust to reduce future errors

Each hidden neuron learned to detect different patterns:
- Some detect edges at specific angles
- Some detect curves or corners
- Together, they form a "vocabulary" for recognizing digits

Check the saved images:
- training_history.png: Loss decreasing, accuracy increasing
- predictions.png: Sample predictions on test images
- weights.png: Patterns each hidden neuron learned to detect
""")
