"""
Generate all images needed for the documentation.
This includes hidden layer activations, step-by-step forward pass, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path to import our neural network
sys.path.insert(0, '.')
from pathlib import Path
import gzip
import struct

np.random.seed(42)

# =============================================================================
# Load MNIST (simplified version)
# =============================================================================

def load_mnist_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols).astype(np.float32) / 255.0
    return images

def load_mnist_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# =============================================================================
# Neural Network (copy of core functionality)
# =============================================================================

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros(output_size)

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def forward_detailed(self, X):
        """Return all intermediate values for visualization."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)
        return {
            'input': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }

    def backward(self, y):
        batch_size = y.shape[0]
        one_hot_y = np.zeros_like(self.a2)
        one_hot_y[np.arange(batch_size), y] = 1
        dz2 = (self.a2 - one_hot_y) / batch_size
        self.dW2 = np.dot(self.a1.T, dz2)
        self.db2 = np.sum(dz2, axis=0)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0).astype(float)
        self.dW1 = np.dot(self.X.T, dz1)
        self.db1 = np.sum(dz1, axis=0)

    def update_weights(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

# =============================================================================
# IMAGE GENERATION FUNCTIONS
# =============================================================================

def save_fig(fig, name):
    """Save figure to docs/images/"""
    path = Path("docs/images") / name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")

# -----------------------------------------------------------------------------
# 1. Network Architecture Diagram (simple visual)
# -----------------------------------------------------------------------------

def generate_architecture_diagram():
    """Create a visual representation of the network architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors
    input_color = '#3498db'
    hidden_color = '#e74c3c'
    output_color = '#2ecc71'

    # Draw input layer (show subset of 784 neurons)
    input_x = 1.5
    for i, y in enumerate(np.linspace(1.5, 6.5, 8)):
        circle = plt.Circle((input_x, y), 0.2, color=input_color, ec='black', lw=1)
        ax.add_patch(circle)
    ax.text(input_x, 7.2, 'Input Layer\n784 neurons', ha='center', fontsize=11, fontweight='bold')
    ax.text(input_x, 0.8, '(28×28 image\nflattened)', ha='center', fontsize=9, style='italic')

    # Draw hidden layer
    hidden_x = 5
    hidden_ys = np.linspace(1.5, 6.5, 6)
    for y in hidden_ys:
        circle = plt.Circle((hidden_x, y), 0.2, color=hidden_color, ec='black', lw=1)
        ax.add_patch(circle)
    ax.text(hidden_x, 7.2, 'Hidden Layer\n128 neurons', ha='center', fontsize=11, fontweight='bold')
    ax.text(hidden_x, 0.8, 'ReLU activation', ha='center', fontsize=9, style='italic')

    # Draw output layer
    output_x = 8.5
    output_ys = np.linspace(2, 6, 5)
    for y in output_ys:
        circle = plt.Circle((output_x, y), 0.2, color=output_color, ec='black', lw=1)
        ax.add_patch(circle)
    ax.text(output_x, 7.2, 'Output Layer\n10 neurons', ha='center', fontsize=11, fontweight='bold')
    ax.text(output_x, 0.8, 'Softmax activation\n(probabilities)', ha='center', fontsize=9, style='italic')

    # Draw connections (subset)
    for iy in np.linspace(1.5, 6.5, 8):
        for hy in hidden_ys[::2]:
            ax.plot([input_x + 0.2, hidden_x - 0.2], [iy, hy], 'gray', alpha=0.2, lw=0.5)

    for hy in hidden_ys:
        for oy in output_ys:
            ax.plot([hidden_x + 0.2, output_x - 0.2], [hy, oy], 'gray', alpha=0.3, lw=0.5)

    # Add weight labels
    ax.annotate('W1\n(784×128)', xy=(3.2, 4), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.annotate('W2\n(128×10)', xy=(6.7, 4), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add dots for "more neurons"
    ax.text(input_x, 4, '⋮', fontsize=20, ha='center', va='center')
    ax.text(hidden_x, 4, '⋮', fontsize=20, ha='center', va='center')

    ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
    save_fig(fig, 'architecture.png')

# -----------------------------------------------------------------------------
# 2. Forward Pass Step by Step
# -----------------------------------------------------------------------------

def generate_forward_pass_visualization(model, X_sample, y_sample):
    """Visualize each step of the forward pass."""
    details = model.forward_detailed(X_sample.reshape(1, -1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Input image
    ax = axes[0, 0]
    ax.imshow(X_sample.reshape(28, 28), cmap='gray')
    ax.set_title(f'Step 1: Input Image\n(True label: {y_sample})', fontsize=11, fontweight='bold')
    ax.axis('off')

    # 2. z1 = np.dot(X, W1) + b1 (pre-activation)
    ax = axes[0, 1]
    z1_reshaped = details['z1'].reshape(8, 16)  # Reshape 128 to 8x16 for visualization
    im = ax.imshow(z1_reshaped, cmap='RdBu', aspect='auto')
    ax.set_title('Step 2: Linear Transform (z1)\nz1 = np.dot(X, W1) + b1', fontsize=11, fontweight='bold')
    ax.set_xlabel('Hidden neurons (128 total)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. a1 = relu(z1) (after activation)
    ax = axes[0, 2]
    a1_reshaped = details['a1'].reshape(8, 16)
    im = ax.imshow(a1_reshaped, cmap='Reds', aspect='auto')
    ax.set_title('Step 3: ReLU Activation (a1)\na1 = max(0, z1)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Hidden neurons (128 total)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 4. z2 = np.dot(a1, W2) + b2
    ax = axes[1, 0]
    z2 = details['z2'].flatten()
    colors = ['#e74c3c' if v == max(z2) else '#3498db' for v in z2]
    bars = ax.bar(range(10), z2, color=colors)
    ax.set_title('Step 4: Output Linear (z2)\nz2 = np.dot(a1, W2) + b2', fontsize=11, fontweight='bold')
    ax.set_xlabel('Output neurons (digits 0-9)')
    ax.set_ylabel('Raw logit value')
    ax.set_xticks(range(10))

    # 5. a2 = softmax(z2)
    ax = axes[1, 1]
    a2 = details['a2'].flatten()
    colors = ['#2ecc71' if v == max(a2) else '#95a5a6' for v in a2]
    bars = ax.bar(range(10), a2, color=colors)
    ax.set_title('Step 5: Softmax (a2)\nConvert to probabilities', fontsize=11, fontweight='bold')
    ax.set_xlabel('Digit class')
    ax.set_ylabel('Probability')
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)

    # 6. Prediction
    ax = axes[1, 2]
    pred = np.argmax(a2)
    conf = a2[pred]
    ax.text(0.5, 0.6, f'{pred}', fontsize=120, ha='center', va='center',
            fontweight='bold', color='#2ecc71' if pred == y_sample else '#e74c3c')
    ax.text(0.5, 0.15, f'Confidence: {conf:.1%}', fontsize=14, ha='center')
    ax.text(0.5, 0.05, f'True: {y_sample}  |  {"✓ Correct!" if pred == y_sample else "✗ Wrong"}',
            fontsize=12, ha='center', color='#2ecc71' if pred == y_sample else '#e74c3c')
    ax.set_title('Step 6: Prediction\nargmax(softmax)', fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_fig(fig, 'forward_pass_steps.png')

# -----------------------------------------------------------------------------
# 3. Hidden Layer Activations
# -----------------------------------------------------------------------------

def generate_hidden_activations(model, X_test, y_test):
    """Show how hidden neurons activate for different digits."""

    # Get one example of each digit
    digit_examples = {}
    for i in range(len(y_test)):
        digit = y_test[i]
        if digit not in digit_examples:
            digit_examples[digit] = X_test[i]
        if len(digit_examples) == 10:
            break

    fig, axes = plt.subplots(3, 10, figsize=(16, 6))

    for digit in range(10):
        X = digit_examples[digit]
        details = model.forward_detailed(X.reshape(1, -1))

        # Row 1: Input image
        axes[0, digit].imshow(X.reshape(28, 28), cmap='gray')
        axes[0, digit].axis('off')
        if digit == 0:
            axes[0, digit].set_ylabel('Input', fontsize=10, fontweight='bold')
        axes[0, digit].set_title(f'Digit {digit}', fontsize=10)

        # Row 2: Hidden activations (a1) - show as heatmap
        a1 = details['a1'].reshape(8, 16)
        axes[1, digit].imshow(a1, cmap='hot', aspect='auto')
        axes[1, digit].axis('off')
        if digit == 0:
            axes[1, digit].set_ylabel('Hidden\nActivations', fontsize=10, fontweight='bold')

        # Row 3: Output probabilities
        a2 = details['a2'].flatten()
        colors = ['#2ecc71' if i == digit else '#bdc3c7' for i in range(10)]
        axes[2, digit].bar(range(10), a2, color=colors, width=0.8)
        axes[2, digit].set_ylim(0, 1)
        axes[2, digit].set_xticks([])
        if digit == 0:
            axes[2, digit].set_ylabel('Output\nProbs', fontsize=10, fontweight='bold')

    plt.suptitle('Hidden Layer Activations for Each Digit\n(Different digits activate different hidden neurons)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'hidden_activations_by_digit.png')

# -----------------------------------------------------------------------------
# 4. Single Neuron Activation Patterns
# -----------------------------------------------------------------------------

def generate_neuron_patterns(model, X_test, y_test):
    """Show what patterns activate specific hidden neurons."""

    # Find interesting neurons (ones with high variance in activation)
    activations = []
    for i in range(min(1000, len(X_test))):
        details = model.forward_detailed(X_test[i:i+1])
        activations.append(details['a1'].flatten())
    activations = np.array(activations)

    # Get neurons with high variance (they're "selective")
    neuron_variance = np.var(activations, axis=0)
    interesting_neurons = np.argsort(neuron_variance)[-16:]  # Top 16 most selective

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for idx, neuron_idx in enumerate(interesting_neurons):
        ax = axes[idx]

        # Get the weights for this neuron (what it looks for)
        weights = model.W1[:, neuron_idx].reshape(28, 28)

        # Also find the image that activates this neuron the most
        max_activation_idx = np.argmax(activations[:, neuron_idx])

        # Create a 2-panel view
        ax.imshow(weights, cmap='RdBu', vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
        ax.set_title(f'Neuron {neuron_idx}\n(weights)', fontsize=9)
        ax.axis('off')

    plt.suptitle('What Each Hidden Neuron "Looks For"\n(Red = positive weight, Blue = negative weight)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'neuron_weight_patterns.png')

# -----------------------------------------------------------------------------
# 5. ReLU Visualization
# -----------------------------------------------------------------------------

def generate_relu_visualization():
    """Visualize ReLU activation function."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = np.linspace(-3, 3, 100)

    # ReLU function
    ax = axes[0]
    relu = np.maximum(0, x)
    ax.plot(x, relu, 'b-', linewidth=3, label='ReLU(x) = max(0, x)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(x[x >= 0], 0, relu[x >= 0], alpha=0.3)
    ax.set_xlabel('Input (z)', fontsize=11)
    ax.set_ylabel('Output', fontsize=11)
    ax.set_title('ReLU Activation Function', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3)

    # ReLU derivative
    ax = axes[1]
    relu_deriv = (x > 0).astype(float)
    ax.plot(x, relu_deriv, 'r-', linewidth=3, label="ReLU'(x)")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Input (z)', fontsize=11)
    ax.set_ylabel('Gradient', fontsize=11)
    ax.set_title('ReLU Derivative (for backprop)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])

    # Before/After ReLU on real data
    ax = axes[2]
    z1_example = np.random.randn(50) * 2  # Simulated pre-activation values
    a1_example = np.maximum(0, z1_example)

    x_pos = np.arange(50)
    ax.bar(x_pos - 0.2, z1_example, width=0.4, label='Before ReLU (z1)', alpha=0.7, color='blue')
    ax.bar(x_pos + 0.2, a1_example, width=0.4, label='After ReLU (a1)', alpha=0.7, color='red')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Neuron index', fontsize=11)
    ax.set_ylabel('Activation value', fontsize=11)
    ax.set_title('ReLU Effect on Hidden Layer', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(-1, 50)

    plt.tight_layout()
    save_fig(fig, 'relu_visualization.png')

# -----------------------------------------------------------------------------
# 6. Softmax Visualization
# -----------------------------------------------------------------------------

def generate_softmax_visualization():
    """Visualize softmax transformation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Example logits
    logits = np.array([2.0, 1.0, 0.1, -0.5, 0.3, -1.0, 0.8, 1.5, -0.2, 0.6])

    # Softmax calculation
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # Plot 1: Raw logits
    ax = axes[0]
    colors = ['#e74c3c' if v == max(logits) else '#3498db' for v in logits]
    ax.bar(range(10), logits, color=colors)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Digit class', fontsize=11)
    ax.set_ylabel('Raw logit value', fontsize=11)
    ax.set_title('Before Softmax (z2)\nRaw output values', fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))

    # Plot 2: Exp transformation
    ax = axes[1]
    ax.bar(range(10), exp_logits, color='#9b59b6')
    ax.set_xlabel('Digit class', fontsize=11)
    ax.set_ylabel('exp(logit)', fontsize=11)
    ax.set_title('Step 1: Exponentiate\nexp(z - max(z))', fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))

    # Plot 3: Probabilities
    ax = axes[2]
    colors = ['#2ecc71' if v == max(probs) else '#95a5a6' for v in probs]
    ax.bar(range(10), probs, color=colors)
    ax.set_xlabel('Digit class', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('After Softmax (a2)\nProbabilities sum to 1', fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)

    # Add sum annotation
    ax.annotate(f'Sum = {probs.sum():.3f}', xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_fig(fig, 'softmax_visualization.png')

# -----------------------------------------------------------------------------
# 7. Cross-Entropy Loss Visualization
# -----------------------------------------------------------------------------

def generate_loss_visualization():
    """Visualize cross-entropy loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: -log(p) curve
    ax = axes[0]
    p = np.linspace(0.01, 1, 100)
    loss = -np.log(p)
    ax.plot(p, loss, 'b-', linewidth=3)
    ax.fill_between(p, 0, loss, alpha=0.2)
    ax.set_xlabel('Predicted probability for correct class', fontsize=11)
    ax.set_ylabel('Loss = -log(p)', fontsize=11)
    ax.set_title('Cross-Entropy Loss Function', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)

    # Add annotations
    ax.annotate('High confidence,\nlow loss', xy=(0.9, -np.log(0.9)),
                xytext=(0.7, 1.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green')
    ax.annotate('Low confidence,\nhigh loss', xy=(0.1, -np.log(0.1)),
                xytext=(0.3, 3.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red')

    # Plot 2: Example predictions and their losses
    ax = axes[1]
    examples = [
        ('Perfect', [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
        ('Good', [0.05, 0.05, 0.05, 0.7, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01], 3),
        ('Uncertain', [0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05], 3),
        ('Wrong', [0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02], 3),
    ]

    names = []
    losses = []
    colors = []
    for name, probs, true_class in examples:
        loss = -np.log(probs[true_class] + 1e-10)
        names.append(f'{name}\n(p={probs[true_class]:.2f})')
        losses.append(loss)
        colors.append('#2ecc71' if loss < 0.5 else '#f39c12' if loss < 1.5 else '#e74c3c')

    ax.barh(names, losses, color=colors)
    ax.set_xlabel('Loss value', fontsize=11)
    ax.set_title('Loss for Different Prediction Qualities\n(True class = 3)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 4)

    for i, (loss, name) in enumerate(zip(losses, names)):
        ax.text(loss + 0.1, i, f'{loss:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    save_fig(fig, 'cross_entropy_loss.png')

# -----------------------------------------------------------------------------
# 8. Backpropagation Flow
# -----------------------------------------------------------------------------

def generate_backprop_visualization():
    """Visualize gradient flow in backpropagation."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Forward pass (top)
    ax.text(7, 9.5, 'Forward Pass →', fontsize=14, ha='center', fontweight='bold', color='#3498db')

    # Boxes for each step
    boxes = [
        (1, 6, 'X\n(input)', '#3498db'),
        (3.5, 6, 'z1 = dot(X,W1)+b1', '#9b59b6'),
        (6, 6, 'a1 = ReLU(z1)', '#e74c3c'),
        (8.5, 6, 'z2 = dot(a1,W2)+b2', '#9b59b6'),
        (11, 6, 'ŷ = softmax(z2)', '#2ecc71'),
        (13, 6, 'Loss', '#f39c12'),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.8, y-0.5), 1.6, 1, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Forward arrows
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(boxes[i+1][0]-0.8, 6), xytext=(boxes[i][0]+0.8, 6),
                   arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))

    # Backward pass (bottom)
    ax.text(7, 1, '← Backward Pass (Gradients)', fontsize=14, ha='center', fontweight='bold', color='#e74c3c')

    # Gradient boxes
    grad_boxes = [
        (3.5, 3, 'dW1, db1', '#ff9ff3'),
        (6, 3, 'da1 × ReLU\'', '#ff9ff3'),
        (8.5, 3, 'dW2, db2', '#ff9ff3'),
        (11, 3, 'dz2 = ŷ - y', '#ff9ff3'),
        (13, 3, '∂L/∂ŷ', '#ff9ff3'),
    ]

    for x, y, text, color in grad_boxes:
        rect = plt.Rectangle((x-0.8, y-0.5), 1.6, 1, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Backward arrows
    for i in range(len(grad_boxes)-1):
        ax.annotate('', xy=(grad_boxes[i][0]+0.8, 3), xytext=(grad_boxes[i+1][0]-0.8, 3),
                   arrowprops=dict(arrowstyle='<-', color='#e74c3c', lw=2))

    # Chain rule annotation
    ax.text(7, 4.5, 'Chain Rule: ∂L/∂W1 = ∂L/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂W1',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Backpropagation: How Gradients Flow', fontsize=14, fontweight='bold', pad=20)
    save_fig(fig, 'backprop_flow.png')

# -----------------------------------------------------------------------------
# 9. Gradient Descent Steps
# -----------------------------------------------------------------------------

def generate_gradient_descent_visualization():
    """Visualize gradient descent optimization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: 2D loss surface with gradient descent path
    ax = axes[0]

    # Create a simple 2D loss surface
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = W1**2 + W2**2 + 0.5*np.sin(3*W1)*np.cos(3*W2)

    ax.contour(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.7)
    ax.contourf(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.3)

    # Simulate gradient descent path
    path_w1 = [2.5]
    path_w2 = [2.5]
    lr = 0.1
    for _ in range(20):
        grad_w1 = 2*path_w1[-1] + 1.5*np.cos(3*path_w1[-1])*np.cos(3*path_w2[-1])
        grad_w2 = 2*path_w2[-1] - 1.5*np.sin(3*path_w1[-1])*np.sin(3*path_w2[-1])
        path_w1.append(path_w1[-1] - lr * grad_w1)
        path_w2.append(path_w2[-1] - lr * grad_w2)

    ax.plot(path_w1, path_w2, 'ro-', markersize=4, linewidth=1.5, label='GD path')
    ax.plot(path_w1[0], path_w2[0], 'go', markersize=12, label='Start')
    ax.plot(path_w1[-1], path_w2[-1], 'r*', markersize=15, label='End')

    ax.set_xlabel('Weight 1', fontsize=11)
    ax.set_ylabel('Weight 2', fontsize=11)
    ax.set_title('Gradient Descent on Loss Surface', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Plot 2: Learning rate comparison
    ax = axes[1]

    steps = np.arange(50)

    # Different learning rates
    for lr, color, label in [(0.01, 'blue', 'lr=0.01 (slow)'),
                              (0.1, 'green', 'lr=0.1 (good)'),
                              (0.5, 'red', 'lr=0.5 (unstable)')]:
        loss = [10]
        w = 3
        for _ in steps[1:]:
            grad = 2 * w
            w = w - lr * grad
            loss.append(w**2)
        ax.plot(steps, loss, color=color, linewidth=2, label=label)

    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Effect of Learning Rate', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'gradient_descent.png')

# -----------------------------------------------------------------------------
# 10. Training Progress Animation Frames
# -----------------------------------------------------------------------------

def generate_training_progress(model, X_train, y_train, X_test, y_test):
    """Generate visualization of training progress."""
    # Train for a few epochs and capture snapshots
    epochs = [0, 1, 3, 5, 10]
    batch_size = 64
    n_batches = len(X_train) // batch_size

    # Store model states
    accuracies = []

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    current_epoch = 0
    snapshot_idx = 0

    # Initial state
    acc = np.mean(model.forward(X_test).argmax(axis=1) == y_test)
    accuracies.append(acc)

    # Show initial weights
    for col, epoch_target in enumerate(epochs):
        # Train to this epoch
        while current_epoch < epoch_target:
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                model.forward(X_batch)
                model.backward(y_batch)
                model.update_weights(0.1)

            current_epoch += 1

        acc = np.mean(model.forward(X_test).argmax(axis=1) == y_test)

        # Top row: sample weights
        weights_sample = model.W1[:, :16].T.reshape(16, 28, 28)
        weight_grid = np.zeros((4*28, 4*28))
        for i in range(4):
            for j in range(4):
                weight_grid[i*28:(i+1)*28, j*28:(j+1)*28] = weights_sample[i*4+j]

        axes[0, col].imshow(weight_grid, cmap='RdBu', vmin=-0.3, vmax=0.3)
        axes[0, col].set_title(f'Epoch {epoch_target}\nWeights', fontsize=10)
        axes[0, col].axis('off')

        # Bottom row: predictions
        sample_idx = 0
        probs = model.forward(X_test[sample_idx:sample_idx+1])
        axes[1, col].bar(range(10), probs.flatten(), color='#3498db')
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_title(f'Acc: {acc:.1%}', fontsize=10)
        axes[1, col].set_xlabel('Digit')
        if col == 0:
            axes[1, col].set_ylabel('Probability')

    plt.suptitle('Training Progress: How Weights and Predictions Evolve', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'training_progress.png')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Generating documentation images...")
    print("=" * 50)

    # Load data
    print("\n1. Loading MNIST data...")
    X_test = load_mnist_images(Path("data/t10k-images-idx3-ubyte.gz"))
    y_test = load_mnist_labels(Path("data/t10k-labels-idx1-ubyte.gz"))
    X_train = load_mnist_images(Path("data/train-images-idx3-ubyte.gz"))
    y_train = load_mnist_labels(Path("data/train-labels-idx1-ubyte.gz"))

    # Create and train model
    print("\n2. Training model for visualizations...")
    model = NeuralNetwork()

    # Quick training
    batch_size = 64
    n_batches = len(X_train) // batch_size
    for epoch in range(10):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            model.forward(X_shuffled[start:end])
            model.backward(y_shuffled[start:end])
            model.update_weights(0.1)

    acc = np.mean(model.forward(X_test).argmax(axis=1) == y_test)
    print(f"   Model accuracy: {acc:.2%}")

    # Generate all images
    print("\n3. Generating images...")

    print("   - Architecture diagram")
    generate_architecture_diagram()

    print("   - Forward pass visualization")
    generate_forward_pass_visualization(model, X_test[0], y_test[0])

    print("   - Hidden layer activations")
    generate_hidden_activations(model, X_test, y_test)

    print("   - Neuron weight patterns")
    generate_neuron_patterns(model, X_test, y_test)

    print("   - ReLU visualization")
    generate_relu_visualization()

    print("   - Softmax visualization")
    generate_softmax_visualization()

    print("   - Cross-entropy loss")
    generate_loss_visualization()

    print("   - Backpropagation flow")
    generate_backprop_visualization()

    print("   - Gradient descent")
    generate_gradient_descent_visualization()

    # Reset and train again for progress visualization
    print("   - Training progress (this takes a moment...)")
    model2 = NeuralNetwork()
    generate_training_progress(model2, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 50)
    print("Done! All images saved to docs/images/")
