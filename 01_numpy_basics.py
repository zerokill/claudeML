"""
01_numpy_basics.py - NumPy Operations for Neural Networks

This file covers the core NumPy operations you need to understand
before building a neural network from scratch. Run each section
and study the outputs!

Key operations:
1. Matrix multiplication (the core of neural networks)
2. Broadcasting (how biases get added)
3. Activation functions (vectorized)
4. Softmax and numerical stability
"""

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. MATRIX MULTIPLICATION - The Heart of Neural Networks
# =============================================================================
print("=" * 60)
print("1. MATRIX MULTIPLICATION")
print("=" * 60)

# In a neural network, each layer does: output = np.dot(input, weights) + bias
# Let's break this down:

# Imagine we have 3 samples, each with 4 features (like a mini-batch)
inputs = np.array([
    [1.0, 2.0, 3.0, 4.0],   # sample 1
    [5.0, 6.0, 7.0, 8.0],   # sample 2
    [9.0, 10., 11., 12.],   # sample 3
])
print(f"Input shape: {inputs.shape}")  # (3, 4) = (batch_size, input_features)

# Weights connect input features to output neurons
# If we want 2 output neurons, we need a (4, 2) weight matrix
weights = np.array([
    [0.1, 0.2],   # weights from input 0 to outputs 0,1
    [0.3, 0.4],   # weights from input 1 to outputs 0,1
    [0.5, 0.6],   # weights from input 2 to outputs 0,1
    [0.7, 0.8],   # weights from input 3 to outputs 0,1
])
print(f"Weights shape: {weights.shape}")  # (4, 2) = (input_features, output_neurons)

# Matrix multiplication: each row of inputs gets multiplied with each column of weights
output = np.dot(inputs, weights)  # np.dot performs matrix multiplication
print(f"Output shape: {output.shape}")  # (3, 2) = (batch_size, output_neurons)
print(f"Output:\n{output}")

# What just happened? For sample 1:
# output[0,0] = 1*0.1 + 2*0.3 + 3*0.5 + 4*0.7 = 0.1 + 0.6 + 1.5 + 2.8 = 5.0
# output[0,1] = 1*0.2 + 2*0.4 + 3*0.6 + 4*0.8 = 0.2 + 0.8 + 1.8 + 3.2 = 6.0
print(f"\nManual verification for sample 1, output 0: {1*0.1 + 2*0.3 + 3*0.5 + 4*0.7}")

# =============================================================================
# 2. BROADCASTING - How Biases Work
# =============================================================================
print("\n" + "=" * 60)
print("2. BROADCASTING")
print("=" * 60)

# Each output neuron has one bias value
biases = np.array([0.5, -0.5])  # shape: (2,)
print(f"Biases shape: {biases.shape}")

# When we add biases to output, NumPy "broadcasts" the (2,) bias
# to match the (3, 2) output shape - it repeats the bias for each sample
output_with_bias = output + biases
print(f"Output with bias:\n{output_with_bias}")

# This is equivalent to:
# [[5.0 + 0.5,  6.0 - 0.5],
#  [13.0 + 0.5, 14.0 - 0.5],
#  [21.0 + 0.5, 22.0 - 0.5]]

# Broadcasting rules:
# - Dimensions are compared from right to left
# - Dimensions must be equal OR one of them must be 1 (or missing)
# - The smaller array is "stretched" to match the larger one

# =============================================================================
# 3. ACTIVATION FUNCTIONS - Adding Non-linearity
# =============================================================================
print("\n" + "=" * 60)
print("3. ACTIVATION FUNCTIONS")
print("=" * 60)

# Without activation functions, stacking layers would just be
# one big linear transformation. We need non-linearity!

# ReLU (Rectified Linear Unit) - the most common activation
# ReLU(x) = max(0, x) - kills negative values, keeps positive ones
def relu(x):
    return np.maximum(0, x)

# Let's see ReLU in action
test_values = np.array([-2, -1, 0, 1, 2])
print(f"Input:       {test_values}")
print(f"After ReLU:  {relu(test_values)}")

# Why ReLU works well:
# 1. Simple gradient: 1 for positive, 0 for negative
# 2. No vanishing gradient problem (like sigmoid has)
# 3. Computationally efficient

# ReLU derivative (needed for backpropagation)
def relu_derivative(x):
    return (x > 0).astype(float)  # 1 where x > 0, else 0

print(f"ReLU derivative: {relu_derivative(test_values)}")

# =============================================================================
# 4. SOFTMAX - Converting Outputs to Probabilities
# =============================================================================
print("\n" + "=" * 60)
print("4. SOFTMAX")
print("=" * 60)

# For classification, we want probabilities that sum to 1
# Softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))

def softmax_naive(x):
    """Naive softmax - has numerical stability issues!"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax(x):
    """Numerically stable softmax - subtract max before exp"""
    # Subtracting max prevents overflow when exp(large_number)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example: raw outputs (logits) from final layer
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
print(f"Logits:        {logits}")
print(f"Probabilities: {probabilities}")
print(f"Sum:           {probabilities.sum():.6f}")  # Should be 1.0

# The highest logit gets the highest probability
# This is how we classify: pick the class with highest probability

# Batched softmax (multiple samples)
batch_logits = np.array([
    [2.0, 1.0, 0.1],   # sample 1: class 0 has highest logit
    [0.1, 2.0, 1.0],   # sample 2: class 1 has highest logit
    [0.5, 0.5, 2.0],   # sample 3: class 2 has highest logit
])
batch_probs = softmax(batch_logits)
print(f"\nBatch probabilities:\n{batch_probs}")
print(f"Predictions (argmax): {np.argmax(batch_probs, axis=1)}")

# =============================================================================
# 5. CROSS-ENTROPY LOSS - Measuring How Wrong We Are
# =============================================================================
print("\n" + "=" * 60)
print("5. CROSS-ENTROPY LOSS")
print("=" * 60)

# Cross-entropy measures difference between predicted and true distributions
# Loss = -sum(true * log(predicted))
# For classification: Loss = -log(predicted_probability_of_true_class)

def cross_entropy_loss(predictions, targets):
    """
    predictions: (batch_size, num_classes) - probabilities from softmax
    targets: (batch_size,) - integer class labels
    """
    batch_size = predictions.shape[0]
    # Get probability of correct class for each sample
    # predictions[range(batch_size), targets] picks the right probability
    correct_probs = predictions[np.arange(batch_size), targets]
    # Add small epsilon to avoid log(0)
    loss = -np.log(correct_probs + 1e-10)
    return np.mean(loss)  # Average over batch

# Example
predictions = np.array([
    [0.7, 0.2, 0.1],  # Predicts class 0 with 70% confidence
    [0.1, 0.8, 0.1],  # Predicts class 1 with 80% confidence
    [0.2, 0.3, 0.5],  # Predicts class 2 with 50% confidence
])
targets = np.array([0, 1, 2])  # True classes

loss = cross_entropy_loss(predictions, targets)
print(f"Predictions:\n{predictions}")
print(f"True classes: {targets}")
print(f"Cross-entropy loss: {loss:.4f}")

# Lower loss = better predictions
# Perfect prediction (100% on true class) → loss = -log(1) = 0
# Wrong prediction (0% on true class) → loss = -log(0) = infinity

# =============================================================================
# 6. PUTTING IT TOGETHER - One Forward Pass
# =============================================================================
print("\n" + "=" * 60)
print("6. COMPLETE FORWARD PASS")
print("=" * 60)

# Let's simulate one forward pass through a simple network:
# Input (4 features) → Hidden (3 neurons, ReLU) → Output (2 classes, Softmax)

# Initialize random weights
W1 = np.random.randn(4, 3) * 0.5  # Input to hidden
b1 = np.zeros(3)
W2 = np.random.randn(3, 2) * 0.5  # Hidden to output
b2 = np.zeros(2)

# Sample input (2 samples, 4 features each)
X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [4.0, 3.0, 2.0, 1.0],
])

# Forward pass
print("Step 1: Input")
print(f"  X shape: {X.shape}")

print("\nStep 2: Hidden layer (linear)")
z1 = np.dot(X, W1) + b1  # Linear transformation
print(f"  z1 = np.dot(X, W1) + b1")
print(f"  z1 shape: {z1.shape}")
print(f"  z1:\n{z1}")

print("\nStep 3: Hidden layer (activation)")
a1 = relu(z1)  # Apply ReLU
print(f"  a1 = relu(z1)")
print(f"  a1:\n{a1}")

print("\nStep 4: Output layer (linear)")
z2 = np.dot(a1, W2) + b2
print(f"  z2 = np.dot(a1, W2) + b2")
print(f"  z2 shape: {z2.shape}")
print(f"  z2:\n{z2}")

print("\nStep 5: Output layer (softmax)")
a2 = softmax(z2)
print(f"  a2 = softmax(z2)")
print(f"  a2 (probabilities):\n{a2}")
print(f"  Predictions: {np.argmax(a2, axis=1)}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY - What You've Learned")
print("=" * 60)
print("""
1. Matrix multiplication with np.dot() is how layers transform data
   - Shape: np.dot(batch, input_features), (input_features, output_neurons)) = (batch, output_neurons)

2. Broadcasting automatically expands biases to match batch dimension

3. ReLU activation: max(0, x) - adds non-linearity, simple gradient

4. Softmax converts raw outputs to probabilities that sum to 1

5. Cross-entropy loss measures how wrong our predictions are

Next up: 02_neural_net_numpy.py - We'll implement backpropagation
and train this network on real MNIST digits!
""")
