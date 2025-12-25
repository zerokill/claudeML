# Neural Network Documentation

A complete guide to understanding neural networks through building an MNIST digit classifier from scratch.

## Chapters

| Chapter | Title | Description |
|---------|-------|-------------|
| [00](00_overview.md) | **Overview & Architecture** | What is a neural network? Our architecture and learning process |
| [01](01_forward_propagation.md) | **Forward Propagation** | How data flows through the network to make predictions |
| [02](02_activation_functions.md) | **Activation Functions** | ReLU and Softmax explained with visualizations |
| [03](03_loss_function.md) | **Loss Function** | Cross-entropy loss - measuring prediction error |
| [04](04_backpropagation.md) | **Backpropagation** | Computing gradients via the chain rule |
| [05](05_training_results.md) | **Training & Results** | Gradient descent, mini-batches, and final results |

## Quick Reference

### Network Architecture
```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

### Key Equations

| Step | Equation |
|------|----------|
| Linear transform | $z = X \cdot W + b$ |
| ReLU | $a = \max(0, z)$ |
| Softmax | $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Cross-entropy | $L = -\log(\hat{y}_{true})$ |
| Gradient descent | $W = W - \eta \cdot \nabla W$ |

### Results

- **Test Accuracy**: 97.71%
- **Training Time**: ~30 seconds
- **Parameters**: 101,770

## Images

All visualization images are in the `images/` directory:

| Image | Description |
|-------|-------------|
| `architecture.png` | Network structure diagram |
| `forward_pass_steps.png` | Step-by-step forward propagation |
| `hidden_activations_by_digit.png` | How hidden neurons activate for each digit |
| `neuron_weight_patterns.png` | What patterns each neuron detects |
| `relu_visualization.png` | ReLU function and its derivative |
| `softmax_visualization.png` | Logits to probabilities transformation |
| `cross_entropy_loss.png` | Loss function visualization |
| `backprop_flow.png` | Gradient flow diagram |
| `gradient_descent.png` | Optimization visualization |
| `training_progress.png` | Weight evolution during training |
| `training_history.png` | Loss and accuracy curves |
| `predictions.png` | Sample predictions |
| `weights.png` | Learned weight patterns |

## Reading Order

For the best learning experience, read the chapters in order:

```mermaid
flowchart LR
    A[00 Overview] --> B[01 Forward]
    B --> C[02 Activations]
    C --> D[03 Loss]
    D --> E[04 Backprop]
    E --> F[05 Training]

    style A fill:#3498db,color:white
    style F fill:#2ecc71,color:white
```

Each chapter builds on the previous, with:
- **Description**: Intuitive explanation
- **Math**: Formal equations
- **Code**: Implementation details
- **Visualizations**: Diagrams and results
