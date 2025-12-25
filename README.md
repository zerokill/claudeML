# ML Learning Project - Neural Network from Scratch

Learn how neural networks work by building an MNIST digit classifier.

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
```

### 2. Activate the virtual environment

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the tutorials

Make sure your venv is activated, then:

```bash
# Step 1: NumPy basics refresher
python 01_numpy_basics.py

# Step 2: Neural network from scratch
python 02_neural_net_numpy.py

# Step 3: PyTorch equivalent
python 03_neural_net_pytorch.py
```

## Files

| File | Description |
|------|-------------|
| `01_numpy_basics.py` | Matrix operations, activations, softmax |
| `02_neural_net_numpy.py` | Full neural network with backpropagation |
| `03_neural_net_pytorch.py` | Same network using PyTorch |
