# AnasFlow - From-Scratch Neural Network Library

A comprehensive artificial neural network library built entirely from scratch using only NumPy and Pandas, featuring multiple gradient descent optimization techniques and advanced metaheuristic algorithms for parameter estimation.

## 🎯 Project Overview

AnasFlow is a custom neural network implementation designed for MNIST digit classification. The library demonstrates various optimization strategies and training methodologies without relying on high-level machine learning frameworks like TensorFlow or PyTorch.

## ✨ Key Features

### Gradient Descent Variants
- **Batch Gradient Descent**: Processes entire dataset before updating parameters
- **Mini-batch Gradient Descent**: Updates parameters using small batches of data
- **Stochastic Gradient Descent (SGD)**: Updates parameters after each individual sample

### Advanced Optimization Algorithms
- **Momentum**: Accelerates convergence by accumulating velocity vectors
- **AdaGrad**: Adapts learning rates based on historical gradients
- **RMSprop**: Improves AdaGrad by using exponential moving averages

### Neural Network Components
- **Activation Functions**: Sigmoid, Softmax, Tanh, ReLU
- **Loss Functions**: Mean Squared Error (MSE), Cross-Entropy
- **Architecture**: Fully customizable layer sizes and depths
- **Weight Initialization**: Uniform random initialization

## 📁 Project Structure

```
anasflow/
├── anasflow.py              # Main neural network implementation
├── batch/                   # Batch gradient descent implementation
│   ├── neural_network.py
│   └── test.py
├── minibatch/              # Mini-batch gradient descent implementation
│   ├── neural_network.py
│   └── test.py
├── stochastic/             # Stochastic gradient descent implementation
│   ├── neural_network.py
│   └── test.py
├── version adagrad/        # AdaGrad optimization implementation
│   ├── neural_network.py
│   ├── test.py
│   └── algorithm.png
├── version momentum/       # Momentum optimization implementation
│   ├── neural_network.py
│   └── test.py
└── version rmspromp/       # RMSprop optimization implementation
    ├── neural_network.py
    └── test.py
```

## 🚀 Quick Start

### Basic Usage

```python
from anasflow import NN
import pandas as pd

# Load MNIST data
data = pd.read_csv("mnist_train.csv")
images, labels = getData(data)

# Initialize neural network
nn = NN(
    lr=0.01,
    activation="sigmouid",
    activatioOut="softmax",
    errorFunc="crossentropy",
    sizes_layers=[784, 128, 64, 10]
)

# Train the model
accuracy = nn.train(
    epoches=50,
    images=images,
    labels=labels,
    minibatch=32
)

print(f"Training Accuracy: {accuracy}%")
```

### Different Optimization Methods

#### Batch Gradient Descent
```python
from batch.neural_network import NN

nn = NN([784, 200, 100, 50, 10])
nn.train(lr=0.1, epoches=70, images=images, labels=labels)
```

#### Mini-batch Gradient Descent
```python
from minibatch.neural_network import NN

nn = NN([784, 50, 31, 46, 11, 35, 10])
nn.train(lr=0.9, epoches=70, images=images, labels=labels, minibatch=10)
```

#### Stochastic Gradient Descent
```python
from stochastic.neural_network import NN

nn = NN()
nn.train(lr=0.1, epoches=70, images=images, labels=labels)
```

#### AdaGrad Optimization
```python
from version_adagrad.neural_network import NN

nn = NN([784, 200, 100, 50, 10])
nn.train(lr=0.1, epoches=70, images=images, labels=labels)
```

#### Momentum Optimization
```python
from version_momentum.neural_network import NN

nn = NN([784, 200, 100, 50, 10])
nn.train(lr=0.1, epoches=70, images=images, labels=labels)
```

## 🧠 Architecture Details

### Network Configuration
- **Input Layer**: 784 neurons (28×28 MNIST images)
- **Hidden Layers**: Customizable sizes and depths
- **Output Layer**: 10 neurons (digit classes 0-9)
- **Activation**: Sigmoid for hidden layers, Softmax for output
- **Loss Function**: Cross-entropy for classification

### Mathematical Implementation

#### Forward Propagation
```
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = σ(z^(l))
```

#### Backpropagation
```
δ^(L) = ∇_a C ⊙ σ'(z^(L))
δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))
```

#### Parameter Updates
- **SGD**: `θ = θ - α∇θ`
- **Momentum**: `v = βv + α∇θ; θ = θ - v`
- **AdaGrad**: `θ = θ - (α/√(G + ε))∇θ`

## 📊 Performance

The library has been tested on the MNIST dataset with various configurations:

- **Dataset**: MNIST handwritten digits (60,000 training samples)
- **Task**: 10-class classification
- **Architecture**: Multiple tested configurations
- **Best Performance**: Achieved competitive accuracy on digit recognition

## 🛠️ Requirements

```
numpy>=1.19.0
pandas>=1.3.0
pickle (built-in)
```

## 📈 Training Features

- **Flexible Architecture**: Define custom layer sizes
- **Multiple Optimizers**: Choose from various gradient descent variants
- **Learning Rate Scheduling**: Automatic learning rate decay
- **Model Persistence**: Save and load trained models using pickle
- **Real-time Monitoring**: Track accuracy during training

## 🔬 Research Applications

This implementation serves as an educational tool for understanding:
- Neural network fundamentals
- Gradient descent optimization
- Backpropagation algorithm
- Parameter initialization strategies
- Activation function behaviors
- Loss function characteristics

## 🤝 Contributing

This project demonstrates a complete from-scratch implementation of neural networks. Feel free to:
- Experiment with different architectures
- Add new activation functions
- Implement additional optimization algorithms
- Extend to other datasets

## 📝 License

This project is open source and available for educational and research purposes.

## 🎓 Educational Value

AnasFlow provides insight into:
- **Mathematical Foundations**: Understanding the math behind neural networks
- **Algorithm Implementation**: How optimization algorithms work internally
- **Performance Comparison**: Comparing different training strategies
- **Parameter Tuning**: Effects of hyperparameters on model performance

---

*Built with ❤️ for deep learning education and research*