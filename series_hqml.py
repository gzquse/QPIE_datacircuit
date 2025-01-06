#!/usr/bin/env python3

import cudaq
from cudaq import spin

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torchvision

from sklearn.model_selection import train_test_split

torch.manual_seed(22)
cudaq.set_random_seed(44)

cudaq.set_target("nvidia")
device = torch.device("cuda:0")

def prepare_data(target_digits, sample_count, test_size):
    """Load and prepare the MNIST dataset to be used

    Args:
        target_digits (list): digits to perform classification of
        sample_count (int): total number of images to be used
        test_size (float): percentage of sample_count to be used as test set, the remainder is the training set

    Returns:
        dataset in train, test format with targets

    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307), (0.3081))])

    dataset = datasets.MNIST("./data",
                             train=True,
                             download=True,
                             transform=transform)

    # Filter out the required labels.
    idx = (dataset.targets == target_digits[0]) | (dataset.targets
                                                   == target_digits[1])
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    # Select a subset based on number of datapoints specified by sample_count.
    subset_indices = torch.randperm(dataset.data.size(0))[:sample_count]

    x = dataset.data[subset_indices].float().unsqueeze(1).to(device)

    y = dataset.targets[subset_indices].to(device).float().to(device)

    # Relabel the targets as a 0 or a 1.
    y = torch.where(y == min(target_digits), 0.0, 1.0)

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size /
                                                        100,
                                                        shuffle=True,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test

def generate_narma_sequence(order, length, seed=0):
    """Generate NARMA sequence of a given order and length."""
    np.random.seed(seed)
    u = np.random.rand(length)
    y = np.zeros(length)
    for t in range(order, length):
        if order == 5:
            y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-5:t]) + 1.5 * u[t-1] * u[t-5] + 0.1
        elif order == 10:
            y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t]) + 1.5 * u[t-1] * u[t-10] + 0.1
    return u, y

def prepare_narma_data(order, sample_count, test_size):
    """Prepare NARMA dataset for training and testing."""
    u, y = generate_narma_sequence(order, sample_count + test_size)
    x_train, x_test, y_train, y_test = train_test_split(u, y, test_size=test_size/sample_count, shuffle=False)
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    return x_train, x_test, y_train, y_test

# Classical parameters.

sample_count = 1000  # Total number of images to use.
target_digits = [5, 6]  # Hand written digits to classify.
test_size = 30  # Percentage of dataset to be used for testing.
classification_threshold = 0.5  # Classification boundary used to measure accuracy.
epochs = 10  # Number of epochs to train for.

# Quantum parmeters.

qubit_count = 1
hamiltonian = spin.z(0)  # Measurement operator.
shift = torch.tensor(torch.pi / 2)  # Magnitude of parameter shift.

x_train, x_test, y_train, y_test = prepare_data(target_digits, sample_count,
                                                test_size)

class QuantumFunction(Function):
    """Allows the quantum circuit to input data, output expectation values
    and calculate gradients of variational parameters via finite difference"""

    def __init__(self, qubit_count: int, hamiltonian: cudaq.SpinOperator):
        """Define the quantum circuit in CUDA Quantum"""

        @cudaq.kernel
        def kernel(qubit_count: int, thetas: np.ndarray):

            qubits = cudaq.qvector(qubit_count)

            ry(thetas[0], qubits[0])
            rx(thetas[1], qubits[0])

        self.kernel = kernel
        self.qubit_count = qubit_count
        self.hamiltonian = hamiltonian

    def run(self, theta_vals: torch.tensor) -> torch.tensor:
        """Excetute the quantum circuit to output an expectation value"""

        #If running on GPU, thetas is a torch.tensor that will live on GPU memory. The observe function calls a .tolist() method on inputs which moves thetas from GPU to CPU.

        qubit_count = [self.qubit_count for _ in range(theta_vals.shape[0])]

        results = cudaq.observe(self.kernel, self.hamiltonian, qubit_count,
                                theta_vals)

        exp_vals = torch.tensor(
            [results[i].expectation() for i in range(len(results))]
        ).to(device)

        return exp_vals

    @staticmethod
    def forward(ctx, thetas: torch.tensor, quantum_circuit,
                shift) -> torch.tensor:

        # Save shift and quantum_circuit in bcontext to use in ackward.
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        # Calculate expectation value.
        exp_vals = ctx.quantum_circuit.run(thetas).reshape(-1, 1)

        ctx.save_for_backward(thetas, exp_vals)

        return exp_vals

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computation via finite difference"""

        thetas, _ = ctx.saved_tensors

        gradients = torch.zeros(thetas.shape, device=device)

        for i in range(thetas.shape[1]):

            thetas_plus = thetas.clone()
            thetas_plus[:, i] += ctx.shift
            exp_vals_plus = ctx.quantum_circuit.run(thetas_plus)

            thetas_minus = thetas.clone()
            thetas_minus[:, i] -= ctx.shift
            exp_vals_minus = ctx.quantum_circuit.run(thetas_minus)

            gradients[:, i] = (exp_vals_plus - exp_vals_minus) / (2 * ctx.shift)

        gradients = torch.mul(grad_output, gradients)

        return gradients, None, None

class QuantumLayer(nn.Module):
    """Encapsulates a quantum circuit into a quantum layer adhering PyTorch convention"""

    def __init__(self, qubit_count: int, hamiltonian, shift: torch.tensor):
        super(QuantumLayer, self).__init__()

        self.quantum_circuit = QuantumFunction(qubit_count, hamiltonian)
        self.shift = shift

    def forward(self, input):

        result = QuantumFunction.apply(input, self.quantum_circuit, self.shift)

        return result

class Hybrid_QNN(nn.Module):
    """Structure of the hybrid neural network with classical fully connected layers and quantum layers"""

    def __init__(self):
        super(Hybrid_QNN, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.25)

        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.25)

        # The 2 outputs from PyTorch fc5 layer feed into the 2 variational gates in the quantum circuit.
        self.quantum = QuantumLayer(qubit_count, hamiltonian, shift)


    def forward(self, x):

        x = x.view(-1, 28 * 28)  # Turns images into vectors.

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)

        # Quantum circuit outputs an expectation value which is fed into the sigmoid activation function to perform classification.
        x = torch.sigmoid(self.quantum(x))

        return x.view(-1)

def accuracy_score(y, y_hat):
    return sum((y == (y_hat >= classification_threshold))) / len(y)

hybrid_model = Hybrid_QNN().to(device)

optimizer = optim.Adadelta(hybrid_model.parameters(),
                           lr=0.001,
                           weight_decay=0.8)

loss_function = nn.BCELoss().to(device)

training_cost = []
testing_cost = []
training_accuracy = []
testing_accuracy = []

hybrid_model.train()
for epoch in range(epochs):

    optimizer.zero_grad()

    y_hat_train = hybrid_model(x_train).to(device)

    train_cost = loss_function(y_hat_train, y_train).to(device)

    train_cost.backward()

    optimizer.step()

    training_accuracy.append(accuracy_score(y_train, y_hat_train))
    training_cost.append(train_cost.item())

    hybrid_model.eval()
    with torch.no_grad():

        y_hat_test = hybrid_model(x_test).to(device)

        test_cost = loss_function(y_hat_test, y_test).to(device)

        testing_accuracy.append(accuracy_score(y_test, y_hat_test))
        testing_cost.append(test_cost.item())


# Ensure all items in the lists are converted to NumPy arrays
training_accuracy = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in training_accuracy]
testing_accuracy = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in testing_accuracy]
training_cost = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in training_cost]
testing_cost = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in testing_cost]

training_accuracy = [float(x) for x in training_accuracy]
testing_accuracy = [float(x) for x in testing_accuracy]
training_cost = [float(x) for x in training_cost]
testing_cost = [float(x) for x in testing_cost]

print("Training Accuracy:", training_accuracy[:5], "Length:", len(training_accuracy))
print("Testing Accuracy:", testing_accuracy[:5], "Length:", len(testing_accuracy))
print("Training Cost:", training_cost[:5], "Length:", len(training_cost))
print("Testing Cost:", testing_cost[:5], "Length:", len(testing_cost))

# Print final results
print("Final Training Accuracy:", training_accuracy[-1])
print("Final Testing Accuracy:", testing_accuracy[-1])
print("Final Training Cost:", training_cost[-1])
print("Final Testing Cost:", testing_cost[-1])

# Plot and save the training and testing cost and accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(training_cost, label='Train')
plt.plot(testing_cost, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Train')
plt.plot(testing_accuracy, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./out/hqml_series.png')
plt.show()

# Plot ground truth and predictions
hybrid_model.eval()
with torch.no_grad():
    y_hat_train = hybrid_model(x_train).cpu().numpy()
    y_hat_test = hybrid_model(x_test).cpu().numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(y_train.cpu().numpy(), label='Ground Truth')
plt.plot(y_hat_train, label='Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test.cpu().numpy(), label='Ground Truth')
plt.plot(y_hat_test, label='Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Testing Data')
plt.legend()

plt.tight_layout()
plt.savefig('./out/hqml_series_predictions.png')
plt.show()

# Plot ground truth and predictions for NARMA 5 model
order = 5
sample_count = 1000
test_size = 200
x_train, x_test, y_train, y_test = prepare_narma_data(order, sample_count, test_size)

hybrid_model.eval()
with torch.no_grad():
    y_hat_train = hybrid_model(x_train).cpu().numpy()
    y_hat_test = hybrid_model(x_test).cpu().numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(y_train.cpu().numpy(), label='Ground Truth')
plt.plot(y_hat_train, label='Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('NARMA 5 Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test.cpu().numpy(), label='Ground Truth')
plt.plot(y_hat_test, label='Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('NARMA 5 Testing Data')
plt.legend()

plt.tight_layout()
plt.savefig('./out/narma5_predictions.png')
plt.show()

# Plot ground truth and predictions for NARMA 10 model
order = 10
x_train, x_test, y_train, y_test = prepare_narma_data(order, sample_count, test_size)

hybrid_model.eval()
with torch.no_grad():
    y_hat_train = hybrid_model(x_train).cpu().numpy()
    y_hat_test = hybrid_model(x_test).cpu().numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(y_train.cpu().numpy(), label='Ground Truth')
plt.plot(y_hat_train, label='Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('NARMA 10 Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test.cpu().numpy(), label='Ground Truth')
plt.plot(y_hat_test, label='Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('NARMA 10 Testing Data')
plt.legend()

plt.tight_layout()
plt.savefig('./out/narma10_predictions.png')
plt.show()
