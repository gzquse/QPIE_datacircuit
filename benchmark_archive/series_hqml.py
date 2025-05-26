#!/usr/bin/env python3

import cudaq
from cudaq import spin

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(22)
cudaq.set_random_seed(44)

cudaq.set_target("nvidia")
device = torch.device("cuda:0")


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
        y[t] = np.clip(y[t], -1e10, 1e10)  # Clip values to avoid overflow
    return u, y


def prepare_narma_data(order, sample_count, test_size):
    """Prepare NARMA dataset for training and testing."""
    u, y = generate_narma_sequence(order, sample_count + test_size)

    # Normalize targets to [0, 1]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    x_train, x_test, y_train, y_test = train_test_split(
        u, y, test_size=test_size/sample_count, shuffle=False
    )

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return x_train, x_test, y_train, y_test



class QuantumFunction(Function):
    def __init__(self, qubit_count: int, hamiltonian: cudaq.SpinOperator):
        @cudaq.kernel
        def kernel(qubit_count: int, thetas: np.ndarray):
            qubits = cudaq.qvector(qubit_count)
            ry(thetas[0], qubits[0])
            rx(thetas[1], qubits[0])

        self.kernel = kernel
        self.qubit_count = qubit_count
        self.hamiltonian = hamiltonian

    def run(self, theta_vals: torch.tensor) -> torch.tensor:
        qubit_count = [self.qubit_count for _ in range(theta_vals.shape[0])]
        results = cudaq.observe(self.kernel, self.hamiltonian, qubit_count, theta_vals)
        exp_vals = torch.tensor(
            [results[i].expectation() for i in range(len(results))]
        ).to(device)
        return exp_vals

    @staticmethod
    def forward(ctx, thetas: torch.tensor, quantum_circuit, shift) -> torch.tensor:
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        exp_vals = ctx.quantum_circuit.run(thetas).reshape(-1, 1)
        ctx.save_for_backward(thetas, exp_vals)
        return exp_vals

    @staticmethod
    def backward(ctx, grad_output):
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
    def __init__(self, qubit_count: int, hamiltonian, shift: torch.tensor):
        super(QuantumLayer, self).__init__()
        self.quantum_circuit = QuantumFunction(qubit_count, hamiltonian)
        self.shift = shift

    def forward(self, input):
        result = QuantumFunction.apply(input, self.quantum_circuit, self.shift)
        return result


class Hybrid_QNN(nn.Module):
    def __init__(self):
        super(Hybrid_QNN, self).__init__()
        self.fc1 = nn.Linear(1, 256)  # Adjusted for NARMA data
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.quantum = QuantumLayer(1, spin.z(0), torch.tensor(torch.pi / 2))

    def forward(self, x):
        x = x.view(-1, x.shape[1])  # Handle 1D input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.quantum(x))
        return x.view(-1)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Prepare NARMA data
order = 5  # Change to 10 for NARMA-10
sample_count = 1000
test_size = 200
x_train, x_test, y_train, y_test = prepare_narma_data(order, sample_count, test_size)

# Model setup
epochs = 10
classification_threshold = 0.5
hybrid_model = Hybrid_QNN().to(device)
hybrid_model.apply(initialize_weights)

# Adjust learning rate
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)
loss_function = nn.BCELoss().to(device)

training_cost = []
testing_cost = []
training_accuracy = []
testing_accuracy = []

# Training loop
for epoch in range(epochs):
    hybrid_model.train()
    optimizer.zero_grad()
    y_hat_train = hybrid_model(x_train).to(device)
    train_cost = loss_function(y_hat_train, y_train).to(device)
    train_cost.backward()
    optimizer.step()
    training_accuracy.append((y_train == (y_hat_train >= classification_threshold)).float().mean().item())
    training_cost.append(train_cost.item())

    hybrid_model.eval()
    with torch.no_grad():
        y_hat_test = hybrid_model(x_test).to(device)
        test_cost = loss_function(y_hat_test, y_test).to(device)
        testing_accuracy.append((y_test == (y_hat_test >= classification_threshold)).float().mean().item())
        testing_cost.append(test_cost.item())

# Plot training and testing results
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
plt.savefig('narma5_acc.pdf', format='pdf')

# Generate predictions for NARMA-5
hybrid_model.eval()
with torch.no_grad():
    y_hat_train = hybrid_model(x_train).cpu().numpy()
    y_hat_test = hybrid_model(x_test).cpu().numpy()

# Debugging: Check ranges of predictions and ground truth
print("Training Ground Truth Range:", y_train.min().item(), y_train.max().item())
print("Training Predictions Range:", y_hat_train.min(), y_hat_train.max())
print("Testing Ground Truth Range:", y_test.min().item(), y_test.max().item())
print("Testing Predictions Range:", y_hat_test.min(), y_hat_test.max())

# Plot NARMA-5 ground truth and predictions
plt.figure(figsize=(10, 5))

# Plot training data
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train.cpu().numpy(), label='Ground Truth', color='blue')
plt.plot(range(len(y_hat_train)), y_hat_train, label='Prediction', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('NARMA-5 Training Data')
plt.legend()

# Plot testing data
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test.cpu().numpy(), label='Ground Truth', color='blue')
plt.plot(range(len(y_hat_test)), y_hat_test, label='Prediction', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('NARMA-5 Testing Data')
plt.legend()

# Save the figure to a PDF
plt.tight_layout()
plt.savefig('narma5_ground_truth_and_prediction_corrected.pdf', format='pdf')
plt.show()

