
# This cell is added by sphinx-gallery
# It can be customized to whatever you like

# !pip install pennylane
import time
seconds = time.time()
print("Time in seconds since beginning of run:", seconds)
local_time = time.ctime(seconds)
print(local_time)


from pennylane import broadcast
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

# Set random seeds
torch.manual_seed(149)
np.random.seed(149)

X, y = make_moons(n_samples=400, noise=0.1)
y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((400, 2)), 1, y_, 1)

c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colours for each class
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


import pennylane as qml

n_qubits = 10
dev = qml.device("lightning.qubit", wires=n_qubits)

# Define the Hadamard layer
def H_layer(n_qubits):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

# Define a custom function to add arbitrary rotations
def custom_rot_layer(weights, wires):
    # Apply Rot (arbitrary rotation) to each qubit individually
    for i, wire in enumerate(wires):
        phi, theta, omega = weights[i]
        qml.Rot(phi, theta, omega, wires=wire)

# Define a custom function for entanglement
def entangling_layer(wires, measured_1, measured_2):
    for j in range(len(wires) - 1):
        # Apply CNOT unconditionally between adjacent qubits
        qml.CNOT(wires=[wires[j], wires[j + 1]])

        # Conditionally apply CRZ if measured_1 is 1 (control qubit on qubit 1)
        qml.cond(measured_1, qml.CRZ)(np.pi / 2, wires=[wires[j], wires[j + 1]])

        # Conditionally apply a controlled-RY gate if measured_2 is 1 (control qubit on qubit 2)
        qml.cond(measured_2, qml.CRY)(np.pi / 4, wires=[wires[j], wires[j + 1]])

@qml.qnode(dev)
def qnode(inputs, weights, x, z):
    H_layer(n_qubits)

    for i in range(n_layers):
        # Embedding with alternating rotation axes
        if i % 2 == 0:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Z')
        
        # Measure qubits 1 and 2 for conditional execution
        measured_1 = qml.measure(1)
        measured_2 = qml.measure(2)

        # Add an entangling layer with partial conditional gates
        entangling_layer(range(n_qubits), measured_1, measured_2)
        
        # Apply a custom layer of arbitrary rotations using qml.Rot
        custom_rot_layer(weights[i], range(n_qubits))

    # Apply a final set of Rot gates to close the circuit
    custom_rot_layer(weights[-1], range(n_qubits))
    
    # Apply a controlled CRZ rotation based on the measurement of qubit 2
    qml.cond(measured_2, qml.CRZ)(z, wires=(3, 0))

    # Final Hadamard layer
    H_layer(n_qubits)

    # Return PauliZ expectation values and the measurement probabilities for qubit 2
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)], qml.probs(op=measured_2)


n_layers = 5
weight_shapes = {"weights": (n_layers, n_qubits)}


w = np.random.random(size=(n_layers, n_qubits))
fig, ax = qml.draw_mpl(qnode, show_all_wires=True, decimals=2)(w,w)
fig.show()


print(qml.draw(qnode, expansion_strategy="device")(w,w))


loss_func = torch.nn.L1Loss()


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(2, 320)
        
        # Define a list of quantum layers using ModuleList
        self.qlayers = torch.nn.ModuleList(
            [qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(320)]
        )
        
        self.clayer_2 = torch.nn.Linear(320, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        # Split input into 320 chunks
        x_split = torch.split(x, 1, dim=1)
        
        # Apply each quantum layer to each chunk
        x_processed = [layer(x_i) for layer, x_i in zip(self.qlayers, x_split)]
        
        # Concatenate the processed outputs
        x = torch.cat(x_processed, dim=1)
        
        x = self.clayer_2(x)
        return self.softmax(x)

model = HybridModel()

model = HybridModel()


from sklearn.model_selection import train_test_split


X = torch.tensor(X, requires_grad=True).float()
y_hot = y_hot.float()

batch_size = 5
batches = 200 // batch_size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=149)
# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, requires_grad=True).float()
y_ = torch.unsqueeze(torch.tensor(y_train), 1)  # used for one-hot encoded labels
y_train_hot = torch.scatter(torch.zeros((280, 2)), 1, y_, 1)
y_train_hot = y_train_hot.float()

X_test = torch.tensor(X_test, requires_grad=True).float()
y_ = torch.unsqueeze(torch.tensor(y_test), 1)  # used for one-hot encoded labels
y_test_hot = torch.scatter(torch.zeros((120, 2)), 1, y_, 1)
y_test_hot = y_test_hot.float()

train_dataset = TensorDataset(X_train, y_train_hot)
test_dataset = TensorDataset(X_test, y_test_hot)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, drop_last=True)

opt = torch.optim.Adam(model.parameters(), lr=0.0004)

train_acc_history = []
test_acc_history = []

epochs = 200

for epoch in range(epochs):
    train_running_loss = 0
    test_running_loss = 0
    for xs, ys in train_loader:
        opt.zero_grad()

        loss_evaluated = loss_func(model(xs), ys)
        loss_evaluated.backward()

        opt.step()

        train_running_loss += loss_evaluated.item()
        
    train_avg_loss = train_running_loss / batches

    print(f"Train - Epoch {epoch + 1}: Loss: {train_avg_loss:.4f}")

    # Calculate train accuracy
    y_pred_train = model(X_train)
    predictions_train = torch.argmax(y_pred_train, axis=1).detach().numpy()

    correct_train = [1 if p == p_true else 0 for p, p_true in zip(predictions_train, y_train)]
    train_accuracy = sum(correct_train) / len(correct_train)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    train_acc_history.append(train_accuracy)
    
    # Calculate testmodel.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        predictions_test = torch.argmax(y_pred_test, axis=1).detach().numpy()
        correct_test = [1 if p == p_true else 0 for p, p_true in zip(predictions_test, y_test)]
        test_accuracy = sum(correct_test) / len(correct_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        test_acc_history.append(test_accuracy)
    


# Plotting the training and testing accuracies
ax = plt.subplot(111)
ax.plot(train_acc_history, linestyle='-', marker='^', label='Train Accuracy', color='C1')
ax.plot(test_acc_history, linestyle='--', marker='o', label='Test Accuracy', color='C2')
ax.set(title='Training and Testing Accuracy over Epochs', xlabel='Epoch', ylabel='Accuracy')
# ax.set_yscale('log')
# Limit Y-axis and X-axis
ax.set_ylim(0.85, 1.01)
ax.set_xlim(0, 50)
ax.grid()
ax.legend(bbox_to_anchor=(0.65,0.75), fontsize=8.5)
ax.axhline(1, ls='--', lw=2, c='m')
ax.text(35, 0.98, '100% train accuracy', c='m')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

# Create the inset of zoomed-out view
axins = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=3)

# Plot the zoomed-out view on the inset
axins.plot(train_acc_history, linestyle='-', marker='^', color='C1')
axins.plot(test_acc_history, linestyle='--', marker='o', color='C2')

# Set limits for the inset (zoomed-out view)
axins.set_xlim(0, len(train_acc_history))
axins.set_ylim(0.85, 1.01)

# Add gridlines and remove ticks to make the inset cleaner
axins.grid(False)
axins.tick_params(left=False, bottom=False)  # Optional: remove ticks on the inset


plt.show()


plt.savefig('plt3.png')




