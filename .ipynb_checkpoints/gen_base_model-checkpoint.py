import argparse

from pennylane import broadcast
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as np

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument("--model",  default=None,help='model name')
    parser.add_argument("--expName",  default=None,help='(optional) ')

    # IO paths
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--outPath",default='out/',help="(optional) redirect all outputs ")
    
    args = parser.parse_args()
    return args

# this example only has one layer

import pennylane as qml
import numpy as np

n_qubits = 10
n_layers = 1  
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


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    if args.model == 'base':
        inputs = np.random.rand(n_qubits)
        weights = np.random.rand(n_layers + 1, n_qubits, 3)  # +1 for the final Rot layer
        x = np.pi / 6  # Example parameter for RX
        z = np.pi / 4  # Example parameter for CRZ

        fig, ax = qml.draw_mpl(qnode, show_all_wires=True, decimals=2, level="device")(inputs,weights,x,z)
        fig.show()
      
        return qnode
    # outF=MD['short_name']
    # fullN=os.path.join(args.outPath,outF)
    # write4_data_hdf5(bigD,fullN,metaD=MD)

    pprint(MD)

    print('local sim for 4 gpus:\n mpirun -np 4  ./run_model.py --circName -n 300   \n')
    print('M:done')