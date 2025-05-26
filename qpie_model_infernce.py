#!/usr/bin/env python3

import cudaq
from cudaq import spin
import numpy as np
import torch

torch.manual_seed(22)
cudaq.set_random_seed(44)
cudaq.set_target("nvidia")

def create_simple_qpie_kernel():
    """Create a simplified QPIE kernel for testing"""
    
    @cudaq.kernel
    def qpie_kernel(data_qubits: int, ansatz_depth: int,
                   inputs: list[float], weights: list[float]):
        
        qubits = cudaq.qvector(data_qubits)
        
        # Initialize with Hadamard gates
        for i in range(data_qubits):
            h(qubits[i])
        
        weight_idx = 0
        
        # Main ansatz loop
        for layer in range(ansatz_depth):
            # Data encoding
            for i in range(data_qubits):
                if i < len(inputs):
                    input_val = inputs[i]
                    if layer % 2 == 0:
                        ry(input_val, qubits[i])
                    else:
                        rx(input_val, qubits[i])
            
            # Entangling layer - simplified
            for i in range(data_qubits - 1):
                if weight_idx + 1 < len(weights):
                    # Simple CX + RZ pattern
                    cx(qubits[i], qubits[i + 1])
                    rz(weights[weight_idx], qubits[i + 1])
                    cx(qubits[i], qubits[i + 1])
                    weight_idx += 1
            
            # Variational layer
            for i in range(data_qubits):
                if weight_idx + 2 < len(weights):
                    rx(weights[weight_idx], qubits[i])
                    ry(weights[weight_idx + 1], qubits[i])
                    rz(weights[weight_idx + 2], qubits[i])
                    weight_idx += 3
    
    return qpie_kernel

def test_kernel_creation():
    """Test kernel creation and basic functionality"""
    
    print("Creating QPIE kernel...")
    kernel = create_simple_qpie_kernel()
    print("✓ Kernel created successfully!")
    
    # Test parameters
    data_qubits = 4  # Small number for testing
    ansatz_depth = 2
    
    # Create test data
    inputs = [0.1, 0.2, 0.3, 0.4]  # Simple test inputs
    weights = [0.5] * (ansatz_depth * (data_qubits * 3 + data_qubits - 1))  # Enough weights
    
    print(f"Data qubits: {data_qubits}")
    print(f"Ansatz depth: {ansatz_depth}")
    print(f"Input size: {len(inputs)}")
    print(f"Weight size: {len(weights)}")
    
    # Create Hamiltonian for observation
    hamiltonian = sum([spin.z(i) for i in range(data_qubits)])
    print(f"Hamiltonian: {hamiltonian}")
    
    try:
        # Test the kernel with cudaq.observe (without measurements in kernel)
        print("\nTesting quantum circuit execution...")
        result = cudaq.observe(kernel, hamiltonian, 
                              data_qubits, ansatz_depth, inputs, weights)
        
        expectation_value = result.expectation()
        print(f"✓ Circuit executed successfully!")
        print(f"Expectation value: {expectation_value}")
        
        # Test sampling
        print("\nTesting sampling...")
        sample_result = cudaq.sample(kernel, data_qubits, ansatz_depth, inputs, weights)
        print(f"✓ Sampling successful!")
        print(f"Sample counts: {sample_result}")
        
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        return False
    
    return True

def test_parameterized_circuit():
    """Test with different parameter sets"""
    
    print("\n" + "="*50)
    print("Testing parameterized circuit variations...")
    
    kernel = create_simple_qpie_kernel()
    data_qubits = 3
    ansatz_depth = 1
    
    # Test different input patterns
    test_cases = [
        ([0.0, 0.0, 0.0], "Zero inputs"),
        ([np.pi/4, np.pi/4, np.pi/4], "π/4 inputs"),
        ([np.pi/2, np.pi/3, np.pi/6], "Mixed inputs"),
    ]
    
    hamiltonian = sum([spin.z(i) for i in range(data_qubits)])
    
    for inputs, description in test_cases:
        weights = [0.1] * (ansatz_depth * (data_qubits * 3 + data_qubits - 1))
        
        try:
            result = cudaq.observe(kernel, hamiltonian, 
                                  data_qubits, ansatz_depth, inputs, weights)
            expectation = result.expectation()
            print(f"✓ {description}: {expectation:.4f}")
            
        except Exception as e:
            print(f"✗ {description} failed: {e}")

def test_different_hamiltonians():
    """Test with different Hamiltonian observables"""
    
    print("\n" + "="*50)
    print("Testing different Hamiltonians...")
    
    kernel = create_simple_qpie_kernel()
    data_qubits = 3
    ansatz_depth = 1
    inputs = [0.1, 0.2, 0.3]
    weights = [0.5] * (ansatz_depth * (data_qubits * 3 + data_qubits - 1))
    
    # Different Hamiltonians to test
    hamiltonians = [
        (spin.z(0), "Single Z(0)"),
        (spin.x(1), "Single X(1)"),
        (spin.z(0) + spin.z(1), "Z(0) + Z(1)"),
        (spin.x(0) + spin.y(1) + spin.z(2), "X(0) + Y(1) + Z(2)"),
        (sum([spin.z(i) for i in range(data_qubits)]), "Sum of all Z"),
    ]
    
    for hamiltonian, description in hamiltonians:
        try:
            result = cudaq.observe(kernel, hamiltonian, 
                                  data_qubits, ansatz_depth, inputs, weights)
            expectation = result.expectation()
            print(f"✓ {description}: {expectation:.4f}")
            
        except Exception as e:
            print(f"✗ {description} failed: {e}")

def main():
    """Main test function"""
    
    print("QPIE CUDA-Q Kernel Dry Run Test")
    print("="*50)
    
    # Test basic kernel creation and execution
    success = test_kernel_creation()
    
    if success:
        # Run additional tests
        test_parameterized_circuit()
        test_different_hamiltonians()
        
        print("\n" + "="*50)
        print("✓ All tests completed successfully!")
        
        # Show what gates are being used
        print("\nGates used in the kernel:")
        print("- h (Hadamard)")
        print("- rx, ry, rz (Pauli rotations)")
        print("- cx (Controlled-X)")
        print("\nThis kernel can be integrated into a PyTorch model for training.")
        
    else:
        print("\n" + "="*50)
        print("✗ Basic tests failed. Check CUDA-Q installation and setup.")

if __name__ == "__main__":
    main()