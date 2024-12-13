from mpi4py import MPI
import numpy as np
from numba import njit, prange
import argparse
import time

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = self.initialize_state(num_qubits)

    def initialize_state(self, num_qubits):
        state = np.array([[1], [0]], dtype=np.complex128)
        for _ in range(1, num_qubits):
            state = np.kron(state, np.array([[1], [0]], dtype=np.complex128))
        return state

    @staticmethod
    @njit(parallel=True)
    def parallel_matrix_vector_mult(matrix, vector):
        """Efficient parallel matrix-vector multiplication."""
        matrix = np.ascontiguousarray(matrix)
        vector = np.ascontiguousarray(vector)
        
        result = np.zeros(matrix.shape[0], dtype=np.complex128)
        
        for i in prange(matrix.shape[0]):
            result[i] = np.dot(matrix[i, :], vector)
        return result

    def apply_gate(self, gate, target_qubit):
        """Apply a single-qubit gate to the target qubit."""
        full_gate = np.eye(1, dtype=np.complex128)
        for i in range(self.num_qubits):
            if i == target_qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2, dtype=np.complex128))
        self.state = self.parallel_matrix_vector_mult(full_gate, self.state.flatten())

    def measure(self):
        """Measure the current quantum state."""
        probabilities = np.abs(self.state.flatten()) ** 2
        probabilities /= np.sum(probabilities)
        measured_state_index = np.random.choice(range(len(self.state)), p=probabilities.flatten())
        measured_state = bin(measured_state_index)[2:].zfill(self.num_qubits)
        return f"|{measured_state}>"

def hadamard():
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Quantum Circuit Simulation with MPI")
    parser.add_argument("--qubits", type=int, required=True, help="Number of qubits in the circuit")
    args = parser.parse_args()

    if rank == 0:
        start_time = time.time()

    qc = QuantumCircuit(args.qubits)
    qc.apply_gate(hadamard(), 0)  
    result = qc.measure()       

    all_results = comm.gather(result, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"Measured States from all processes: {all_results}")
        print(f"Execution Time: {end_time - start_time:.6f} seconds")
