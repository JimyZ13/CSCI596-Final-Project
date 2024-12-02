import numpy as np

class Qubit:
    def __init__(self, state=None):
        if state is None:
            self.state = np.array([[1], [0]], dtype=complex)
        else:
            self.state = np.array(state, dtype=complex)

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = Qubit().state
        for _ in range(1, num_qubits):
            self.state = np.kron(self.state, Qubit().state)

    def apply_gate(self, gate, target_qubit):
        full_gate = np.eye(1)
        for i in range(self.num_qubits):
            if i == target_qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        self.state = np.dot(full_gate, self.state)

    def apply_controlled_gate(self, gate, control, target):
        size = 2 ** self.num_qubits
        full_gate = np.eye(size, dtype=complex)
        for i in range(size):
            if (i >> (self.num_qubits - 1 - control)) & 1:
                index = i ^ (1 << (self.num_qubits - 1 - target))
                full_gate[i, i] = 0
                full_gate[i, index] = 1
        self.state = np.dot(full_gate, self.state)

    def apply_cnot(self, control, target):
        self.apply_controlled_gate(pauli_x(), control, target)

    def qft(self):
        for i in range(self.num_qubits):
            self.apply_gate(hadamard(), i)
            for j in range(i + 1, self.num_qubits):
                angle = np.pi / (2 ** (j - i))
                self.apply_controlled_gate(phase_shift(angle), j, i)
        for i in range(self.num_qubits // 2):
            self.swap_qubits(i, self.num_qubits - i - 1)

    def swap_qubits(self, q1, q2):
        self.state = np.dot(self.swap_gate(q1, q2), self.state)

    def swap_gate(self, q1, q2):
        size = 2 ** self.num_qubits
        full_gate = np.eye(size, dtype=complex)
        for i in range(size):
            swapped_index = i ^ ((i >> (self.num_qubits - 1 - q1)) & 1) << (self.num_qubits - 1 - q2)
            swapped_index ^= ((i >> (self.num_qubits - 1 - q2)) & 1) << (self.num_qubits - 1 - q1)
            full_gate[i, swapped_index] = 1
        return full_gate

    def measure(self):
        probabilities = np.abs(self.state) ** 2
        probabilities /= np.sum(probabilities)  # Normalize the probabilities
        measured_state_index = np.random.choice(range(len(self.state)), p=probabilities.flatten())
        measured_state = bin(measured_state_index)[2:].zfill(self.num_qubits)
        return f"|{measured_state}>"


def hadamard():
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def phase_shift(angle):
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)

if __name__ == "__main__":
    for i in range(5):
        qc = QuantumCircuit(5)
        qc.qft()
        result = qc.measure()
        print(f"QFT result {i}: {result}")
