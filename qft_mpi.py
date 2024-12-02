import numpy as np
from mpi4py import MPI
from numba import njit, prange

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

class Qubit:
    def __init__(self, state=None):
        if state is None:
            self.state = np.array([[1], [0]], dtype=complex)
        else:
            self.state = np.array(state, dtype=complex)

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.size = 2 ** num_qubits
        self.local_size = self.size // size_mpi
        self.local_start = rank * self.local_size
        self.local_end = self.local_start + self.local_size if rank != size_mpi - 1 else self.size
        self.state = np.zeros(self.local_end - self.local_start, dtype=complex)
        if rank == 0:
            self.state[0] = 1.0

    def apply_gate(self, gate, target_qubit):
        self.state = apply_gate_numba(self.state, gate, target_qubit, self.num_qubits, self.local_start, self.local_end)

    def apply_controlled_gate(self, gate, control, target):
        self.state = apply_controlled_gate_numba(self.state, gate, control, target, self.num_qubits, self.local_start, self.local_end)

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
        self.state = swap_gate_numba(self.state, q1, q2, self.num_qubits, self.local_start, self.local_end)

    def measure(self):
        # Gather state from all processes
        full_state = None
        if rank == 0:
            full_state = np.empty(self.size, dtype=complex)
        comm.Gather(self.state, full_state, root=0)

        if rank == 0:
            probabilities = np.abs(full_state) ** 2
            probabilities /= np.sum(probabilities)  # Normalize the probabilities
            measured_state_index = np.random.choice(range(len(full_state)), p=probabilities)
            measured_state = bin(measured_state_index)[2:].zfill(self.num_qubits)
            return f"|{measured_state}>"
        else:
            return None

@njit(parallel=True)
def apply_gate_numba(state, gate, target_qubit, num_qubits, local_start, local_end):
    size = local_end - local_start
    new_state = np.zeros_like(state)
    for idx in prange(size):
        i = idx + local_start
        bit = (i >> (num_qubits - 1 - target_qubit)) & 1
        flipped_i = i ^ (1 << (num_qubits - 1 - target_qubit))
        if local_start <= flipped_i < local_end:
            flipped_idx = flipped_i - local_start
            new_state[idx] += gate[bit, bit] * state[idx] + gate[bit, 1 - bit] * state[flipped_idx]
        else:
            # Communication with other processes
            partner_rank = flipped_i // size
            partner_idx = flipped_i % size
            partner_value = np.zeros(1, dtype=complex)
            comm.Sendrecv(state[idx:idx+1], dest=partner_rank, recvbuf=partner_value, source=partner_rank)
            new_state[idx] += gate[bit, bit] * state[idx] + gate[bit, 1 - bit] * partner_value[0]
    return new_state

@njit(parallel=True)
def apply_controlled_gate_numba(state, gate, control, target, num_qubits, local_start, local_end):
    size = local_end - local_start
    new_state = np.copy(state)
    for idx in prange(size):
        i = idx + local_start
        control_bit = (i >> (num_qubits - 1 - control)) & 1
        target_bit = (i >> (num_qubits - 1 - target)) & 1
        if control_bit:
            flipped_i = i ^ (1 << (num_qubits - 1 - target))
            if local_start <= flipped_i < local_end:
                flipped_idx = flipped_i - local_start
                new_state[idx] = gate[target_bit, 0] * state[idx] + gate[target_bit, 1] * state[flipped_idx]
            else:
                partner_rank = flipped_i // size
                partner_idx = flipped_i % size
                partner_value = np.zeros(1, dtype=complex)
                comm.Sendrecv(state[idx:idx+1], dest=partner_rank, recvbuf=partner_value, source=partner_rank)
                new_state[idx] = gate[target_bit, 0] * state[idx] + gate[target_bit, 1] * partner_value[0]
    return new_state

@njit(parallel=True)
def swap_gate_numba(state, q1, q2, num_qubits, local_start, local_end):
    size = local_end - local_start
    new_state = np.copy(state)
    for idx in prange(size):
        i = idx + local_start
        bit1 = (i >> (num_qubits - 1 - q1)) & 1
        bit2 = (i >> (num_qubits - 1 - q2)) & 1
        if bit1 != bit2:
            swapped_i = i ^ ((1 << (num_qubits - 1 - q1)) | (1 << (num_qubits - 1 - q2)))
            if local_start <= swapped_i < local_end:
                swapped_idx = swapped_i - local_start
                new_state[idx], new_state[swapped_idx] = new_state[swapped_idx], new_state[idx]
            else:
                partner_rank = swapped_i // size
                partner_idx = swapped_i % size
                partner_value = np.zeros(1, dtype=complex)
                comm.Sendrecv(new_state[idx:idx+1], dest=partner_rank, recvbuf=partner_value, source=partner_rank)
                send_buffer = np.copy(new_state[idx])
                new_state[idx] = partner_value[0]
                comm.Sendrecv(send_buffer, dest=partner_rank, recvbuf=partner_value, source=partner_rank)
    return new_state

def hadamard():
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def phase_shift(angle):
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)

if __name__ == "__main__":
    qc = QuantumCircuit(5)
    qc.qft()
    result = qc.measure()
    if rank == 0 and result is not None:
        print(f"QFT result: {result}")
