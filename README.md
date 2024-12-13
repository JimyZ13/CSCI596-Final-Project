# CSCI 596 Final project

For this project, I am exploring how basic Quantum Computing circuits can be implemented with the help of HPC. Here are the components to this repo:

### Quantum Fourier Transform (qft.py and qft_mpi.py)

This part explores how the quantum fourier transform (qft) can be efficiently simulated. qft.py is a single threaded version of the qft and qft_mpi.py utilizes mpi and openmp to parallellize simulation. The number of simulations is equal to the number of mpi ranks and each mpi rank have a number of omp threads to parallelize vector matrix operations.

### Quantum GNN (qgan.ipynb)

This is copied from Qiskit's sample where a quantum state is used as a generative neural network. The quantum state is tunned with a classical neural net which tries to distinguish the probability distribution sampled from the quantum stat with a given distribution. 

### Quantum SVM (qsvm.py)

This is an work in progress. This code is meant to demonstrate how support vector machines are efficiently simulated on HPC. 

---

## QFT Usage:

```bash
pip install numpy numba mpi4py

# Usage
# Run with command-line arguments to specify the number of qubits:
python qft_mpi.py --qubits 3

# Run with MPI for distributed execution:
mpirun -n <number_of_processes> python qft_mpi.py --qubits <number_of_qubits>

# Example using MPI:
mpirun -n 4 python qft_mpi.py --qubits 3
