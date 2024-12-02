import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning import QuantumKernel

from qiskit.primitives import Sampler

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = 2 
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2)

sampler = Sampler()

quantum_kernel = QuantumKernel(feature_map=feature_map, sampler=sampler)

kernel_matrix_train = quantum_kernel.evaluate(X_train)
kernel_matrix_test = quantum_kernel.evaluate(X_test, X_train)

qsvm = SVC(kernel='precomputed')
qsvm.fit(kernel_matrix_train, y_train)

y_pred = qsvm.predict(kernel_matrix_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
