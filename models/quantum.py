import torch
import torch.nn as nn
import pennylane as qml

class QuantumLayer(nn.Module):
    """4-qubit VQC with manual batching. Runs on CPU (quantum sim)."""

    def __init__(self, n_qubits=4, n_layers=2, init_std=0.01):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * init_std)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def _circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self._circuit = _circuit

    def forward(self, x):
        x_cpu = x.cpu()
        batch_out = []
        for i in range(x_cpu.shape[0]):
            out = self._circuit(x_cpu[i], self.weights)
            batch_out.append(torch.stack(list(out)))
        result = torch.stack(batch_out).float()
        return result.to(x.device)
