"""Pipeline B — Quantum head (Qiskit, simulator only).

1. Angle encoding: 8 bottleneck values -> RY rotation on 8 qubits.
2. Variational ansatz: EfficientSU2(8 qubits, reps=2-3).
3. Measurement: Pauli-Z expectation value per qubit -> 8 classical numbers.
4. Post-processing: Linear(8->1) -> Sigmoid.

Wrapped as an EstimatorQNN + TorchConnector so it drops into the same
PyTorch training loop (same optimizer, same loss, same epochs) as the
classical head. Runs on Qiskit's classical simulator — no real quantum
hardware, no speed benefit; the comparison is accuracy only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


def build_qnn(num_qubits: int = 8, reps: int = 3) -> EstimatorQNN:
    input_params = ParameterVector("x", num_qubits)
    encoding = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        encoding.ry(input_params[i], i)

    ansatz = EfficientSU2(num_qubits, reps=reps, entanglement="circular")

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(encoding, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=list(input_params),
        weight_params=list(ansatz.parameters),
        observables=observables,
    )
    return qnn


class QuantumHead(nn.Module):
    def __init__(self, num_qubits: int = 8, reps: int = 3):
        super().__init__()
        qnn = build_qnn(num_qubits, reps)
        self.quantum_layer = TorchConnector(qnn)
        self.post_process = nn.Sequential(nn.Linear(num_qubits, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expectations = self.quantum_layer(x)
        return self.post_process(expectations).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
