import unittest

import numpy as np
from qiskit.quantum_info import Statevector, entropy, partial_trace

from src.backbone.quantum_kernel import circuit, states, fidelity_kernel


class QuantumKernelTests(unittest.TestCase):
    def test_kernel_is_normalized_psd_and_swap_invariant(self):
        pairs = np.random.default_rng(17).normal(size=(5, 2, 4))
        encoded = states(pairs)
        gram = fidelity_kernel(encoded, encoded)
        np.testing.assert_allclose(gram, gram.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(gram).min(), -1e-10)
        swapped = states(pairs[:, ::-1])
        np.testing.assert_allclose(fidelity_kernel(swapped, encoded), gram, atol=1e-10)

    def test_state_overlap_agrees_with_qiskit_inverse_circuit(self):
        pair_a, pair_b = np.random.default_rng(3).normal(size=(2, 2, 4))
        ca, cb = circuit(pair_a), circuit(pair_b)
        expected = abs(Statevector.from_instruction(ca).inner(Statevector.from_instruction(cb))) ** 2
        overlap = cb.compose(ca.inverse())
        actual = Statevector.from_instruction(overlap).probabilities()[0]
        self.assertAlmostEqual(expected, actual, places=12)

    def test_ablation_removes_cross_register_entanglement(self):
        pair = np.random.default_rng(4).normal(size=(2, 4))
        no_cross = Statevector.from_instruction(circuit(pair, entangle=False))
        coupled = Statevector.from_instruction(circuit(pair, entangle=True))
        self.assertLess(abs(entropy(partial_trace(no_cross, [4, 5, 6, 7]))), 1e-10)
        self.assertGreater(entropy(partial_trace(coupled, [4, 5, 6, 7])), 0.01)


if __name__ == '__main__':
    unittest.main()
