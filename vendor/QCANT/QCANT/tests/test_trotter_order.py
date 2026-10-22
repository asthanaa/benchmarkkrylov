"""Product-formula checks on synthetic operators; no molecular calculations."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

try:
    import qulacs
except ImportError:
    raise unittest.SkipTest("Qulacs is not installed")
from qulacs import gate as qg

from QCANT import qulacs_accel as qa


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0])
TERMS = [{"indices": [0], "pauli_ids": [pauli], "coeff": coeff}
         for pauli, coeff in ((1, 0.7), (2, -0.3), (3, 0.4))]
H = 0.7 * X - 0.3 * Y + 0.4 * Z


def circuit(dt, steps=1, **kwargs):
    return qa._build_time_evolution_circuit(
        1, TERMS, dt, steps, qg, qulacs.QuantumCircuit, np, **kwargs)


def matrix(circuit):
    columns = []
    for index in range(2):
        state = qulacs.QuantumState(1)
        state.set_computational_basis(index)
        circuit.update_quantum_state(state)
        columns.append(state.get_vector())
    return np.column_stack(columns)


class TrotterOrderTests(unittest.TestCase):
    def test_first_order_default_keeps_original_gate_sequence(self):
        dt = 0.2
        expected = expm(-1j * 0.4 * Z * dt) @ expm(1j * 0.3 * Y * dt) @ expm(-1j * 0.7 * X * dt)
        np.testing.assert_allclose(matrix(circuit(dt)), expected, atol=1e-14)
        np.testing.assert_allclose(matrix(circuit(dt)), matrix(circuit(dt, trotter_order=1)), atol=1e-14)

    def test_second_order_is_the_symmetric_half_step_formula(self):
        dt = 0.2
        a = expm(-1j * 0.7 * X * dt / 2)
        b = expm(1j * 0.3 * Y * dt / 2)
        c = expm(-1j * 0.4 * Z * dt)
        np.testing.assert_allclose(matrix(circuit(dt, trotter_order=2)), a @ b @ c @ b @ a, atol=1e-14)

    def test_second_order_is_time_reversible(self):
        for steps in (1, 3):
            with self.subTest(steps=steps):
                forward = matrix(circuit(0.3, steps, trotter_order=2))
                backward = matrix(circuit(-0.3, steps, trotter_order=2))
                np.testing.assert_allclose(backward @ forward, np.eye(2), atol=1e-14)

    def test_single_interval_error_has_the_expected_order(self):
        for order, ratio in ((1, 4), (2, 8)):
            with self.subTest(order=order):
                errors = [np.linalg.norm(matrix(circuit(dt, trotter_order=order)) - expm(-1j * H * dt))
                          for dt in (0.12, 0.06)]
                self.assertAlmostEqual(errors[0] / errors[1], ratio, delta=0.04 * ratio)

    def test_more_slices_improve_accuracy_at_fixed_total_time(self):
        exact = expm(-1j * H * 0.2)
        for order, ratio in ((1, 2), (2, 4)):
            with self.subTest(order=order):
                errors = [np.linalg.norm(matrix(circuit(0.2, steps, trotter_order=order)) - exact)
                          for steps in (1, 2)]
                self.assertAlmostEqual(errors[0] / errors[1], ratio, delta=0.04 * ratio)

    def test_commuting_terms_are_exact_and_identity_is_only_a_global_phase(self):
        terms = [{"indices": [], "pauli_ids": [], "coeff": 7.0},
                 {"indices": [0], "pauli_ids": [3], "coeff": 0.3},
                 {"indices": [0], "pauli_ids": [3], "coeff": -0.1}]
        result = qa._build_time_evolution_circuit(
            1, terms, 0.2, 3, qg, qulacs.QuantumCircuit, np, trotter_order=2)
        np.testing.assert_allclose(matrix(result), expm(-1j * 0.2 * Z * 0.2), atol=1e-14)
        self.assertEqual(circuit(0.0, trotter_order=2).get_gate_count(), 0)

    def test_invalid_order_is_rejected_before_hamiltonian_construction(self):
        for order in (0, 3):
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, "trotter_order"):
                    circuit(0.2, trotter_order=order)
                with patch.object(qa, "_build_hamiltonian_payload") as builder:
                    with self.assertRaisesRegex(ValueError, "trotter_order"):
                        qa.qrte_qulacs(["H"], np.zeros((1, 3)), delta_t=0.2, n_steps=1,
                                       active_electrons=1, active_orbitals=1, trotter_order=order)
                    builder.assert_not_called()

    def test_qrte_passes_second_order_and_repeats_the_symmetric_propagator(self):
        observable = qulacs.Observable(1)
        for term in TERMS:
            observable.add_operator(term["coeff"], "XYZ"[term["pauli_ids"][0] - 1] + " 0")
        payload = {"np": np, "QuantumState": qulacs.QuantumState,
                   "QuantumCircuit": qulacs.QuantumCircuit, "qg": qg,
                   "expm_multiply": expm_multiply, "qubits": 1,
                   "h_qulacs": observable, "hf_bits": np.array([0]), "pauli_terms": TERMS}
        expected = matrix(circuit(0.2, 2, trotter_order=2))
        with patch.object(qa, "_build_hamiltonian_payload", return_value=payload), \
                patch.object(qa, "_build_time_evolution_circuit", wraps=qa._build_time_evolution_circuit) as build:
            energies, states, times, history = qa.qrte_qulacs(
                ["H"], np.zeros((1, 3)), delta_t=0.2, n_steps=2,
                active_electrons=1, active_orbitals=1, evolution_mode="trotter",
                trotter_order=2, trotter_steps=2, max_workers=1, return_min_energy_history=True)
            self.assertEqual(build.call_args.kwargs["trotter_order"], 2)
        np.testing.assert_allclose(states[1], expected[:, 0], atol=1e-14)
        np.testing.assert_allclose(states[2], (expected @ expected)[:, 0], atol=1e-14)
        np.testing.assert_allclose(times, [0, 0.2, 0.4], atol=1e-14)
        self.assertTrue(np.all(np.isfinite(energies)) and np.all(np.isfinite(history)))


if __name__ == "__main__":
    unittest.main()
