from __future__ import annotations

import unittest

import numpy as np

from sim_vitals.model import (
    ArrayConfig,
    RadarConfig,
    SIMSystem,
    estimate_vital_rates,
    morphology_signature,
)
from sim_vitals.optimization import quantize_phases
from sim_vitals.sensitivity import remap_phase_bank


class SimulationTests(unittest.TestCase):
    def test_adjoint_gradient_matches_finite_difference(self) -> None:
        system = SIMSystem(ArrayConfig(nx=3, ny=3, layers=2), RadarConfig())
        rng = np.random.default_rng(4)
        phases = rng.normal(size=(2, 9))
        field = rng.normal(size=9) + 1j * rng.normal(size=9)
        field /= np.linalg.norm(field)
        _, gradient = system.power_and_gradient(phases, field)
        epsilon = 1.0e-6
        for layer, element in [(0, 0), (0, 5), (1, 3), (1, 8)]:
            plus, minus = phases.copy(), phases.copy()
            plus[layer, element] += epsilon
            minus[layer, element] -= epsilon
            p_plus = abs(system.gain(plus, field)) ** 2
            p_minus = abs(system.gain(minus, field)) ** 2
            numerical = (p_plus - p_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(gradient[layer, element], numerical, places=6)

    def test_clean_vital_rate_estimation(self) -> None:
        radar = RadarConfig(duration_s=40.0)
        t = np.arange(int(radar.sample_rate_hz * radar.duration_s)) / radar.sample_rate_hz
        displacement = 3.0e-3 * np.sin(2 * np.pi * 0.27 * t)
        displacement += 0.25e-3 * np.sin(2 * np.pi * 1.23 * t)
        received = np.exp(1j * 4.0 * np.pi / radar.wavelength_m * displacement)
        respiration, heart, _, _ = estimate_vital_rates(received, radar)
        self.assertLess(abs(respiration - 0.27 * 60.0), 0.2)
        self.assertLess(abs(heart - 1.23 * 60.0), 0.2)

    def test_morphology_signature_is_normalised(self) -> None:
        system = SIMSystem()
        for bmi in (18.0, 25.0, 32.0, 40.0):
            self.assertAlmostEqual(np.linalg.norm(morphology_signature(system, bmi)), 1.0)

    def test_phase_quantisation(self) -> None:
        phases = np.linspace(-np.pi, np.pi, 19)
        quantised = quantize_phases(phases, bits=2)
        self.assertLessEqual(np.unique(np.round(quantised, 12)).size, 4)

    def test_bmi_mismatch_remaps_only_controller_code(self) -> None:
        bank = {18.0: np.array([[18.0]]), 22.0: np.array([[22.0]]), 26.0: np.array([[26.0]])}
        remapped = remap_phase_bank(bank, np.array([18.0, 22.0]), bmi_offset=4.0)
        self.assertEqual(float(remapped[18.0][0, 0]), 22.0)
        self.assertEqual(float(remapped[22.0][0, 0]), 26.0)


if __name__ == "__main__":
    unittest.main()
