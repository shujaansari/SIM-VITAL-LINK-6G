"""Adjoint optimization for phase-only stacked intelligent metasurfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import SIMSystem


@dataclass
class OptimizationResult:
    phases: np.ndarray
    objective_history: np.ndarray
    sir_history_db: np.ndarray


def objective_and_gradient(
    system: SIMSystem,
    phases: np.ndarray,
    target_fields: np.ndarray,
    target_weights: np.ndarray,
    clutter_fields: np.ndarray,
    clutter_powers: np.ndarray,
    noise_power: float = 0.004,
) -> tuple[float, np.ndarray, float]:
    """Mean log desired-to-interference-and-noise ratio and gradient."""

    target_fields = np.atleast_2d(target_fields)
    target_weights = np.asarray(target_weights, dtype=float)
    target_terms: list[float] = []
    target_gradients: list[np.ndarray] = []
    for field, weight in zip(target_fields, target_weights, strict=True):
        power, gradient = system.power_and_gradient(phases, field)
        weighted_power = max(float(weight * power), 1.0e-15)
        target_terms.append(weighted_power)
        target_gradients.append(weight * gradient)

    interference = float(noise_power)
    interference_gradient = np.zeros_like(phases)
    for field, power_weight in zip(clutter_fields, clutter_powers, strict=True):
        power, gradient = system.power_and_gradient(phases, field)
        interference += float(power_weight * power)
        interference_gradient += power_weight * gradient
    interference = max(interference, 1.0e-15)

    # Averaging log-SINR rather than raw target power prevents low-BMI cases
    # from dominating universal training solely through their larger amplitude.
    objective = float(np.mean(np.log(target_terms)) - np.log(interference))
    gradient = np.mean(
        [g / p for g, p in zip(target_gradients, target_terms, strict=True)], axis=0
    ) - interference_gradient / interference
    geometric_target = float(np.exp(np.mean(np.log(target_terms))))
    sir_db = 10.0 * np.log10(geometric_target / interference)
    return objective, gradient, float(sir_db)


def optimize_phases(
    system: SIMSystem,
    target_fields: np.ndarray,
    target_weights: np.ndarray,
    clutter_fields: np.ndarray,
    clutter_powers: np.ndarray,
    iterations: int = 180,
    learning_rate: float = 0.045,
    noise_power: float = 0.004,
    seed: int = 7,
) -> OptimizationResult:
    """Maximise the spatial log-SINR using Adam and exact wave gradients."""

    rng = np.random.default_rng(seed)
    phases = rng.normal(0.0, 0.08, size=(system.array.layers, system.array.elements))
    first_moment = np.zeros_like(phases)
    second_moment = np.zeros_like(phases)
    beta1, beta2 = 0.9, 0.999
    objective_history = np.empty(iterations)
    sir_history = np.empty(iterations)

    for step in range(1, iterations + 1):
        objective, gradient, sir_db = objective_and_gradient(
            system,
            phases,
            target_fields,
            target_weights,
            clutter_fields,
            clutter_powers,
            noise_power,
        )
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 20.0:
            gradient *= 20.0 / gradient_norm
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        corrected_moment = first_moment / (1.0 - beta1**step)
        corrected_variance = second_moment / (1.0 - beta2**step)
        phases += learning_rate * corrected_moment / (
            np.sqrt(corrected_variance) + 1.0e-8
        )
        phases = (phases + np.pi) % (2.0 * np.pi) - np.pi
        objective_history[step - 1] = objective
        sir_history[step - 1] = sir_db

    # Store the objective at the returned parameter point, not one step behind.
    objective_history[-1], _, sir_history[-1] = objective_and_gradient(
        system,
        phases,
        target_fields,
        target_weights,
        clutter_fields,
        clutter_powers,
        noise_power,
    )
    return OptimizationResult(phases, objective_history, sir_history)


def quantize_phases(phases: np.ndarray, bits: int) -> np.ndarray:
    """Quantise phases to a practical B-bit control alphabet."""

    if bits < 1:
        raise ValueError("bits must be at least 1")
    levels = 2**bits
    wrapped = phases % (2.0 * np.pi)
    quantized = np.round(wrapped / (2.0 * np.pi) * levels) % levels
    return quantized * (2.0 * np.pi / levels)
