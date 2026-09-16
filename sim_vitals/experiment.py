"""End-to-end training, Monte Carlo evaluation and paper-ready output."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Keep plotting caches inside a writable temporary location in restricted and CI
# environments.  These variables must be set before Matplotlib is imported.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim_vitals_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sim_vitals_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import (
    ArrayConfig,
    RadarConfig,
    SIMSystem,
    clutter_signatures,
    estimate_vital_rates,
    morphology_scalars,
    morphology_signature,
    physiological_displacement,
)
from .optimization import OptimizationResult, optimize_phases


@dataclass(frozen=True)
class Method:
    name: str
    system: SIMSystem | None
    phase_bank: dict[float, np.ndarray] | None
    color: str
    digital_bank: dict[float, np.ndarray] | None = None


def nearest_phase(phases: dict[float, np.ndarray], bmi: float) -> np.ndarray:
    key = min(phases, key=lambda value: abs(value - bmi))
    return phases[key]


def training_fields(
    system: SIMSystem,
    bmis: np.ndarray,
    rng: np.random.Generator,
    variants_per_bmi: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    fields, weights = [], []
    for bmi in bmis:
        amplitude = morphology_scalars(float(bmi))["reflection_amplitude"]
        for _ in range(variants_per_bmi):
            fields.append(
                morphology_signature(
                    system,
                    float(bmi),
                    lateral_jitter_deg=float(rng.normal(0.0, 1.2)),
                    elevation_jitter_deg=float(rng.normal(0.0, 0.7)),
                )
            )
            weights.append(amplitude**2)
    return np.stack(fields), np.asarray(weights)


def train_digital_beamformer(
    system: SIMSystem,
    bmis: np.ndarray,
    diagonal_loading: float = 0.004,
    seed: int = 2026,
) -> dict[float, np.ndarray]:
    """Create a robust BMI-conditioned digital beamformer bank."""

    clutter, clutter_power = clutter_signatures(system)
    covariance = diagonal_loading * np.eye(system.array.elements, dtype=complex)
    for field, power in zip(clutter, clutter_power, strict=True):
        covariance += power * np.outer(field, np.conj(field))
    bank: dict[float, np.ndarray] = {}
    for index, bmi in enumerate(bmis):
        fields, _ = training_fields(
            system,
            np.array([bmi]),
            np.random.default_rng(seed + index),
            variants_per_bmi=6,
        )
        target_covariance = np.mean(
            [np.outer(field, np.conj(field)) for field in fields], axis=0
        )
        # The generalized eigenvector supplies a strong initialization.  We then
        # optimize the same mean-log-SINR objective used for the adaptive SIM.
        operator = np.linalg.solve(covariance, target_covariance)
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        initial = eigenvectors[:, int(np.argmax(eigenvalues.real))]
        initial /= max(np.linalg.norm(initial), 1.0e-15)

        def objective_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray]:
            gains = fields @ np.conj(weights)
            target_powers = np.maximum(np.abs(gains) ** 2, 1.0e-15)
            interference = max(
                float(np.real(np.vdot(weights, covariance @ weights))), 1.0e-15
            )
            objective = float(np.mean(np.log(target_powers)) - np.log(interference))
            target_gradient = np.mean(
                fields * np.conj(gains)[:, None] / target_powers[:, None], axis=0
            )
            gradient = target_gradient - covariance @ weights / interference
            # Remove the radial component because only directions on the complex
            # unit sphere are admissible digital combiners.
            gradient -= weights * np.real(np.vdot(weights, gradient))
            return objective, gradient

        best_weights, best_objective = initial.copy(), -np.inf
        rng = np.random.default_rng(seed + 100 + index)
        for restart in range(3):
            weights = initial.copy() if restart == 0 else (
                rng.standard_normal(system.array.elements)
                + 1j * rng.standard_normal(system.array.elements)
            )
            weights /= max(np.linalg.norm(weights), 1.0e-15)
            first = np.zeros_like(weights)
            second = np.zeros(system.array.elements)
            for step in range(1, 251):
                _, gradient = objective_and_gradient(weights)
                first = 0.9 * first + 0.1 * gradient
                second = 0.999 * second + 0.001 * np.abs(gradient) ** 2
                direction = (first / (1.0 - 0.9**step)) / (
                    np.sqrt(second / (1.0 - 0.999**step)) + 1.0e-8
                )
                weights += 0.02 * direction
                weights /= max(np.linalg.norm(weights), 1.0e-15)
            objective, _ = objective_and_gradient(weights)
            if objective > best_objective:
                best_weights, best_objective = weights.copy(), objective
        bank[float(bmi)] = best_weights
    return bank


def train_methods(
    bmis: np.ndarray, iterations: int, seed: int
) -> tuple[list[Method], dict[str, OptimizationResult]]:
    radar = RadarConfig()
    sim3 = SIMSystem(ArrayConfig(layers=3), radar)
    sim1 = SIMSystem(ArrayConfig(layers=1), radar)
    clutter3, clutter_power = clutter_signatures(sim3)
    clutter1, _ = clutter_signatures(sim1)
    rng = np.random.default_rng(seed)
    histories: dict[str, OptimizationResult] = {}

    universal_fields, universal_weights = training_fields(sim3, bmis, rng)
    universal = optimize_phases(
        sim3,
        universal_fields,
        universal_weights,
        clutter3,
        clutter_power,
        iterations=iterations,
        seed=seed + 10,
    )
    histories["3-layer universal"] = universal

    adaptive3: dict[float, np.ndarray] = {}
    adaptive1: dict[float, np.ndarray] = {}
    for index, bmi in enumerate(bmis):
        fields3, weights3 = training_fields(
            sim3, np.array([bmi]), rng, variants_per_bmi=6
        )
        result3 = optimize_phases(
            sim3,
            fields3,
            weights3,
            clutter3,
            clutter_power,
            iterations=iterations,
            seed=seed + 100 + index,
        )
        adaptive3[float(bmi)] = result3.phases
        histories[f"3-layer adaptive BMI {bmi:g}"] = result3

        fields1, weights1 = training_fields(
            sim1, np.array([bmi]), rng, variants_per_bmi=6
        )
        result1 = optimize_phases(
            sim1,
            fields1,
            weights1,
            clutter1,
            clutter_power,
            iterations=iterations,
            seed=seed + 200 + index,
        )
        adaptive1[float(bmi)] = result1.phases

    methods = [
        Method("No intelligent surface", None, None, "#6b7280"),
        Method(
            "Digital MVDR array (BMI-adaptive)",
            sim3,
            None,
            "#8b5cf6",
            digital_bank=train_digital_beamformer(sim3, bmis, seed=seed + 300),
        ),
        Method("1-layer RIS (BMI-adaptive)", sim1, adaptive1, "#f59e0b"),
        Method(
            "3-layer SIM (universal)",
            sim3,
            {float(bmi): universal.phases for bmi in bmis},
            "#3b82f6",
        ),
        Method("3-layer SIM (BMI-adaptive)", sim3, adaptive3, "#10b981"),
    ]
    return methods, histories


def method_gains(
    method: Method,
    bmi: float,
    target_field: np.ndarray,
    clutter_fields: np.ndarray,
) -> tuple[complex, np.ndarray]:
    if method.digital_bank is not None:
        weights = nearest_phase(method.digital_bank, bmi)
        return np.vdot(weights, target_field), np.asarray(
            [np.vdot(weights, field) for field in clutter_fields]
        )
    if method.system is None:
        reference = SIMSystem(ArrayConfig(layers=3), RadarConfig())
        return reference.direct_gain(target_field), np.asarray(
            [reference.direct_gain(field) for field in clutter_fields]
        )
    if method.phase_bank is None:
        raise RuntimeError("a SIM/RIS method requires a phase bank")
    phases = nearest_phase(method.phase_bank, bmi)
    return method.system.gain(phases, target_field), np.asarray(
        [method.system.gain(phases, field) for field in clutter_fields]
    )


def spatial_sinr_db(
    target_gain: complex,
    clutter_gains: np.ndarray,
    target_amplitude: float,
    clutter_powers: np.ndarray,
    noise_power: float = 0.004,
) -> float:
    signal = target_amplitude**2 * abs(target_gain) ** 2
    interference = float(np.sum(clutter_powers * np.abs(clutter_gains) ** 2))
    return float(10.0 * np.log10(max(signal, 1e-15) / (interference + noise_power)))


def simulate_received_signal(
    method: Method,
    bmi: float,
    respiration_hz: float,
    heart_hz: float,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    radar = method.system.radar if method.system is not None else RadarConfig()
    reference_system = method.system or SIMSystem(ArrayConfig(layers=3), radar)
    target = morphology_signature(
        reference_system,
        bmi,
        lateral_jitter_deg=float(rng.normal(0.0, 1.5)),
        elevation_jitter_deg=float(rng.normal(0.0, 0.9)),
    )
    clutter, clutter_power = clutter_signatures(reference_system)
    target_gain, clutter_gain = method_gains(method, bmi, target, clutter)
    time = np.arange(int(radar.duration_s * radar.sample_rate_hz)) / radar.sample_rate_hz
    displacement, amplitudes = physiological_displacement(
        time, bmi, respiration_hz, heart_hz, rng
    )
    morphology = morphology_scalars(bmi)
    target_signal = morphology["reflection_amplitude"] * np.exp(
        1j * 4.0 * np.pi / radar.wavelength_m * displacement
    )

    nuisance_frequencies = np.array([0.04, 0.29, 1.08, 1.72, 0.21, 0.37, 1.31, 0.90])
    nuisance_motion_m = np.array(
        [1.8e-3, 1.1e-3, 0.22e-3, 0.14e-3, 0.7e-3, 0.6e-3, 0.18e-3, 0.17e-3]
    )
    nuisance = np.zeros_like(target_signal)
    for gain, power, frequency, motion in zip(
        clutter_gain, clutter_power, nuisance_frequencies, nuisance_motion_m, strict=True
    ):
        initial_phase = rng.uniform(0.0, 2.0 * np.pi)
        micro_motion = motion * np.sin(2.0 * np.pi * frequency * time + initial_phase)
        nuisance_component = np.exp(
            1j * 4.0 * np.pi / radar.wavelength_m * micro_motion
        )
        # Standard stationary-clutter/IQ-centre calibration removes the DC term
        # before vital-rate processing while retaining nuisance micro-motion.
        nuisance += np.sqrt(power) * gain * (
            nuisance_component - np.mean(nuisance_component)
        )

    clean = target_gain * target_signal + nuisance
    # SNR is referenced to the target power at one receiving element.  The SIM
    # can then realise collection/processing gain without receiving an artificial
    # noise penalty proportional to aperture size.
    reference_power = (
        morphology["reflection_amplitude"] ** 2 / reference_system.array.elements
    )
    noise_variance = reference_power * 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(time.size) + 1j * rng.standard_normal(time.size)
    )
    received = clean + noise
    diagnostics = {
        **amplitudes,
        "target_gain_power": float(abs(target_gain) ** 2),
        "spatial_sinr_db": spatial_sinr_db(
            target_gain,
            clutter_gain,
            morphology["reflection_amplitude"],
            clutter_power,
        ),
    }
    return received, diagnostics


def evaluate(
    methods: list[Method],
    bmis: np.ndarray,
    trials: int,
    snr_db: float,
    seed: int,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    example: dict[str, np.ndarray] = {}
    for bmi in bmis:
        for trial in range(trials):
            respiration_hz = float(rng.uniform(0.18, 0.42))
            heart_hz = float(rng.uniform(0.95, 1.65))
            # Use common trial seeds so methods see the same physiology/noise law.
            trial_seed = int(rng.integers(0, 2**31 - 1))
            for method in methods:
                method_rng = np.random.default_rng(trial_seed)
                received, diagnostics = simulate_received_signal(
                    method,
                    float(bmi),
                    respiration_hz,
                    heart_hz,
                    snr_db,
                    method_rng,
                )
                respiration_est, heart_est, frequencies, spectrum = estimate_vital_rates(
                    received, method.system.radar if method.system else RadarConfig()
                )
                rows.append(
                    {
                        "method": method.name,
                        "bmi": float(bmi),
                        "trial": float(trial),
                        "snr_db": float(snr_db),
                        "respiration_true_bpm": 60.0 * respiration_hz,
                        "respiration_est_bpm": respiration_est,
                        "respiration_abs_error_bpm": abs(
                            respiration_est - 60.0 * respiration_hz
                        ),
                        "heart_true_bpm": 60.0 * heart_hz,
                        "heart_est_bpm": heart_est,
                        "heart_abs_error_bpm": abs(heart_est - 60.0 * heart_hz),
                        "spatial_sinr_db": diagnostics["spatial_sinr_db"],
                    }
                )
                if (
                    bmi == bmis[len(bmis) // 2]
                    and trial == 0
                    and method.name in {methods[0].name, methods[-1].name}
                ):
                    example[f"{method.name}:frequency"] = frequencies
                    example[f"{method.name}:spectrum"] = spectrum
                    example["respiration_true_hz"] = np.array([respiration_hz])
                    example["heart_true_hz"] = np.array([heart_hz])
    return rows, example


def aggregate_metrics(
    rows: list[dict[str, float | str]], methods: list[Method], bmis: np.ndarray
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        method_result = {"respiration_rmse_bpm": [], "heart_rmse_bpm": [], "sinr_db": []}
        for bmi in bmis:
            selected = [
                row
                for row in rows
                if row["method"] == method.name and float(row["bmi"]) == float(bmi)
            ]
            resp_error = np.array(
                [float(row["respiration_est_bpm"]) - float(row["respiration_true_bpm"]) for row in selected]
            )
            heart_error = np.array(
                [float(row["heart_est_bpm"]) - float(row["heart_true_bpm"]) for row in selected]
            )
            method_result["respiration_rmse_bpm"].append(
                float(np.sqrt(np.mean(resp_error**2)))
            )
            method_result["heart_rmse_bpm"].append(
                float(np.sqrt(np.mean(heart_error**2)))
            )
            method_result["sinr_db"].append(
                float(np.mean([float(row["spatial_sinr_db"]) for row in selected]))
            )
        result[method.name] = method_result
    return result


def save_outputs(
    output_dir: Path,
    rows: list[dict[str, float | str]],
    aggregate: dict[str, dict[str, list[float]]],
    methods: list[Method],
    bmis: np.ndarray,
    histories: dict[str, OptimizationResult],
    example: dict[str, np.ndarray],
    metadata: dict[str, float | int | list[float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "metrics": aggregate}, handle, indent=2)
    with (output_dir / "summary_table.md").open("w", encoding="utf-8") as handle:
        handle.write("# Simulated experiment summary\n\n")
        handle.write(
            "These are deterministic Monte Carlo results under the assumptions in "
            "`MATHEMATICAL_MODEL.md`; they are not clinical measurements.\n\n"
        )
        handle.write(
            "| Method | Mean spatial SINR (dB) | Respiration RMSE (bpm) | "
            "Heart RMSE (bpm) |\n"
        )
        handle.write("|---|---:|---:|---:|\n")
        for method in methods:
            values = aggregate[method.name]
            handle.write(
                f"| {method.name} | {np.mean(values['sinr_db']):.2f} | "
                f"{np.mean(values['respiration_rmse_bpm']):.2f} | "
                f"{np.mean(values['heart_rmse_bpm']):.2f} |\n"
            )
        handle.write("\nRun settings: `" + json.dumps(metadata, sort_keys=True) + "`.\n")
    phase_payload: dict[str, np.ndarray] = {}
    for method in methods:
        if method.phase_bank:
            for bmi, phases in method.phase_bank.items():
                safe_name = method.name.lower().replace(" ", "_").replace("-", "_")
                phase_payload[f"{safe_name}_bmi_{bmi:g}"] = phases
    np.savez(output_dir / "optimized_phases.npz", **phase_payload)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method in methods:
        ax.plot(
            bmis,
            aggregate[method.name]["sinr_db"],
            marker="o",
            linewidth=2,
            color=method.color,
            label=method.name,
        )
    ax.set(xlabel="BMI conditioning value (kg/m²)", ylabel="Mean spatial SINR (dB)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_sinr_vs_bmi.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    keys = ["respiration_rmse_bpm", "heart_rmse_bpm"]
    titles = ["Respiration-rate RMSE", "Heart-rate RMSE"]
    for ax, key, title in zip(axes, keys, titles, strict=True):
        for method in methods:
            ax.plot(
                bmis,
                aggregate[method.name][key],
                marker="o",
                linewidth=2,
                color=method.color,
                label=method.name,
            )
        ax.set_title(title)
        ax.set_xlabel("BMI conditioning value (kg/m²)")
        ax.set_ylabel("RMSE (beats/min)")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "vital_rate_rmse_vs_bmi.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for label, result in histories.items():
        if label in {"3-layer universal", f"3-layer adaptive BMI {bmis[len(bmis)//2]:g}"}:
            ax.plot(result.sir_history_db, linewidth=2, label=label)
    ax.set(xlabel="Optimization iteration", ylabel="Training spatial SINR (dB)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "optimization_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    max_frequency = 2.5
    for method in (methods[0], methods[-1]):
        f = example[f"{method.name}:frequency"]
        spectrum = example[f"{method.name}:spectrum"]
        spectrum_db = 10.0 * np.log10(spectrum / max(np.max(spectrum), 1e-30) + 1e-12)
        mask = f <= max_frequency
        ax.plot(f[mask], spectrum_db[mask], linewidth=1.6, label=method.name, color=method.color)
    ax.axvline(float(example["respiration_true_hz"][0]), color="#111827", linestyle="--", linewidth=1, label="True rates")
    ax.axvline(float(example["heart_true_hz"][0]), color="#111827", linestyle="--", linewidth=1)
    ax.set(xlabel="Frequency (Hz)", ylabel="Normalised phase spectrum (dB)", ylim=(-60, 2))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "example_phase_spectrum.png", dpi=180)
    plt.close(fig)


def run_experiment(
    output_dir: Path,
    iterations: int = 180,
    trials: int = 10,
    snr_db: float = 8.0,
    seed: int = 2026,
) -> dict[str, dict[str, list[float]]]:
    bmis = np.array([18.0, 22.0, 26.0, 30.0, 34.0, 38.0])
    methods, histories = train_methods(bmis, iterations, seed)
    rows, example = evaluate(methods, bmis, trials, snr_db, seed + 1000)
    aggregate = aggregate_metrics(rows, methods, bmis)
    save_outputs(
        output_dir,
        rows,
        aggregate,
        methods,
        bmis,
        histories,
        example,
        {
            "seed": seed,
            "iterations": iterations,
            "trials_per_bmi": trials,
            "snr_db": snr_db,
            "bmi_values": bmis.tolist(),
        },
    )
    return aggregate
