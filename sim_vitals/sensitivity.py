"""Reproducible sensitivity studies for the adaptive SIM vital-sign model."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim_vitals_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sim_vitals_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import (
    Method,
    aggregate_metrics,
    evaluate,
    nearest_phase,
    train_methods,
    training_fields,
)
from .model import ArrayConfig, RadarConfig, SIMSystem, clutter_signatures
from .optimization import optimize_phases, quantize_phases


DEFAULT_BMIS = np.array([18.0, 22.0, 26.0, 30.0, 34.0, 38.0])


def _mean_metric(
    aggregate: dict[str, dict[str, list[float]]], method: str, metric: str
) -> float:
    return float(np.mean(aggregate[method][metric]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _adaptive_sim(methods: list[Method]) -> Method:
    return next(method for method in methods if method.name == "3-layer SIM (BMI-adaptive)")


def run_snr_sweep(
    methods: list[Method],
    bmis: np.ndarray,
    snr_values: np.ndarray,
    trials: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate rate recovery versus receiver SNR using paired trials."""

    summary: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for snr_db in snr_values:
        trial_rows, _ = evaluate(methods, bmis, trials, float(snr_db), seed)
        for row in trial_rows:
            raw.append({"study": "snr", **row})
        aggregate = aggregate_metrics(trial_rows, methods, bmis)
        for method in methods:
            summary.append(
                {
                    "snr_db": float(snr_db),
                    "method": method.name,
                    "respiration_rmse_bpm": _mean_metric(
                        aggregate, method.name, "respiration_rmse_bpm"
                    ),
                    "heart_rmse_bpm": _mean_metric(
                        aggregate, method.name, "heart_rmse_bpm"
                    ),
                }
            )
    return summary, raw


def run_quantization_sweep(
    adaptive: Method,
    bmis: np.ndarray,
    phase_bits: list[int],
    trials: int,
    snr_db: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare continuous and B-bit phase control without retraining."""

    if adaptive.system is None or adaptive.phase_bank is None:
        raise ValueError("adaptive SIM method must contain a phase bank")
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#10b981"]
    quantized_methods = [
        Method("Continuous phase", adaptive.system, adaptive.phase_bank, "#10b981")
    ]
    for index, bits in enumerate(phase_bits):
        bank = {
            bmi: quantize_phases(phases, bits)
            for bmi, phases in adaptive.phase_bank.items()
        }
        quantized_methods.append(
            Method(f"{bits}-bit phase", adaptive.system, bank, colors[index % len(colors)])
        )
    trial_rows, _ = evaluate(quantized_methods, bmis, trials, snr_db, seed)
    aggregate = aggregate_metrics(trial_rows, quantized_methods, bmis)
    summary: list[dict[str, Any]] = []
    for method in quantized_methods:
        bits: int | str = (
            "continuous" if method.name == "Continuous phase" else int(method.name.split("-")[0])
        )
        summary.append(
            {
                "phase_bits": bits,
                "method": method.name,
                "spatial_sinr_db": _mean_metric(aggregate, method.name, "sinr_db"),
                "respiration_rmse_bpm": _mean_metric(
                    aggregate, method.name, "respiration_rmse_bpm"
                ),
                "heart_rmse_bpm": _mean_metric(
                    aggregate, method.name, "heart_rmse_bpm"
                ),
            }
        )
    raw = [{"study": "quantization", **row} for row in trial_rows]
    return summary, raw


def train_depth_methods(
    bmis: np.ndarray,
    depths: list[int],
    iterations: int,
    seed: int,
    layer_loss_db: float,
) -> list[Method]:
    """Train a BMI-adaptive phase bank for each requested number of layers."""

    colors = ["#f59e0b", "#84cc16", "#10b981", "#06b6d4", "#3b82f6", "#8b5cf6"]
    methods: list[Method] = []
    for depth_index, depth in enumerate(depths):
        system = SIMSystem(
            ArrayConfig(layers=depth, insertion_loss_db_per_layer=layer_loss_db),
            RadarConfig(),
        )
        clutter, clutter_power = clutter_signatures(system)
        bank: dict[float, np.ndarray] = {}
        for bmi_index, bmi in enumerate(bmis):
            # Identical seed construction supplies the same morphology variations
            # for every depth, making the layer comparison paired.
            rng = np.random.default_rng(seed + 100 * bmi_index)
            fields, weights = training_fields(
                system, np.array([bmi]), rng, variants_per_bmi=6
            )
            result = optimize_phases(
                system,
                fields,
                weights,
                clutter,
                clutter_power,
                iterations=iterations,
                seed=seed + 1000 * depth + bmi_index,
            )
            bank[float(bmi)] = result.phases
        methods.append(
            Method(
                f"{depth}-layer adaptive SIM",
                system,
                bank,
                colors[depth_index % len(colors)],
            )
        )
    return methods


def run_depth_sweep(
    bmis: np.ndarray,
    depths: list[int],
    iterations: int,
    trials: int,
    snr_db: float,
    seed: int,
    layer_loss_db: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = train_depth_methods(bmis, depths, iterations, seed, layer_loss_db)
    trial_rows, _ = evaluate(methods, bmis, trials, snr_db, seed + 1)
    aggregate = aggregate_metrics(trial_rows, methods, bmis)
    summary = []
    for depth, method in zip(depths, methods, strict=True):
        summary.append(
            {
                "layers": depth,
                "trainable_phases": depth * method.system.array.elements,
                "insertion_loss_db": depth * layer_loss_db,
                "spatial_sinr_db": _mean_metric(aggregate, method.name, "sinr_db"),
                "respiration_rmse_bpm": _mean_metric(
                    aggregate, method.name, "respiration_rmse_bpm"
                ),
                "heart_rmse_bpm": _mean_metric(
                    aggregate, method.name, "heart_rmse_bpm"
                ),
            }
        )
    raw = [{"study": "depth", **row} for row in trial_rows]
    return summary, raw


def remap_phase_bank(
    phase_bank: dict[float, np.ndarray],
    actual_bmis: np.ndarray,
    bmi_offset: float,
) -> dict[float, np.ndarray]:
    """Map each actual BMI to the code selected using a biased BMI estimate."""

    return {
        float(actual): nearest_phase(phase_bank, float(actual + bmi_offset)).copy()
        for actual in actual_bmis
    }


def run_mismatch_sweep(
    adaptive: Method,
    actual_bmis: np.ndarray,
    bmi_offsets: np.ndarray,
    trials: int,
    snr_db: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate codebook-selection bias while holding true morphology fixed."""

    if adaptive.system is None or adaptive.phase_bank is None:
        raise ValueError("adaptive SIM method must contain a phase bank")
    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(bmi_offsets)))
    methods: list[Method] = []
    for offset, color in zip(bmi_offsets, colors, strict=True):
        methods.append(
            Method(
                f"BMI bias {offset:+g}",
                adaptive.system,
                remap_phase_bank(adaptive.phase_bank, actual_bmis, float(offset)),
                matplotlib.colors.to_hex(color),
            )
        )
    trial_rows, _ = evaluate(methods, actual_bmis, trials, snr_db, seed)
    aggregate = aggregate_metrics(trial_rows, methods, actual_bmis)
    summary: list[dict[str, Any]] = []
    reference_sinr = None
    for offset, method in zip(bmi_offsets, methods, strict=True):
        sinr = _mean_metric(aggregate, method.name, "sinr_db")
        if float(offset) == 0.0:
            reference_sinr = sinr
        summary.append(
            {
                "bmi_bias": float(offset),
                "spatial_sinr_db": sinr,
                "respiration_rmse_bpm": _mean_metric(
                    aggregate, method.name, "respiration_rmse_bpm"
                ),
                "heart_rmse_bpm": _mean_metric(
                    aggregate, method.name, "heart_rmse_bpm"
                ),
            }
        )
    if reference_sinr is None:
        reference_sinr = max(row["spatial_sinr_db"] for row in summary)
    for row in summary:
        row["sinr_loss_vs_unbiased_db"] = float(reference_sinr - row["spatial_sinr_db"])
    raw = [{"study": "bmi_mismatch", **row} for row in trial_rows]
    return summary, raw


def _plot_snr(path: Path, rows: list[dict[str, Any]], methods: list[Method]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True)
    for method in methods:
        selected = [row for row in rows if row["method"] == method.name]
        x = [row["snr_db"] for row in selected]
        axes[0].plot(
            x,
            [max(row["respiration_rmse_bpm"], 0.01) for row in selected],
            marker="o",
            linewidth=2,
            color=method.color,
            label=method.name,
        )
        axes[1].plot(
            x,
            [max(row["heart_rmse_bpm"], 0.01) for row in selected],
            marker="o",
            linewidth=2,
            color=method.color,
            label=method.name,
        )
    for ax, title in zip(axes, ["Respiration robustness", "Heartbeat robustness"], strict=True):
        ax.set_yscale("log")
        ax.set_xlabel("Single-antenna reference SNR (dB)")
        ax.set_ylabel("Rate RMSE (bpm)")
        ax.set_title(title)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_two_metric_sweep(
    path: Path,
    x: list[float],
    labels: list[str],
    sinr: list[float],
    heart: list[float],
    xlabel: str,
    heart_log: bool = True,
) -> None:
    fig, sinr_axis = plt.subplots(figsize=(7.4, 4.5))
    heart_axis = sinr_axis.twinx()

    sinr_line = sinr_axis.plot(
        x,
        sinr,
        marker="o",
        linewidth=2.2,
        color="#006d77",
        label="Mean spatial SINR",
    )[0]
    heart_line = heart_axis.plot(
        x,
        [max(value, 0.01) for value in heart],
        marker="s",
        linestyle="--",
        linewidth=2.2,
        color="#c1121f",
        label="Heart-rate RMSE",
    )[0]

    sinr_axis.set_xlabel(xlabel)
    sinr_axis.set_ylabel("Mean spatial SINR (dB)", color=sinr_line.get_color())
    heart_scale = "beats/min, log scale" if heart_log else "beats/min"
    heart_axis.set_ylabel(f"Heart-rate RMSE ({heart_scale})", color=heart_line.get_color())
    if heart_log:
        heart_axis.set_yscale("log")
    sinr_axis.tick_params(axis="y", colors=sinr_line.get_color())
    heart_axis.tick_params(axis="y", colors=heart_line.get_color())
    sinr_axis.spines["left"].set_color(sinr_line.get_color())
    heart_axis.spines["right"].set_color(heart_line.get_color())
    sinr_axis.grid(True, axis="y", alpha=0.3)
    sinr_axis.grid(False, axis="x")
    heart_axis.grid(False)
    sinr_axis.legend(
        [sinr_line, heart_line],
        [sinr_line.get_label(), heart_line.get_label()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=True,
    )
    if labels:
        sinr_axis.set_xticks(x, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_summary_markdown(
    path: Path,
    metadata: dict[str, Any],
    snr: list[dict[str, Any]],
    quantization: list[dict[str, Any]],
    depth: list[dict[str, Any]],
    mismatch: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# SIM sensitivity-study summary\n\n")
        handle.write(
            "All values are simulated under the assumptions in `MATHEMATICAL_MODEL.md`; "
            "they are not clinical measurements.\n\n"
        )
        handle.write("## SNR robustness\n\n")
        handle.write("| SNR (dB) | Method | Respiration RMSE | Heart RMSE |\n")
        handle.write("|---:|---|---:|---:|\n")
        for row in snr:
            handle.write(
                f"| {row['snr_db']:g} | {row['method']} | "
                f"{row['respiration_rmse_bpm']:.2f} | {row['heart_rmse_bpm']:.2f} |\n"
            )
        handle.write("\n## Phase resolution\n\n")
        handle.write("| Resolution | SINR (dB) | Respiration RMSE | Heart RMSE |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in quantization:
            handle.write(
                f"| {row['phase_bits']} | {row['spatial_sinr_db']:.2f} | "
                f"{row['respiration_rmse_bpm']:.2f} | {row['heart_rmse_bpm']:.2f} |\n"
            )
        handle.write("\n## Layer depth\n\n")
        handle.write(
            "| Layers | Trainable phases | Total insertion loss (dB) | SINR (dB) | Heart RMSE |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|\n")
        for row in depth:
            handle.write(
                f"| {row['layers']} | {row['trainable_phases']} | "
                f"{row['insertion_loss_db']:.2f} | {row['spatial_sinr_db']:.2f} | "
                f"{row['heart_rmse_bpm']:.2f} |\n"
            )
        handle.write("\n## BMI-conditioning mismatch\n\n")
        handle.write("| BMI bias | SINR (dB) | SINR loss (dB) | Heart RMSE |\n")
        handle.write("|---:|---:|---:|---:|\n")
        for row in mismatch:
            handle.write(
                f"| {row['bmi_bias']:+g} | {row['spatial_sinr_db']:.2f} | "
                f"{row['sinr_loss_vs_unbiased_db']:.2f} | {row['heart_rmse_bpm']:.2f} |\n"
            )
        handle.write("\nRun settings: `" + json.dumps(metadata, sort_keys=True) + "`.\n")


def run_sensitivity_suite(
    output_dir: Path,
    iterations: int = 180,
    trials: int = 12,
    snr_values: np.ndarray = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]),
    phase_bits: list[int] = [1, 2, 3, 4, 6],
    depths: list[int] = [1, 2, 3, 4, 5],
    bmi_offsets: np.ndarray = np.array([-8.0, -4.0, 0.0, 4.0, 8.0]),
    evaluation_snr_db: float = 8.0,
    layer_loss_db: float = 0.5,
    seed: int = 2026,
) -> dict[str, list[dict[str, Any]]]:
    """Train the reference methods and execute all four sensitivity studies."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    methods, _ = train_methods(DEFAULT_BMIS, iterations, seed)
    adaptive = _adaptive_sim(methods)

    snr_summary, snr_raw = run_snr_sweep(
        methods, DEFAULT_BMIS, snr_values, trials, seed + 2000
    )
    quant_summary, quant_raw = run_quantization_sweep(
        adaptive, DEFAULT_BMIS, phase_bits, trials, evaluation_snr_db, seed + 3000
    )
    depth_summary, depth_raw = run_depth_sweep(
        DEFAULT_BMIS,
        depths,
        iterations,
        trials,
        evaluation_snr_db,
        seed + 4000,
        layer_loss_db,
    )
    actual_bmis = np.array([22.0, 26.0, 30.0, 34.0])
    mismatch_summary, mismatch_raw = run_mismatch_sweep(
        adaptive,
        actual_bmis,
        bmi_offsets,
        trials,
        evaluation_snr_db,
        seed + 5000,
    )

    summaries = {
        "snr": snr_summary,
        "quantization": quant_summary,
        "depth": depth_summary,
        "bmi_mismatch": mismatch_summary,
    }
    metadata: dict[str, Any] = {
        "seed": seed,
        "iterations": iterations,
        "trials_per_condition": trials,
        "training_bmis": DEFAULT_BMIS.tolist(),
        "snr_values_db": snr_values.tolist(),
        "phase_bits": phase_bits,
        "depths": depths,
        "bmi_offsets": bmi_offsets.tolist(),
        "evaluation_snr_db": evaluation_snr_db,
        "insertion_loss_db_per_layer": layer_loss_db,
    }
    with (output_dir / "sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "studies": summaries}, handle, indent=2)
    _write_csv(output_dir / "snr_sweep.csv", snr_summary)
    _write_csv(output_dir / "quantization_sweep.csv", quant_summary)
    _write_csv(output_dir / "layer_sweep.csv", depth_summary)
    _write_csv(output_dir / "bmi_mismatch_sweep.csv", mismatch_summary)
    _write_csv(output_dir / "snr_trials.csv", snr_raw)
    _write_csv(output_dir / "quantization_trials.csv", quant_raw)
    _write_csv(output_dir / "layer_trials.csv", depth_raw)
    _write_csv(output_dir / "bmi_mismatch_trials.csv", mismatch_raw)

    _plot_snr(output_dir / "snr_robustness.png", snr_summary, methods)
    quant_order = sorted(
        quant_summary,
        key=lambda row: 99 if row["phase_bits"] == "continuous" else int(row["phase_bits"]),
    )
    quant_x = list(range(len(quant_order)))
    _plot_two_metric_sweep(
        output_dir / "phase_quantization.png",
        quant_x,
        [str(row["phase_bits"]) for row in quant_order],
        [row["spatial_sinr_db"] for row in quant_order],
        [row["heart_rmse_bpm"] for row in quant_order],
        "Phase resolution (bits; continuous at right)",
    )
    _plot_two_metric_sweep(
        output_dir / "layer_depth.png",
        [float(row["layers"]) for row in depth_summary],
        [],
        [row["spatial_sinr_db"] for row in depth_summary],
        [row["heart_rmse_bpm"] for row in depth_summary],
        "Number of SIM layers",
        heart_log=False,
    )
    _plot_two_metric_sweep(
        output_dir / "bmi_mismatch.png",
        [row["bmi_bias"] for row in mismatch_summary],
        [],
        [row["spatial_sinr_db"] for row in mismatch_summary],
        [row["heart_rmse_bpm"] for row in mismatch_summary],
        "BMI conditioning bias (kg/m²)",
    )
    _save_summary_markdown(
        output_dir / "sensitivity_summary.md",
        metadata,
        snr_summary,
        quant_summary,
        depth_summary,
        mismatch_summary,
    )
    return summaries
