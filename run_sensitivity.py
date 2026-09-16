#!/usr/bin/env python3
"""Run the full robustness and hardware-sensitivity study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim_vitals.sensitivity import run_sensitivity_suite


def comma_floats(value: str) -> np.ndarray:
    return np.asarray([float(item.strip()) for item in value.split(",")])


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SNR, phase-resolution, layer-depth and BMI-mismatch studies."
    )
    parser.add_argument("--output", type=Path, default=Path("sensitivity_results"))
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--evaluation-snr-db", type=float, default=8.0)
    parser.add_argument(
        "--layer-loss-db",
        type=float,
        default=0.5,
        help="Assumed insertion loss of each layer for the depth sweep.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-values", type=comma_floats, default=comma_floats("-10,-5,0,5,10,15"))
    parser.add_argument("--phase-bits", type=comma_ints, default=comma_ints("1,2,3,4,6"))
    parser.add_argument("--depths", type=comma_ints, default=comma_ints("1,2,3,4,5"))
    parser.add_argument("--bmi-offsets", type=comma_floats, default=comma_floats("-8,-4,0,4,8"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short three-point version of each study for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.iterations = min(args.iterations, 60)
        args.trials = min(args.trials, 3)
        args.snr_values = np.array([-5.0, 5.0, 15.0])
        args.phase_bits = [1, 2, 4]
        args.depths = [1, 2, 3]
        args.bmi_offsets = np.array([-4.0, 0.0, 4.0])
    summaries = run_sensitivity_suite(
        output_dir=args.output,
        iterations=args.iterations,
        trials=args.trials,
        snr_values=args.snr_values,
        phase_bits=args.phase_bits,
        depths=args.depths,
        bmi_offsets=args.bmi_offsets,
        evaluation_snr_db=args.evaluation_snr_db,
        layer_loss_db=args.layer_loss_db,
        seed=args.seed,
    )
    print(f"Sensitivity results saved to: {args.output.resolve()}")
    print("Study sizes:")
    for name, rows in summaries.items():
        print(f"  {name:14s} {len(rows):3d} aggregate conditions")


if __name__ == "__main__":
    main()
