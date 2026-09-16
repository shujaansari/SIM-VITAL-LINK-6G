#!/usr/bin/env python3
"""Command-line entry point for the adaptive SIM vital-sign experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim_vitals.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a BMI-adaptive SIM for contactless vital sensing."
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke experiment (60 iterations, 3 trials per BMI).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.iterations = min(args.iterations, 60)
        args.trials = min(args.trials, 3)
    metrics = run_experiment(
        output_dir=args.output,
        iterations=args.iterations,
        trials=args.trials,
        snr_db=args.snr_db,
        seed=args.seed,
    )
    print(f"Results saved to: {args.output.resolve()}")
    print("Mean metrics across BMI values:")
    for method, values in metrics.items():
        resp = float(np.mean(values["respiration_rmse_bpm"]))
        heart = float(np.mean(values["heart_rmse_bpm"]))
        sinr = float(np.mean(values["sinr_db"]))
        print(
            f"  {method:31s} SINR={sinr:7.2f} dB, "
            f"resp-RMSE={resp:6.2f} bpm, heart-RMSE={heart:6.2f} bpm"
        )


if __name__ == "__main__":
    main()
