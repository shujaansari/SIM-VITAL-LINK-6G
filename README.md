# Adaptive stacked intelligent metasurface for vital-sign radar

This repository is a directly executable research prototype for a **BMI-conditioned,
stacked intelligent metasurface (SIM)** that behaves as a phase-only wave-domain
neural network in a continuous-wave radar receiver. It simulates respiration and
heartbeat micro-motion, morphology-dependent spatial scattering, multiple nuisance
reflectors, analytical electromagnetic backpropagation, and vital-rate estimation.

## Run it

Python 3.10+ is recommended. NumPy and Matplotlib are the only dependencies.

```bash
python3 run_experiment.py --quick
```

For the fuller reproducible experiment:

```bash
python3 run_experiment.py --iterations 180 --trials 10 --snr-db 8 --seed 2026
```

Run the complete robustness and hardware-sensitivity suite with:

```bash
python3 run_sensitivity.py
```

For a short verification run:

```bash
python3 run_sensitivity.py --quick
```

Run the self-contained tests with:

```bash
python3 -m unittest discover -s tests -v
```

The run creates `results/` containing:

- `trial_metrics.csv`: subject-level truth, estimates, errors and spatial SINR;
- `summary.json`: aggregated metrics and complete run settings;
- `summary_table.md`: a paper-drafting table with an explicit simulation caveat;
- `optimized_phases.npz`: reusable phase coefficients;
- four publication-oriented PNG figures.

The sensitivity run creates `sensitivity_results/` with raw trial CSV files,
aggregate CSV/JSON/Markdown tables, and figures for:

- receiver-SNR robustness;
- 1/2/3/4/6-bit phase quantization versus continuous control;
- one through five SIM layers, including configurable per-layer insertion loss;
- BMI-codebook selection bias from -8 to +8 kg/m².

## What is compared

1. A conventional single receiving antenna with no intelligent surface.
2. A fully digital, BMI-adaptive MVDR array with one RF chain per element.
3. A one-layer, BMI-adaptive RIS.
4. A three-layer universal SIM trained across every BMI condition.
5. A three-layer BMI-adaptive SIM with a nearest-condition phase-codebook.

The numerical BMI mapping is deliberately isolated in `morphology_signature()` and
`morphology_scalars()` so measured relationships can replace the current hypotheses
without changing the optimization or evaluation pipeline.

## Important research boundary

This is a theoretical digital-twin experiment, not a clinical model. BMI is treated
as an observable conditioning variable and imperfect proxy for morphology. The
default attenuation, scattering-spread and displacement mappings are hypotheses for
sensitivity analysis; publication claims should say so unless they are calibrated
against phantom, full-wave, or human-subject measurements.

The corresponding equations and their exact code locations are documented in
[`MATHEMATICAL_MODEL.md`](MATHEMATICAL_MODEL.md).
