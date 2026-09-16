# SIM sensitivity-study summary

All values are simulated under the assumptions in `MATHEMATICAL_MODEL.md`; they are not clinical measurements.

## SNR robustness

| SNR (dB) | Method | Respiration RMSE | Heart RMSE |
|---:|---|---:|---:|
| -10 | No intelligent surface | 12.54 | 25.41 |
| -10 | Digital MVDR array (BMI-adaptive) | 11.89 | 21.40 |
| -10 | 1-layer RIS (BMI-adaptive) | 11.89 | 24.21 |
| -10 | 3-layer SIM (universal) | 12.25 | 23.14 |
| -10 | 3-layer SIM (BMI-adaptive) | 11.21 | 28.53 |
| -5 | No intelligent surface | 12.20 | 25.68 |
| -5 | Digital MVDR array (BMI-adaptive) | 7.37 | 18.50 |
| -5 | 1-layer RIS (BMI-adaptive) | 9.81 | 23.19 |
| -5 | 3-layer SIM (universal) | 8.46 | 21.50 |
| -5 | 3-layer SIM (BMI-adaptive) | 7.70 | 20.39 |
| 0 | No intelligent surface | 13.56 | 26.03 |
| 0 | Digital MVDR array (BMI-adaptive) | 3.24 | 10.38 |
| 0 | 1-layer RIS (BMI-adaptive) | 3.71 | 17.28 |
| 0 | 3-layer SIM (universal) | 4.62 | 14.76 |
| 0 | 3-layer SIM (BMI-adaptive) | 1.62 | 10.26 |
| 5 | No intelligent surface | 10.90 | 24.04 |
| 5 | Digital MVDR array (BMI-adaptive) | 0.05 | 1.23 |
| 5 | 1-layer RIS (BMI-adaptive) | 0.06 | 3.21 |
| 5 | 3-layer SIM (universal) | 2.88 | 4.67 |
| 5 | 3-layer SIM (BMI-adaptive) | 0.05 | 0.12 |
| 10 | No intelligent surface | 14.35 | 24.78 |
| 10 | Digital MVDR array (BMI-adaptive) | 0.05 | 0.08 |
| 10 | 1-layer RIS (BMI-adaptive) | 0.05 | 0.13 |
| 10 | 3-layer SIM (universal) | 0.05 | 1.69 |
| 10 | 3-layer SIM (BMI-adaptive) | 0.05 | 0.07 |
| 15 | No intelligent surface | 15.45 | 25.47 |
| 15 | Digital MVDR array (BMI-adaptive) | 0.05 | 0.06 |
| 15 | 1-layer RIS (BMI-adaptive) | 0.05 | 0.11 |
| 15 | 3-layer SIM (universal) | 0.05 | 0.06 |
| 15 | 3-layer SIM (BMI-adaptive) | 0.05 | 0.05 |

## Phase resolution

| Resolution | SINR (dB) | Respiration RMSE | Heart RMSE |
|---|---:|---:|---:|
| continuous | 9.25 | 0.05 | 0.08 |
| 1 | -5.63 | 6.88 | 25.63 |
| 2 | -6.14 | 11.39 | 23.92 |
| 3 | 1.50 | 0.76 | 5.61 |
| 4 | 5.70 | 0.05 | 1.91 |
| 6 | 9.00 | 0.05 | 0.08 |

## Layer depth

| Layers | Trainable phases | Total insertion loss (dB) | SINR (dB) | Heart RMSE |
|---:|---:|---:|---:|---:|
| 1 | 36 | 0.50 | 5.34 | 0.15 |
| 2 | 72 | 1.00 | 8.29 | 0.08 |
| 3 | 108 | 1.50 | 8.08 | 0.09 |
| 4 | 144 | 2.00 | 7.72 | 0.11 |
| 5 | 180 | 2.50 | 7.35 | 0.11 |

## BMI-conditioning mismatch

| BMI bias | SINR (dB) | SINR loss (dB) | Heart RMSE |
|---:|---:|---:|---:|
| -8 | 6.26 | 2.41 | 0.15 |
| -4 | 7.75 | 0.91 | 0.11 |
| +0 | 8.67 | 0.00 | 0.09 |
| +4 | 8.11 | 0.56 | 0.11 |
| +8 | 7.29 | 1.37 | 5.89 |

Run settings: `{"bmi_offsets": [-8.0, -4.0, 0.0, 4.0, 8.0], "depths": [1, 2, 3, 4, 5], "evaluation_snr_db": 8.0, "insertion_loss_db_per_layer": 0.5, "iterations": 180, "phase_bits": [1, 2, 3, 4, 6], "seed": 2026, "snr_values_db": [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0], "training_bmis": [18.0, 22.0, 26.0, 30.0, 34.0, 38.0], "trials_per_condition": 12}`.
