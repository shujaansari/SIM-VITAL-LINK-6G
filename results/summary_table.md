# Simulated experiment summary

These are deterministic Monte Carlo results under the assumptions in `MATHEMATICAL_MODEL.md`; they are not clinical measurements.

| Method | Mean spatial SINR (dB) | Respiration RMSE (bpm) | Heart RMSE (bpm) |
|---|---:|---:|---:|
| No intelligent surface | -9.90 | 13.29 | 22.31 |
| Digital MVDR array (BMI-adaptive) | 9.32 | 0.05 | 1.09 |
| 1-layer RIS (BMI-adaptive) | 5.72 | 0.05 | 0.16 |
| 3-layer SIM (universal) | 8.38 | 0.05 | 1.16 |
| 3-layer SIM (BMI-adaptive) | 9.29 | 0.05 | 0.08 |

Run settings: `{"bmi_values": [18.0, 22.0, 26.0, 30.0, 34.0, 38.0], "iterations": 180, "seed": 2026, "snr_db": 8.0, "trials_per_bmi": 10}`.
