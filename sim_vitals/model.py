"""Physical and signal models for adaptive SIM-assisted vital-sign radar.

The model is intentionally parametric: BMI is used as a conditioning variable for
a morphology surrogate, not as a claim that BMI uniquely determines RF tissue
properties.  All vectors are narrowband complex baseband fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


LIGHT_SPEED = 299_792_458.0


@dataclass(frozen=True)
class ArrayConfig:
    """Geometry of the square stacked intelligent metasurface."""

    nx: int = 6
    ny: int = 6
    pitch_m: float = 5.0e-3
    layer_spacing_m: float = 12.0e-3
    layers: int = 3
    insertion_loss_db_per_layer: float = 0.0

    @property
    def elements(self) -> int:
        return self.nx * self.ny


@dataclass(frozen=True)
class RadarConfig:
    """CW radar and observation settings."""

    carrier_hz: float = 30.0e9
    sample_rate_hz: float = 20.0
    duration_s: float = 30.0
    respiration_band_hz: tuple[float, float] = (0.10, 0.60)
    heart_band_hz: tuple[float, float] = (0.80, 2.20)

    @property
    def wavelength_m(self) -> float:
        return LIGHT_SPEED / self.carrier_hz


def element_coordinates(config: ArrayConfig) -> np.ndarray:
    """Return centred x-y element coordinates with shape (N, 2)."""

    x = (np.arange(config.nx) - (config.nx - 1) / 2.0) * config.pitch_m
    y = (np.arange(config.ny) - (config.ny - 1) / 2.0) * config.pitch_m
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def steering_vector(
    xy: np.ndarray,
    wavelength_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    taper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unit-norm planar-wave signature over an aperture."""

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    kx = np.sin(az) * np.cos(el)
    ky = np.sin(el)
    phase = 2.0 * np.pi / wavelength_m * (xy[:, 0] * kx + xy[:, 1] * ky)
    field = np.exp(1j * phase)
    if taper is not None:
        field = field * taper
    return field / max(np.linalg.norm(field), 1.0e-15)


def fresnel_propagator(
    xy: np.ndarray, wavelength_m: float, separation_m: float
) -> np.ndarray:
    """Lossless modal approximation to Green-function inter-layer coupling."""

    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    distance = np.sqrt(dx * dx + dy * dy + separation_m * separation_m)
    coupling = np.exp(1j * 2.0 * np.pi * distance / wavelength_m) / distance
    # Project the sampled, aperture-truncated Green matrix onto its closest
    # unitary polar factor.  This retains Fresnel spatial mixing while modelling
    # an ideal lossless set of propagating modes.  Configured insertion loss is
    # applied separately to every propagation matrix in SIMSystem.
    left, _, right_h = np.linalg.svd(coupling, full_matrices=False)
    return left @ right_h


class SIMSystem:
    """A phase-only diffractive neural network implemented by SIM layers."""

    def __init__(
        self,
        array: ArrayConfig = ArrayConfig(),
        radar: RadarConfig = RadarConfig(),
    ) -> None:
        self.array = array
        self.radar = radar
        self.xy = element_coordinates(array)
        base = fresnel_propagator(
            self.xy, radar.wavelength_m, array.layer_spacing_m
        )
        layer_amplitude = 10.0 ** (-array.insertion_loss_db_per_layer / 20.0)
        self.propagators = [layer_amplitude * base.copy() for _ in range(array.layers)]
        # Receiver-mode field at the final layer.  A slight off-axis position
        # breaks an otherwise unnecessary geometric symmetry.
        self.receiver = steering_vector(
            self.xy, radar.wavelength_m, azimuth_deg=5.0, elevation_deg=-3.0
        )

    def with_layers(self, layers: int) -> "SIMSystem":
        """Return an otherwise identical system with a different depth."""

        return SIMSystem(
            ArrayConfig(
                nx=self.array.nx,
                ny=self.array.ny,
                pitch_m=self.array.pitch_m,
                layer_spacing_m=self.array.layer_spacing_m,
                layers=layers,
                insertion_loss_db_per_layer=self.array.insertion_loss_db_per_layer,
            ),
            self.radar,
        )

    def forward(
        self, phases: np.ndarray, input_field: np.ndarray
    ) -> tuple[complex, list[np.ndarray], list[np.ndarray]]:
        """Propagate one field and return detector amplitude plus layer states."""

        phases = np.asarray(phases, dtype=float)
        if phases.shape != (self.array.layers, self.array.elements):
            raise ValueError(
                f"phases must have shape {(self.array.layers, self.array.elements)}"
            )
        u = np.asarray(input_field, dtype=complex)
        layer_inputs: list[np.ndarray] = []
        modulated_fields: list[np.ndarray] = []
        for layer, propagator in enumerate(self.propagators):
            layer_inputs.append(u)
            v = np.exp(1j * phases[layer]) * u
            modulated_fields.append(v)
            u = propagator @ v
        detected = np.vdot(self.receiver, u)
        return detected, layer_inputs, modulated_fields

    def gain(self, phases: np.ndarray, input_field: np.ndarray) -> complex:
        """Return the complex source-to-detector gain."""

        return self.forward(phases, input_field)[0]

    def power_and_gradient(
        self, phases: np.ndarray, input_field: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return |gain|^2 and its exact real phase gradient.

        This is the adjoint/backpropagation rule for the wave-domain neural
        network.  It avoids finite differences and external autodiff packages.
        """

        detected, _, modulated_fields = self.forward(phases, input_field)
        gradient = np.empty_like(phases, dtype=float)
        adjoint = self.receiver.copy()
        for layer in range(self.array.layers - 1, -1, -1):
            before_mask = self.propagators[layer].conj().T @ adjoint
            derivative = 1j * modulated_fields[layer] * np.conj(before_mask)
            gradient[layer] = 2.0 * np.real(np.conj(detected) * derivative)
            adjoint = np.exp(-1j * phases[layer]) * before_mask
        return float(np.abs(detected) ** 2), gradient

    def direct_gain(self, input_field: np.ndarray) -> complex:
        """Single-antenna gain used as the no-surface baseline."""

        # A central element is the reference antenna.  Comparing the SIM with a
        # fully coherent N-element digital array would conflate the paper's
        # analogue-computing question with an already optimal digital beamformer.
        centre = int(np.argmin(np.sum(self.xy * self.xy, axis=1)))
        return complex(input_field[centre])


def morphology_signature(
    system: SIMSystem,
    bmi: float,
    lateral_jitter_deg: float = 0.0,
    elevation_jitter_deg: float = 0.0,
) -> np.ndarray:
    """Return a unit-norm BMI-conditioned chest-scattering surrogate.

    The chest return is a coherent combination of a dominant central component
    and two diffuse components.  Their angular spread, balance and excess phase
    vary smoothly with BMI.  These mappings are tunable hypotheses for a digital
    twin; they are not clinical population laws.
    """

    xy = system.xy
    wavelength = system.radar.wavelength_m
    delta = bmi - 22.0
    centroid_az = 2.0 + 0.08 * delta + lateral_jitter_deg
    centroid_el = -1.0 + 0.04 * delta + elevation_jitter_deg
    angular_spread = 4.0 + 0.22 * max(bmi - 18.0, 0.0)
    diffuse_weight = np.clip(0.18 + 0.012 * delta, 0.12, 0.42)
    tissue_phase = 0.10 * delta

    central = steering_vector(xy, wavelength, centroid_az, centroid_el)
    left = steering_vector(
        xy, wavelength, centroid_az - angular_spread, centroid_el + 1.5
    )
    right = steering_vector(
        xy, wavelength, centroid_az + angular_spread, centroid_el - 1.0
    )
    field = (
        (1.0 - diffuse_weight) * central
        + 0.55 * diffuse_weight * np.exp(1j * tissue_phase) * left
        + 0.45 * diffuse_weight * np.exp(-0.7j * tissue_phase) * right
    )
    radius_squared = np.sum(xy * xy, axis=1)
    radius_squared /= max(np.max(radius_squared), 1.0e-15)
    # Effective curvature is a compact surrogate for BMI-correlated changes in
    # scattering depth and chest-wall composition.  It is intentionally exposed
    # here for later calibration from full-wave or measured data.
    field *= np.exp(1j * 0.18 * delta * radius_squared)
    return field / max(np.linalg.norm(field), 1.0e-15)


def morphology_scalars(bmi: float) -> dict[str, float]:
    """Return explicit, replaceable morphology-to-parameter mappings."""

    excess = max(bmi - 22.0, 0.0)
    return {
        "reflection_amplitude": float(np.exp(-0.035 * excess)),
        "respiration_displacement_m": float(4.0e-3 * np.exp(-0.012 * excess)),
        "heart_displacement_m": float(0.24e-3 * np.exp(-0.020 * excess)),
    }


def clutter_signatures(system: SIMSystem) -> tuple[np.ndarray, np.ndarray]:
    """Return nuisance spatial signatures and their relative powers."""

    directions = [
        (-25.0, 8.0),
        (18.0, -10.0),
        (31.0, 13.0),
        (-9.0, -21.0),
        (-2.0, 5.0),
        (8.0, -4.0),
        (-7.0, -6.0),
        (13.0, 6.0),
    ]
    fields = np.stack(
        [
            steering_vector(system.xy, system.radar.wavelength_m, az, el)
            for az, el in directions
        ]
    )
    # In-band moving reflectors receive larger effective weights because their
    # residual gain is more damaging to vital-rate estimation than static clutter.
    powers = np.array([0.80, 0.65, 1.10, 0.95, 0.35, 0.35, 0.80, 0.75], dtype=float)
    return fields, powers


def physiological_displacement(
    time_s: np.ndarray,
    bmi: float,
    respiration_hz: float,
    heart_hz: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate respiration, heartbeat and slow nonstationarity in metres."""

    morphology = morphology_scalars(bmi)
    ar = morphology["respiration_displacement_m"] * rng.normal(1.0, 0.08)
    ah = morphology["heart_displacement_m"] * rng.normal(1.0, 0.10)
    phase_r, phase_h = rng.uniform(0.0, 2.0 * np.pi, size=2)
    # Mild frequency modulation represents physiologic variability without
    # requiring a hardware-derived waveform dataset.
    respiratory_phase = (
        2.0 * np.pi * respiration_hz * time_s
        + 0.06 * np.sin(2.0 * np.pi * 0.025 * time_s)
        + phase_r
    )
    cardiac_phase = (
        2.0 * np.pi * heart_hz * time_s
        + 0.035 * np.sin(2.0 * np.pi * 0.07 * time_s)
        + phase_h
    )
    displacement = ar * np.sin(respiratory_phase) + ah * np.sin(cardiac_phase)
    return displacement, {"A_r_m": float(ar), "A_h_m": float(ah)}


def estimate_vital_rates(
    received: np.ndarray, radar: RadarConfig
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate respiration and heart rates from unwrapped radar phase."""

    phase = np.unwrap(np.angle(received))
    time = np.arange(phase.size) / radar.sample_rate_hz
    linear_fit = np.polyfit(time, phase, deg=1)
    phase = phase - np.polyval(linear_fit, time)
    phase = phase - np.mean(phase)
    window = np.hanning(phase.size)
    nfft = int(2 ** np.ceil(np.log2(max(phase.size * 8, 16))))
    spectrum = np.abs(np.fft.rfft(phase * window, n=nfft)) ** 2
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / radar.sample_rate_hz)

    def peak_frequency(band: tuple[float, float]) -> float:
        mask = (frequencies >= band[0]) & (frequencies <= band[1])
        indices = np.flatnonzero(mask)
        local = int(indices[np.argmax(spectrum[mask])])
        if 0 < local < spectrum.size - 1:
            y0, y1, y2 = np.log(spectrum[local - 1 : local + 2] + 1.0e-30)
            denominator = y0 - 2.0 * y1 + y2
            offset = 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-15 else 0.0
            return float(frequencies[local] + offset * (frequencies[1] - frequencies[0]))
        return float(frequencies[local])

    respiration_bpm = 60.0 * peak_frequency(radar.respiration_band_hz)
    heart_bpm = 60.0 * peak_frequency(radar.heart_band_hz)
    return respiration_bpm, heart_bpm, frequencies, spectrum
