# Full-wave validation and paper-development roadmap

## Recommended direction

Keep the present article centred on the system model, optimization method and
simulation evidence. Its main contribution is the morphology-conditioned receive
architecture. A complete programmable unit-cell design would add a second research
problem, a different validation methodology and a substantial set of results. It is
better developed as a follow-on paper, while the current manuscript states clearly
how full-wave data will replace the ideal operators.

The next paper should answer a distinct question:

> Can a physically realizable multilayer transmissive metasurface reproduce the
> optimized wave-domain transformation over a useful bandwidth after insertion loss,
> phase--amplitude coupling, incidence angle, mutual coupling and manufacturing
> tolerance are included?

That question is strong enough for a separate paper and creates a natural bridge from
the present digital twin to fabrication and measurement.

## What HFSS or CST can provide

Both solvers can be used. The most efficient starting point is an infinite periodic
unit cell rather than the full 6-by-6 stack.

### Stage 1: programmable unit-cell model

Model one transmissive cell at the 30-GHz centre frequency using:

- periodic or linked boundaries on the four lateral faces;
- Floquet ports above and below the cell;
- the actual conductor, substrate and spacer losses;
- one model parameter for each controllable state, such as varactor capacitance,
  switch state, MEMS displacement or an equivalent surface impedance;
- co- and cross-polarized fundamental Floquet modes;
- de-embedded port reference planes at the cell surfaces.

A useful initial frequency sweep is 24--36 GHz. It is deliberately wider than the
eventual operating band so that the resonances and the usable-band edges are visible.
The following variables should be swept:

| Variable | Suggested first sweep |
|---|---|
| Frequency | 24--36 GHz |
| Control state | continuous tuning variable, then the intended 2-, 4- or 6-bit states |
| Incidence elevation | 0, 10, 20, 30 and 40 degrees |
| Azimuth | 0 and 45 degrees initially |
| Polarization | two orthogonal linear polarizations |
| Substrate loss tangent | nominal and tolerance limits |
| Metal conductivity | nominal and reduced-conductivity cases |
| Unit-cell geometry | manufacturing-tolerance sweep |

HFSS supports planar-periodic unit cells with linked boundaries and Floquet ports,
and returns a modal S-matrix. CST offers the equivalent unit-cell/Floquet workflow;
its time-domain solver is attractive for broadband screening, while its
frequency-domain solver is useful near resonant states. Use one solver for the main
design and confirm a small number of operating points in the other solver if both
licences are available.

### Stage 2: quantities to extract

For tuning state $u$, frequency $f$ and incidence direction $\Omega$, export the
complex co-polarized transmission and reflection coefficients

\[
T(f,u,\Omega)=S_{21}(f,u,\Omega),\qquad
R(f,u,\Omega)=S_{11}(f,u,\Omega).
\]

The main reported quantities should be:

1. **Transmission magnitude and insertion loss**

   \[
   \mathrm{IL}(f,u,\Omega)=-20\log_{10}|S_{21}(f,u,\Omega)|\ \mathrm{dB}.
   \]

   This expression is valid when the fundamental transmitted mode represents the
   transmitted power. If higher Floquet modes or cross-polarized modes propagate,
   use the sum of all transmitted modal powers instead:

   \[
   \mathrm{IL}_{\mathrm{power}}=-10\log_{10}
   \left(\frac{P_{\mathrm{transmitted}}}{P_{\mathrm{incident}}}\right).
   \]

2. **Return loss**

   \[
   \mathrm{RL}(f,u,\Omega)=-20\log_{10}|S_{11}(f,u,\Omega)|.
   \]

3. **Transmission phase and usable phase coverage**

   \[
   \phi_t(f,u,\Omega)=\operatorname{unwrap}\{\angle S_{21}(f,u,\Omega)\}.
   \]

   Plot phase against the control state at each frequency. Report the available
   phase span, the non-uniformity between discrete states and the accompanying
   amplitude ripple. A nominal 360-degree span is not sufficient if much of it is
   obtained only in a high-loss resonance.

4. **Group delay**

   \[
   \tau_g(f,u,\Omega)=-\frac{1}{2\pi}\frac{\partial\phi_t}{\partial f}.
   \]

   Large state-dependent delay variation can distort broadband waveforms and make a
   phase code optimized at 30 GHz unreliable away from the centre frequency.

5. **Absorption and polarization conversion**

   Below the onset of higher modes,

   \[
   A=1-|S_{11}^{\mathrm{co}}|^2-|S_{21}^{\mathrm{co}}|^2
     -|S_{11}^{\mathrm{cross}}|^2-|S_{21}^{\mathrm{cross}}|^2.
   \]

   This separates dissipative loss from reflection and unwanted polarization
   conversion.

6. **Bandwidth**

   Report three bandwidths rather than one:

   - amplitude bandwidth: insertion loss remains below a chosen engineering limit;
   - phase-state bandwidth: the required phase coverage and maximum phase error are
     maintained across all programmed states;
   - system bandwidth: the trained stack remains within 3 dB of its centre-frequency
     spatial SINR or gain.

The thresholds should be declared before the sweep. They are design criteria, not
universal metasurface standards.

### Stage 3: connect the full-wave model to the Python simulator

The present propagation block

\[
\eta_\ell\mathbf P_\ell\mathbf D_\ell
\]

should be replaced by a calibrated, frequency-dependent operator

\[
\mathbf T_\ell(f,u_\ell,\Omega),
\]

derived from the exported complex S-parameters or sampled near fields. The first
implementation can assign the unit-cell transmission coefficient to each diagonal
element while retaining the analytical free-space propagation matrix. A stronger
implementation uses full-wave near-field sampling to estimate the complete
inter-layer coupling matrix, including non-local coupling.

The comparison should quantify:

- complex gain error between the ideal and full-wave stacks;
- normalized field correlation at each layer and at the detector;
- spatial SINR versus frequency;
- rate RMSE versus frequency and control resolution;
- loss accumulation versus the number of layers;
- degradation under angular, material and fabrication tolerances;
- codebook transfer: performance when masks optimized in the ideal model are applied
  directly to the full-wave model, followed by performance after full-wave-aware
  retraining.

The last comparison is especially important. It distinguishes a model-calibration
problem from a limitation of the SIM architecture itself.

### Stage 4: finite stack

After selecting a unit cell, model finite 6-by-6 layers with the 5-mm pitch and
12-mm spacing used in the present paper. Begin with one and two layers, then increase
the depth only if the electromagnetic result justifies it. Excite the model using
plane waves from the desired and clutter directions and sample the field at the
detector port or receive antenna.

The finite model should examine:

- edge diffraction and finite-aperture efficiency;
- coupling between adjacent cells and between layers;
- sensitivity to layer misalignment and spacing error;
- detector position and mode matching;
- undesired Floquet or surface-wave modes;
- thermal and bias-network losses if active tuning components are represented.

This stage can become computationally heavy. It should not begin until the unit-cell
sweep has reduced the control states and geometry to a manageable candidate set.

## Suggested paper sequence

### Paper 1: current manuscript

**Core claim:** morphology-conditioned wave-domain processing can improve simulated
vital-sign extraction and exposes clear SNR, quantization, layer-depth and mismatch
trade-offs.

Keep in this paper:

- the physiological and spatial digital twin;
- adjoint phase optimization;
- comparison with single-antenna, one-layer, universal and digital-array baselines;
- robustness and hardware-sensitivity sweeps;
- a concise full-wave validation pathway in the Discussion.

Do not add an unoptimized unit cell merely to claim that HFSS or CST was used. A weak
electromagnetic appendix would distract from the present contribution and invite
questions that the current study is not designed to answer.

### Paper 2: full-wave-calibrated SIM

Possible title:

**Full-Wave Design and Broadband Validation of a Programmable Stacked Intelligent
Metasurface for Contactless Vital-Sign Radar**

Main contributions:

- a physically realizable transmissive unit cell near 30 GHz;
- phase-state, loss, bandwidth, angle and polarization characterization;
- one- to five-layer full-wave comparison;
- transfer of the ideal phase codebook into the full-wave model;
- full-wave-aware retraining and tolerance analysis;
- an exported electromagnetic surrogate coupled to the physiological simulator.

### Paper 3: phantom and benchtop validation

Possible title:

**Morphology-Aware Stacked Metasurface Radar for Contactless Cardiopulmonary
Monitoring: Phantom and Measurement Validation**

Main contributions:

- fabricated layer and stack S-parameter measurements;
- free-space VNA insertion-loss and phase measurements;
- repeatability across programmed states, angle and polarization;
- a moving multilayer thorax phantom with controlled respiratory and cardiac
  displacement;
- comparison among simulated, full-wave and measured transfer functions;
- fixed, preregistered phase masks evaluated without retuning to the test cases.

### Paper 4: adaptive multi-person 6G sensing

This paper can extend from a BMI codebook to uncertainty-aware conditioning using
multiple descriptors, online calibration or a learned low-dimensional morphology
embedding. It can also address multiple people, communication--sensing coexistence,
latency, privacy and edge control. This should follow physical validation, otherwise
the additional algorithmic complexity will rest on the same unverified hardware
assumptions.

## Patent-oriented checkpoints

Potentially protectable value is more likely to lie in the complete sensing and
control method than in the broad idea of a programmable metasurface. Candidate claim
families for discussion with the university technology-transfer office include:

- selecting a multilayer electromagnetic transformation from a subject-conditioned
  physiological codebook;
- jointly preserving cardiopulmonary micromotion and suppressing spatially distinct
  motion before a reduced-channel detector;
- calibrating an analytical SIM model from full-wave or measured transfer matrices
  and updating the phase codebook from that calibration;
- uncertainty-aware selection or interpolation between morphology-conditioned phase
  stacks;
- a closed-loop procedure that uses a short radar calibration interval to select or
  refine the passive analogue front end;
- a specific low-loss multilayer unit-cell and detector arrangement, if the follow-on
  electromagnetic design produces a genuinely new structure.

Before publishing detailed unit-cell geometry, control circuitry, calibration logic
or a new adaptive control method, submit an invention disclosure to the relevant
university office and obtain professional patent advice. Public disclosure can become
prior art against a later filing in many jurisdictions; grace-period rules are not
uniform. This roadmap is research planning, not a patentability opinion.

## Practical decision

If only one commercial solver is available, use that solver. If both are available,
CST is a convenient first choice for broad unit-cell screening and HFSS is an equally
strong choice for a carefully controlled Floquet-port FEM study with Python
automation. The quality of boundary conditions, de-embedding, mesh convergence,
material data and modal power accounting will matter more than the product name.

The first concrete deliverable should be a single programmable unit-cell project and
a CSV export containing frequency, control state, incidence angle, polarization,
complex $S_{11}$ and complex $S_{21}$. Once that table exists, it can be connected to
the current Python model without waiting for a full finite-stack solve.

## Primary technical sources

- Ansys HFSS Floquet-port guide:
  https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v252/en/PDFs/HFSS%20Floquet%20Ports.pdf
- Ansys HFSS periodic unit-cell overview:
  https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v251/en/Subsystems/HFSS/Subsystems/HFSS%20Floquet%20Ports/Content/Introduction.htm
- Ansys PyAEDT automation overview:
  https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v261/en/Subsystems/HFSS/Subsystems/HFSS%20Scripting/Content/PyAEDT.htm
- CST Studio Suite electromagnetic solver overview:
  https://www.3ds.com/products/simulia/cst-studio-suite/electromagnetic-simulation-solvers
- CST parameterization and optimization overview:
  https://www.3ds.com/products/simulia/cst-studio-suite/automatic-optimization
- WIPO patent FAQ on disclosure and prior art:
  https://www.wipo.int/en/web/patents/faq_patents
