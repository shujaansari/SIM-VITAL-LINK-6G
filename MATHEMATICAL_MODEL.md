# Mathematical model: morphology-adaptive SIM radar for vital signs

## 1. Scope and hypothesis

The working hypothesis is that a patient-conditioned stacked intelligent
metasurface can spatially enhance a cardiopulmonary return while suppressing
nuisance reflectors before analogue-to-digital conversion. BMI, denoted by \(b\),
is used only as a readily available proxy for morphology. The simulator does **not**
assume that BMI uniquely determines tissue permittivity or radar response.

## 2. Cardiopulmonary motion and radar phase

For observation time \(t\), the anterior chest displacement is

\[
d(t;b)=A_r(b)\sin(2\pi f_r t+\phi_r(t))
      +A_h(b)\sin(2\pi f_h t+\phi_h(t)),
\]

where \(f_r\) and \(f_h\) are respiration and heartbeat frequencies. Mild phase
modulation in \(\phi_r(t)\) and \(\phi_h(t)\) represents physiological
nonstationarity. A monostatic narrowband radar produces the complex return

\[
s_v(t;b)=\alpha(b)\exp\!\left(j\frac{4\pi}{\lambda}d(t;b)\right),
\qquad \lambda=\frac{c}{f_c}.
\]

The factor \(4\pi/\lambda\) accounts for round-trip propagation. The replaceable
default morphology maps are

\[
\alpha(b)=\exp[-0.035\max(b-22,0)],
\]

\[
A_r(b)=4.0\,\mathrm{mm}\,\exp[-0.012\max(b-22,0)],
\quad
A_h(b)=0.24\,\mathrm{mm}\,\exp[-0.020\max(b-22,0)].
\]

These equations are implemented in `sim_vitals/model.py:morphology_scalars`.

## 3. Spatial morphology surrogate

The field incident on the first SIM layer is not a scalar. For the vital target it
is a coherent mixture

\[
\mathbf a_v(b)=\frac{1}{C_b}\left[
(1-\rho_b)\mathbf a(\Omega_0(b))
+0.55\rho_b e^{j\psi_b}\mathbf a(\Omega_-(b))
+0.45\rho_b e^{-j0.7\psi_b}\mathbf a(\Omega_+(b))
\right],
\]

where \(C_b\) gives unit norm, \(\rho_b\) is a diffuse-scattering weight, and
\(\Omega_0,\Omega_-,\Omega_+\) are the central and diffuse angular components.
Their BMI-dependent values are explicit in
`sim_vitals/model.py:morphology_signature`.

The implementation also applies a BMI-conditioned quadratic phase curvature across
the aperture as a compact surrogate for altered effective scattering depth. Like
the attenuation mapping, this is a sensitivity-analysis parameter to be calibrated,
not an asserted population law.

This spatial model is essential. If all sources were described only by one global
phase, a static passive linear SIM could multiply the received signal but could not
separate a heartbeat from clutter. Distinct spatial signatures allow physically
meaningful analogue discrimination.

## 4. Stacked metasurface as a wave-domain neural network

For \(N\) elements per layer and \(L\) layers, layer \(\ell\) applies the diagonal
phase matrix

\[
\mathbf D_\ell(\boldsymbol\theta_\ell)
=\operatorname{diag}\{e^{j\theta_{\ell,1}},\ldots,e^{j\theta_{\ell,N}}\}.
\]

Inter-layer propagation starts from the scalar Green matrix, whose sampled entries
are

\[
[\widetilde{\mathbf P}_\ell]_{mn}
=\frac{\exp(jk r_{mn})}{r_{mn}},\qquad
r_{mn}=\sqrt{(x_m-x_n)^2+(y_m-y_n)^2+z_\ell^2},
\]

with \(k=2\pi/\lambda\). The sampled finite-aperture matrix is projected onto its
closest unitary polar factor,
\(\mathbf P_\ell=\mathbf U_\ell\mathbf V_\ell^H\) for
\(\widetilde{\mathbf P}_\ell=\mathbf U_\ell\boldsymbol\Sigma_\ell
\mathbf V_\ell^H\). This gives an ideal lossless modal propagation model; measured
insertion loss can later be applied separately. The field recursion is

\[
\mathbf u_{\ell+1}=\mathbf P_\ell\mathbf D_\ell\mathbf u_\ell.
\]

For detector mode \(\mathbf w\), the effective complex gain for spatial source
\(q\) is

\[
g_q(\boldsymbol\Theta)=\mathbf w^H
\mathbf P_L\mathbf D_L\cdots
\mathbf P_1\mathbf D_1\mathbf a_q.
\]

The phase masks are the trainable weights and free-space coupling supplies the
inter-layer connections. Unlike a conventional neural network, this prototype is
linear in the complex field; its useful selectivity comes from wave interference
and multiple spatial modes, not an assumed electronic activation function.

## 5. Received signal

With \(Q\) nuisance reflectors, the detector receives

\[
y(t;b)=g_v(\boldsymbol\Theta,b)s_v(t;b)
+\sum_{q=1}^{Q}\sqrt{p_q}\,g_q(\boldsymbol\Theta)c_q(t)+n(t),
\]

where \(c_q(t)\) includes static and micro-motion clutter and
\(n(t)\sim\mathcal{CN}(0,\sigma_n^2)\). The simulator uses nuisance components
inside and outside the vital bands to test whether spatial preprocessing improves
frequency recovery. A conventional stationary-clutter/IQ-centre calibration
subtracts the temporal mean of each nuisance component; moving-clutter residuals
remain and are processed by the SIM.

## 6. BMI-conditioned training objective

For morphology training set \(\mathcal B\), define

\[
S_b=\alpha(b)^2|g_v(\boldsymbol\Theta,b)|^2,
\qquad
I=\sum_{q=1}^{Q}p_q|g_q(\boldsymbol\Theta)|^2+\sigma_0^2.
\]

The phase-only optimization maximises the mean log-SINR

\[
\max_{\boldsymbol\Theta}\quad
\mathcal J(\boldsymbol\Theta)
=\frac{1}{|\mathcal B|}\sum_{b\in\mathcal B}\log S_b-\log I,
\quad \theta_{\ell,n}\in[-\pi,\pi).
\]

A universal SIM uses all BMI conditions in \(\mathcal B\). An adaptive SIM trains
a bank \(\{\boldsymbol\Theta_b\}\) and selects the nearest BMI condition at run
time. A later continuous controller can replace this lookup bank.

## 7. Electromagnetic backpropagation

Let \(\mathbf v_\ell=\mathbf D_\ell\mathbf u_\ell\), and propagate the detector
mode backwards to obtain

\[
\mathbf r_\ell=\mathbf P_\ell^H\mathbf q_{\ell+1},
\qquad
\mathbf q_\ell=\mathbf D_\ell^H\mathbf r_\ell,
\quad \mathbf q_L=\mathbf w.
\]

Then the exact phase derivative of source power is

\[
\frac{\partial |g|^2}{\partial\theta_{\ell,n}}
=2\operatorname{Re}\left\{
g^*\,jv_{\ell,n}r_{\ell,n}^*
\right\}.
\]

The code combines these derivatives by the chain rule for \(\mathcal J\) and uses
Adam for phase updates. The unit test checks the adjoint gradient against central
finite differences.

## 8. Hardware constraints

### 8.1 Finite phase resolution

For a controller with \(B\) bits, the continuous phase is projected onto
\(2^B\) uniformly spaced states:

\[
\widehat\theta_{\ell,n}^{(B)}
=\Delta_B\left\lfloor\frac{\theta_{\ell,n}}{\Delta_B}+\frac12\right\rfloor
\bmod 2\pi,
\qquad \Delta_B=\frac{2\pi}{2^B}.
\]

The sensitivity experiment applies this projection after continuous-phase
training. Consequently, results measure post-training quantization loss, not the
potentially smaller loss attainable with quantization-aware training. Non-monotonic
behaviour at very low bit depth is possible because a coarser codebook can
occasionally preserve a favourable interference null by chance.

### 8.2 Layer insertion loss and depth

If one layer has power insertion loss \(\xi_\ell\) dB, its field-transmission
coefficient is

\[
\eta_\ell=10^{-\xi_\ell/20}.
\]

The non-ideal field recursion becomes

\[
\mathbf u_{\ell+1}
=\eta_\ell\mathbf P_\ell\mathbf D_\ell\mathbf u_\ell.
\]

For identical layers the end-to-end signal power includes the factor
\(10^{-L\xi/10}\). The depth sweep therefore exposes the tradeoff between the
extra \(LN\) controllable phases and accumulated insertion loss. Its default
assumption is 0.5 dB per layer and is configurable with `--layer-loss-db`.

## 9. BMI-conditioning mismatch

Let \(b\) be the true digital-twin morphology value and \(\widehat b=b+\delta_b\)
the value supplied to the SIM controller. For codebook set \(\mathcal B_c\), the
implemented mask is

\[
\boldsymbol\Theta_{c(\widehat b)},\qquad
c(\widehat b)=\arg\min_{\beta\in\mathcal B_c}|\widehat b-\beta|.
\]

The mismatch study holds \(\mathbf a_v(b)\), attenuation, displacement and noise
fixed while changing only \(\delta_b\). Thus any SINR or rate-estimation change is
attributable to incorrect codebook selection rather than a different simulated
subject.

## 10. Fully digital comparison baseline

In addition to a single antenna and a one-layer RIS, the simulator includes an
\(N\)-channel digital minimum-variance distortionless-response (MVDR) array. With
nuisance covariance

\[
\mathbf R_i=\sum_{q=1}^{Q}p_q\mathbf a_q\mathbf a_q^H
+\sigma_0^2\mathbf I,
\]

define the morphology-robust target covariance from \(M\) training variants as

\[
\mathbf R_s(b)=\frac{1}{M}\sum_{m=1}^{M}
\mathbf a_{v,m}(b)\mathbf a_{v,m}^H(b).
\]

The dominant generalized eigenvector satisfying
\(\mathbf R_s\mathbf w=\lambda\mathbf R_i\mathbf w\) initializes the robust
MVDR-family combiner. It is then refined on the unit sphere using the same
mean-log-SINR criterion as the SIM. Its unconstrained complex gradient is

\[
\frac{\partial\mathcal J_d}{\partial\mathbf w^*}
=\frac{1}{M}\sum_{m=1}^{M}
\frac{\mathbf a_{v,m}\mathbf a_{v,m}^H\mathbf w}
{|\mathbf w^H\mathbf a_{v,m}|^2}
-\frac{\mathbf R_i\mathbf w}{\mathbf w^H\mathbf R_i\mathbf w},
\qquad \|\mathbf w\|_2=1.
\]

This is a strong signal-processing benchmark but requires one receiver/RF chain per
element. The SIM uses passive layers and one detector, so comparable performance
would represent an RF-chain-complexity advantage rather than proof of superiority
to unconstrained digital processing.

## 11. Vital-rate estimator and metrics

The receiver phase is unwrapped and linearly detrended. After Hann windowing, the
respiration estimate is the interpolated spectral maximum in 0.10--0.60 Hz and the
heart estimate is the corresponding maximum in 0.80--2.20 Hz. For \(K\) Monte
Carlo trials, rate RMSE is

\[
\operatorname{RMSE}_x
=\sqrt{\frac{1}{K}\sum_{k=1}^{K}
(\widehat f_{x,k}-f_{x,k})^2}\times 60,
\qquad x\in\{r,h\}.
\]

The reported spatial SINR is

\[
\Gamma_b=10\log_{10}\frac{
\alpha(b)^2|g_v(\boldsymbol\Theta,b)|^2}
{\sum_q p_q|g_q(\boldsymbol\Theta)|^2+\sigma_0^2}.
\]

Receiver SNR \(\gamma\) is referenced to the target power at one aperture element:

\[
\sigma_n^2=\frac{\alpha(b)^2}{N}10^{-\gamma/10}.
\]

This definition lets passive aperture gain appear in the SIM result without
assigning it an artificial noise penalty proportional to aperture size.

## 12. Monte Carlo protocol and reproducibility

The default robustness experiment uses:

- BMI codebook values \(\{18,22,26,30,34,38\}\) kg/m²;
- \(f_r\sim\mathcal U(0.18,0.42)\) Hz and
  \(f_h\sim\mathcal U(0.95,1.65)\) Hz;
- independent respiration and heartbeat amplitude perturbations with standard
  deviations of 8% and 10%;
- random lateral/elevation morphology perturbations;
- paired random seeds, so competing methods see identical subjects and noise draws;
- SNR sweep from -10 to 15 dB;
- 1, 2, 3, 4 and 6-bit phase control plus continuous control;
- one through five SIM layers;
- BMI-conditioning bias from -8 to +8 kg/m².

Raw trial-level CSV files are retained so confidence intervals, paired significance
tests and alternative estimators can be added without rerunning training.

## 13. Claims supported by this prototype

The simulation can support comparative claims **within the assumed model**, such
as adaptive versus universal masks, passive SIM versus digital MVDR processing,
layer-depth/insertion-loss tradeoffs, finite phase resolution, robustness to noise,
and sensitivity to morphology mismatch. It cannot by itself support clinical
accuracy, safety, population-level BMI dependence, or a full-wave hardware claim.

The no-surface baseline is one conventional receiving antenna. The digital MVDR
baseline has the same spatial aperture as the SIM but substantially greater RF-chain
complexity. Final paper claims should show both comparisons and label the assumed
morphology and insertion-loss mappings explicitly.
