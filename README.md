# Monte Carlo Simulation of Electron Drift in a Copper Wire

A Monte Carlo simulation that independently verifies the Drude model's prediction for electron drift velocity in a copper wire under an applied electric field. The simulation recovers the theoretical drift velocity — including the classical factor of 1/2 — purely from statistical averaging over random scattering events.

## Physics Background

In the Drude model, conduction electrons travel through a metal lattice at high thermal velocity, scattering repeatedly off phonons (lattice vibrations). In the presence of an electric field **E**, the electron accelerates between collisions, producing a net drift velocity:

$$v_{dr} = \frac{\lambda_f e E}{2 m_e v_{th}}$$

The factor of 1/2 arises from averaging the field-induced velocity over exponentially distributed free-flight times. This simulation verifies that this factor emerges naturally from the underlying statistics, without assuming it directly.

**Two sources of randomness are modeled:**
- **Scattering angles**: Each collision randomizes the electron's direction while preserving speed. The azimuthal angle is sampled as φ ~ U(0, 2π), and the polar angle via cos(θ) ~ U(−1, 1) — the latter ensuring uniform coverage of the sphere rather than oversampling the poles.
- **Free-flight time**: Collisions follow a Poisson process, so the time between them is sampled as t ~ Exp(τ), where τ = λ_f / v_th is the mean free time.

## Simulation Design

Rather than running one very long trajectory (which would require ~10⁸ steps to overcome thermal noise), the simulation runs **M = 1000 independent trajectories** of **N = 10⁵ scattering events** each. For each free-flight step, the mean x-velocity during the flight is recorded as:

$$\bar{v}_x = v_x(0) + \frac{1}{2} a \Delta t$$

The drift velocity for each trajectory is the average of these per-flight means. Taking the mean across all M trajectories recovers the Drude prediction.

## Results

| Parameter | Value |
|---|---|
| Theoretical drift velocity (½aτ) | 2184.61 m/s |
| Simulated mean x-velocity | 2027.77 m/s |
| Percent error | 7.18% |

The y and z components average to ~0 m/s, confirming no directional bias from the scattering model. The distribution of drift velocities across simulations is Gaussian, consistent with the Central Limit Theorem applied to the many independent scattering events.

## Parameters

| Symbol | Value | Description |
|---|---|---|
| E | 10⁶ N/C | Applied electric field |
| v_th | 1.57 × 10⁶ m/s | Electron thermal velocity |
| λ_f | 3.9 × 10⁻⁸ m | Mean free path (copper) |
| τ | λ_f / v_th | Mean free time |
| M | 1000 | Number of independent simulations |
| N | 10⁵ | Scattering events per simulation |

## Requirements

```
numpy
matplotlib
numba
pandas
seaborn
```

Install with:
```bash
pip install numpy matplotlib numba pandas seaborn
```

> **Note:** `numba` is used to JIT-compile the core simulation loop and rotation function, giving a significant speedup for M × N = 10⁸ total iterations. It is a required dependency. Runtime is approximately 2 minutes on a standard laptop; the first run may be slightly slower due to JIT compilation overhead.

## Usage

```bash
python 414H_Monte_Carlo.py
```

The script produces:
- **Velocity distributions**: Histograms of mean drift velocity in x, y, and z across all M simulations, with fitted Gaussian curves
- **Bar chart**: Mean drift velocity per direction with standard deviation error bars

## References

1. University of Illinois, ECE 539 — *Introduction to Monte Carlo Simulation* (2012)
2. Columbia University, Department of Physics — *The Drude Model* (2024)
