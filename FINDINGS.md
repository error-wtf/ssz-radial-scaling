# RSG Quantum Mechanics -- Findings

## Central Question

> **Can we solve quantum systems using RSG without the TDSE?**

## Answer: YES

The Radial Scaling Gauge operates entirely on the **TISE (time-independent SE)**.
No time evolution, no propagator, no TDSE required.

## The Transformation Chain

```
Step 1: Radial TISE
  [-hbar^2/2m (d^2/dr^2 - l(l+1)/r^2) + V(r)] R(r) = E R(r)

Step 2: RSG log-transformation  r = exp(x)
  r -> 0  maps to  x -> -inf  (singularity removed from finite domain)
  Effective 1D Morse-like potential -- regular everywhere

Step 3: Consistent operator transformation
  R(r) -> u(x) = sqrt(r) * R(r)
  Measure:        dr -> exp(x) dx
  Angular term:   l(l+1) -> (l+1/2)^2   [Langer: geometric, not ad hoc]

Step 4: Bohr-Sommerfeld quantization
  integral p_r(r) dr = pi*hbar*(n_r + 1/2)
  p_r^2(r) = 2m(E - V(r)) - hbar^2*(l+1/2)^2/r^2

Step 5: Exact Coulomb spectrum
  n = n_r + l + 1
  E_n = -1/(2n^2)  [atomic units]
```

## Test Results

### Bohr Spectrum (l=0, atomic units)

| n | E_WKB+RSG | E_exact   | rel. error |
|---|-----------|-----------|------------|
| 1 | -0.500000 | -0.500000 | < 0.001%   |
| 2 | -0.125000 | -0.125000 | < 0.001%   |
| 3 | -0.055556 | -0.055556 | < 0.001%   |
| 4 | -0.031250 | -0.031250 | < 0.001%   |
| 5 | -0.020000 | -0.020000 | < 0.001%   |

### Langer Correction Analysis

| l | naive l(l+1) | Langer (l+1/2)^2 | diff   |
|---|-------------|------------------|--------|
| 0 | 0           | 0.25             | 0.25   |
| 1 | 2           | 2.25             | 0.25   |
| 2 | 6           | 6.25             | 0.25   |

The extra 1/4 is **universal** -- same for all l.
It arises from the log-transform geometry, not from fitting.

## Why No TDSE?

The TDSE: `i*hbar * d-psi/dt = H * psi`

For stationary states it factors as:
```
psi(r, t) = R(r) * exp(-iEt/hbar)
```
The spatial part satisfies: `H * R(r) = E * R(r)` -- the TISE.

RSG solves the TISE directly. The TDSE adds only the phase clock `exp(-iEt/hbar)`.
No time evolution is needed to find energy eigenvalues.

## Geometric Interpretation

The RSG scaling function `s(r)` acts as a **geometric phase-accounting factor**:

- **Gravity** (SSZ): `s(r) = 1 + Xi(r)` -- accounts for curved-spacetime
  phase accumulation (lensing, Shapiro, GPS)
- **Quantum** (RSG): `s(r) = r` (log-scaling) -- accounts for radial measure
  and singular boundary

Same principle:
> Replace naive coordinate distance with geometrically-weighted path length.

## Connection to SSZ Validation (28/28 tests)

The same geometric phase-accounting structure that explains gravitational
observables (28 tests, no free parameters) also explains the Langer
correction in quantum mechanics. The scaling function is fixed by geometry.

## Limitations

1. Semiclassical only -- full operator formulation pending
2. Radial systems -- non-radial extensions need work
3. Coulomb is special -- not all potentials admit exact WKB
4. Not a replacement for QM -- a geometric language for phase counting

## Run Tests

```bash
pip install numpy scipy pytest
python -m pytest tests/ -v
```
