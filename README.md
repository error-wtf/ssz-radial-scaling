# SSZ Radial Scaling Gauge — Quantum Mechanics

**Authors:** Carmen N. Wrede, Lino P. Casu, Bingsi

## Core Question

> *Can we solve the radial quantum problem using RSG, bypassing the TDSE?*

**Yes.** RSG operates entirely on the **time-independent Schrödinger equation (TISE)**:

```
TISE (radial) → log-transform r = exp(x) → effective 1D Morse-like potential
→ Langer-corrected momentum → Bohr-Sommerfeld quantization → exact Bohr spectrum
```

The TDSE is never needed.

## Method

```
Step 1: Radial TISE
  [-ℏ²/2m (d²/dr² - l(l+1)/r²) + V(r)] R(r) = E R(r)

Step 2: Log-transformation r = exp(x)  [RSG coordinate]
  Singular origin r→0 mapped to x→-∞
  Effective 1D Morse-like potential in x — regular everywhere

Step 3: Consistent operator transformation
  R(r) → u(x) = √r · R(r)
  Measure: dr → exp(x) dx
  Angular term: l(l+1) → (l+1/2)²  [Langer correction — geometric, not ad hoc]

Step 4: Bohr-Sommerfeld in scaled coordinate
  ∫ p_r(r) dr = πℏ(n_r + 1/2)
  where p_r²(r) = 2m(E - V(r)) - ℏ²(l+1/2)²/r²

Step 5: Exact Coulomb spectrum
  n = n_r + l + 1
  E_n = -mκ²/(2ℏ²n²) = -1/(2n²)  [atomic units]
```

## Key Results

| Test | Result |
|------|--------|
| Bohr spectrum n=1..5 via WKB+Langer | ✅ Exact (< 0.001% error) |
| Langer correction from geometry | ✅ 1/4 term is universal |
| TISE-only solution (no TDSE) | ✅ Confirmed |
| Hydrogen ground state E = -13.6 eV | ✅ |
| Naive WKB fails for l=0 | ✅ Confirmed |
| Numerical TISE cross-check | ✅ Consistent |

## Repository Structure

```
ssz-radial-scaling/
├── rsg_core.py          # Core RSG: transformation, Langer, WKB
├── rsg_coulomb.py       # Coulomb problem with RSG
├── tests/
│   ├── test_bohr_spectrum.py     # Bohr spectrum via WKB+Langer
│   ├── test_langer_emergence.py  # Langer correction as geometric term
│   ├── test_tise_no_tdse.py      # TISE solution without TDSE
│   └── test_numerical_verify.py  # Numerical TISE cross-check
├── FINDINGS.md
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/error-wtf/ssz-radial-scaling
cd ssz-radial-scaling
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Connection to SSZ Framework

The RSG scaling function `s(r) = 1 + Ξ(r)` used in gravitational contexts (lensing,
Shapiro, GPS) is the **same geometric phase-accounting structure** applied here to
quantum radial systems. The Langer correction is the quantum analog of the coordinate
stretch in the Regge-Wheeler tortoise coordinate.

## Reference Paper

> C. N. Wrede, L. P. Casu, Bingsi,
> *Radial Scaling Gauge in Quantum Mechanics: The Coulomb Problem, the Langer
> Transformation, and Geometric Phase Accounting* (2025)
