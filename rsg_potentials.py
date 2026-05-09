"""rsg_potentials.py
RSG extension to non-Coulomb potentials.

Tests whether the RSG + WKB + Langer framework (proven exact for Coulomb)
also works for other radial potentials. The paper explicitly states this
is NOT guaranteed -- this module quantifies HOW WELL it works and WHERE
it breaks down.

Potentials tested:
  1. 3D Harmonic Oscillator  -- WKB exact with Langer (like Coulomb)
  2. Morse potential          -- WKB approximate (finite anharmonicity)
  3. Kratzer potential        -- WKB nearly exact (Coulomb-like structure)
  4. Power-law V = A*r^p      -- tests RSG scaling with potential shape

Exact energies (where known):
  HO:     E(n,l) = hbar*omega*(2n + l + 3/2)   [n = radial quantum number]
  Morse:  E_v = hbar*omega_e*(v+1/2) - hbar*omega_e*x_e*(v+1/2)^2
  Kratzer: E(n,l) = -m*D_e^2*r_e^4 / (2*hbar^2*(n_r+gamma)^2)
           gamma = 0.5 + sqrt((l+0.5)^2 + 2*m*D_e*r_e^2/hbar^2)

Authors: Carmen N. Wrede, Lino P. Casu, Bingsi
"""

import numpy as np
from rsg_core import bohr_sommerfeld_energy, HBAR, M_E


# ---------------------------------------------------------------------------
# 1. 3D Isotropic Harmonic Oscillator
# ---------------------------------------------------------------------------

def harmonic_potential(r, omega=1.0, m=M_E):
    """3D isotropic harmonic oscillator V(r) = 0.5 * m * omega^2 * r^2."""
    return 0.5 * m * omega**2 * r**2


def ho_energy_exact(n_r, l, omega=1.0, hbar=HBAR):
    """Exact 3D HO energy: E = hbar*omega*(2*n_r + l + 3/2).

    n_r = radial quantum number (0, 1, 2, ...)
    l   = angular momentum quantum number
    N   = 2*n_r + l = shell number
    """
    return hbar * omega * (2.0 * n_r + l + 1.5)


def solve_ho_spectrum_rsg(n_max=4, l=0, omega=1.0,
                          hbar=HBAR, m=M_E):
    """Solve 3D HO spectrum using RSG + WKB + Langer.

    Returns list of (n_r, E_wkb, E_exact, rel_error).
    """
    V = lambda r: harmonic_potential(r, omega, m)
    results = []
    for n_r in range(n_max):
        E_exact = ho_energy_exact(n_r, l, omega, hbar)
        E_min = E_exact * 0.3
        E_max = E_exact * 3.0
        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            rel_err = abs(E_wkb - E_exact) / abs(E_exact)
            results.append((n_r, E_wkb, E_exact, rel_err))
        except Exception:
            results.append((n_r, float('nan'), E_exact, float('nan')))
    return results


# ---------------------------------------------------------------------------
# 2. Morse Potential
# ---------------------------------------------------------------------------

def morse_potential(r, D_e=10.0, alpha=1.0, r_e=2.0):
    """Morse potential: V(r) = D_e * (1 - exp(-alpha*(r-r_e)))^2 - D_e.

    Parameters (in atomic units, typical diatomic-like):
      D_e   = dissociation energy (depth)
      alpha = well width parameter
      r_e   = equilibrium bond length
    """
    return D_e * (1.0 - np.exp(-alpha * (r - r_e)))**2 - D_e


def morse_energy_exact(v, D_e=10.0, alpha=1.0, m=M_E, hbar=HBAR):
    """Exact Morse energy levels:

    E_v = -D_e + hbar*omega_e*(v+1/2) - (hbar*omega_e)^2*(v+1/2)^2/(4*D_e)

    where omega_e = alpha * sqrt(2*D_e/m)
    """
    omega_e = alpha * np.sqrt(2.0 * D_e / m)
    xe = hbar * omega_e / (4.0 * D_e)  # anharmonicity
    v_half = v + 0.5
    return -D_e + hbar * omega_e * v_half - hbar * omega_e * xe * v_half**2


def morse_v_max(D_e=10.0, alpha=1.0, m=M_E, hbar=HBAR):
    """Maximum bound vibrational quantum number for Morse potential."""
    omega_e = alpha * np.sqrt(2.0 * D_e / m)
    xe = hbar * omega_e / (4.0 * D_e)
    # E_v < 0 requires v < 1/(2*xe) - 1/2
    return int(np.floor(1.0 / (2.0 * xe) - 0.5))


def solve_morse_spectrum_rsg(v_max=5, l=0, D_e=10.0, alpha=1.0, r_e=2.0,
                             hbar=HBAR, m=M_E):
    """Solve Morse bound states using RSG + WKB + Langer.

    Returns list of (v, E_wkb, E_exact, rel_error).
    Note: l=0 for pure vibrational states (no angular momentum).
    """
    V = lambda r: morse_potential(r, D_e, alpha, r_e)
    results = []
    for v in range(v_max):
        E_exact = morse_energy_exact(v, D_e, alpha, m, hbar)
        if E_exact >= 0:
            break  # unbound
        E_min = -D_e * 1.05
        E_max = -1e-4
        try:
            E_wkb = bohr_sommerfeld_energy(
                v, l, V,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            rel_err = abs(E_wkb - E_exact) / abs(E_exact)
            results.append((v, E_wkb, E_exact, rel_err))
        except Exception:
            results.append((v, float('nan'), E_exact, float('nan')))
    return results


# ---------------------------------------------------------------------------
# 3. Kratzer Potential
# ---------------------------------------------------------------------------

def kratzer_potential(r, D_e=5.0, r_e=2.0):
    """Kratzer potential: V(r) = D_e * [(r_e/r)^2 - 2*(r_e/r)].

    Coulomb-like at large r, repulsive at small r.
    Used for molecular rotation-vibration spectra.
    """
    x = r_e / r
    return D_e * (x**2 - 2.0 * x)


def kratzer_energy_exact(n_r, l, D_e=5.0, r_e=2.0, hbar=HBAR, m=M_E):
    """Exact Kratzer energy levels.

    E(n_r, l) = -m * D_e^2 * r_e^4 / (2 * hbar^2 * N^2)
    where N = n_r + gamma
    and   gamma = 0.5 + sqrt((l+0.5)^2 + 2*m*D_e*r_e^2/hbar^2)

    In atomic units (hbar=m=1):
      gamma = 0.5 + sqrt((l+0.5)^2 + 2*D_e*r_e^2)
    """
    langer_l = (l + 0.5)**2
    gamma = 0.5 + np.sqrt(langer_l + 2.0 * m * D_e * r_e**2 / hbar**2)
    N = n_r + gamma
    return -m * D_e**2 * r_e**4 / (2.0 * hbar**2 * N**2)


def solve_kratzer_spectrum_rsg(n_max=4, l=0, D_e=5.0, r_e=2.0,
                               hbar=HBAR, m=M_E):
    """Solve Kratzer spectrum using RSG + WKB + Langer.

    Returns list of (n_r, E_wkb, E_exact, rel_error).
    """
    V = lambda r: kratzer_potential(r, D_e, r_e)
    results = []
    for n_r in range(n_max):
        try:
            E_exact = kratzer_energy_exact(n_r, l, D_e, r_e, hbar, m)
        except Exception:
            continue
        if E_exact >= 0:
            continue
        E_min = 2.0 * E_exact
        E_max = -1e-6
        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            rel_err = abs(E_wkb - E_exact) / abs(E_exact)
            results.append((n_r, E_wkb, E_exact, rel_err))
        except Exception:
            results.append((n_r, float('nan'), E_exact, float('nan')))
    return results


# ---------------------------------------------------------------------------
# Helper: generic WKB scan
# ---------------------------------------------------------------------------

def wkb_scan(V_func, n_r, l, E_min, E_max, label="",
             use_langer=True, hbar=HBAR, m=M_E):
    """Generic WKB energy scan for any potential.

    Returns (E_wkb, success).
    """
    try:
        E = bohr_sommerfeld_energy(
            n_r, l, V_func,
            E_min=E_min, E_max=E_max,
            use_langer=use_langer, hbar=hbar, m=m
        )
        return E, True
    except Exception:
        return float('nan'), False
