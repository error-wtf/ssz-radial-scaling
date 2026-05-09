"""rsg_coulomb.py
Coulomb problem solved via Radial Scaling Gauge.

Demonstrates TISE solution WITHOUT the TDSE, using only:
  RSG log-transformation + Langer correction + WKB quantization

Authors: Carmen N. Wrede, Lino P. Casu, Bingsi
"""

import numpy as np
from rsg_core import (
    bohr_energy_exact, bohr_sommerfeld_energy,
    langer_angular_term, radial_momentum_langer,
    find_turning_points, HBAR, M_E, KAPPA_H
)


def coulomb_potential(r, kappa=KAPPA_H):
    """Attractive Coulomb potential V(r) = -kappa/r."""
    return -kappa / r


def effective_potential_rsg(x, l, E=0.0, kappa=KAPPA_H, hbar=HBAR, m=M_E):
    """Effective potential in RSG (log-scaled) coordinate x = log(r).

    After transformation r = exp(x), u(x) = sqrt(r) * R(r):

      V_eff(x) = -kappa*exp(x) - E*exp(2x) + hbar^2*(l+1/2)^2/(2m)

    This is a Morse-like potential -- regular everywhere in x.
    The Coulomb singularity is absorbed into x -> -inf.
    """
    r = np.exp(x)
    langer = langer_angular_term(l)
    return -kappa * r - E * r**2 + hbar**2 * langer / (2.0 * m)


def solve_bohr_spectrum_rsg(n_max=5, l=0, kappa=KAPPA_H, hbar=HBAR, m=M_E):
    """Solve Coulomb bound states using RSG + WKB + Langer.

    Returns list of (n, E_wkb, E_exact, rel_error) for
    n = l+1, l+2, ..., l+n_max.

    Search range: from 2 * E_exact (deeper than exact) up to -1e-6.
    This is wide enough to bracket any bound state for n up to ~10.
    """
    V = lambda r: coulomb_potential(r, kappa)
    results = []

    for n_r in range(n_max):
        n = n_r + l + 1
        E_exact = bohr_energy_exact(n, kappa, hbar, m)

        # Search from twice as deep as exact energy to just below zero.
        # E_exact is negative, so 2*E_exact is deeper (more negative).
        E_search_min = 2.0 * E_exact
        E_search_max = -1e-6

        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V,
                E_min=E_search_min, E_max=E_search_max,
                use_langer=True, hbar=hbar, m=m
            )
            rel_err = abs(E_wkb - E_exact) / abs(E_exact)
            results.append((n, E_wkb, E_exact, rel_err))
        except Exception:
            results.append((n, float('nan'), E_exact, float('nan')))

    return results


def rsg_phase_accumulation(r_array, E, l, kappa=KAPPA_H, hbar=HBAR, m=M_E):
    """Compute RSG phase accumulation phi(r) = integral p_r dr."""
    V = lambda r: coulomb_potential(r, kappa)
    r1, r2 = find_turning_points(E, V, l, use_langer=True, hbar=hbar, m=m)

    r_inner = r_array[(r_array >= r1) & (r_array <= r2)]
    if len(r_inner) == 0:
        return np.array([]), np.array([])

    pr2 = radial_momentum_langer(r_inner, E, V, l, hbar, m)
    pr = np.where(pr2 > 0, np.sqrt(pr2), 0.0)

    phase = np.zeros(len(r_inner))
    for i in range(1, len(r_inner)):
        dr = r_inner[i] - r_inner[i - 1]
        phase[i] = phase[i - 1] + 0.5 * (pr[i - 1] + pr[i]) * dr

    return r_inner, phase
