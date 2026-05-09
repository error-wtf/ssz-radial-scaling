"""test_numerical_verify.py
Numerical verification: RSG/WKB vs direct TISE integration.

Anti-circular: uses scipy.integrate.solve_ivp (DOP853 adaptive ODE solver)
completely independent of the WKB/RSG code path.

The radial TISE for hydrogen:
  -1/2 u_rr + [l(l+1)/(2r^2) - 1/r] u = E u
  u(0) = 0, u(inf) -> 0

Eigenvalues: E_n = -1/(2n^2) in atomic units.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from rsg_core import bohr_energy_exact, bohr_sommerfeld_energy
from rsg_coulomb import coulomb_potential


def shoot(E, l=0, kappa=1.0, r_max_factor=5.0):
    """Shoot from r_min outward; returns u(r_max).

    Eigenvalue condition: u(r_max) -> 0 with correct exponential decay.
    Uses DOP853 adaptive solver -- stable and accurate.
    r_max is chosen as r_max_factor * outer_turning_point.
    """
    r_min = 1e-3
    r_tp = max(-kappa / E, 1.0)
    r_max = r_tp * r_max_factor

    def rhs(r, y):
        u, du = y
        ddu = (l * (l + 1) / r**2 + 2.0 * ((-kappa / r) - E)) * u
        return [du, ddu]

    # Initial conditions: u ~ r^(l+1) near origin
    u0 = r_min**(l + 1)
    du0 = (l + 1) * r_min**l
    sol = solve_ivp(rhs, [r_min, r_max], [u0, du0],
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=False)
    if not sol.success:
        return float('nan')
    return float(sol.y[0, -1])


def find_eigenvalue(n_r, l=0, kappa=1.0, n_scan=300):
    """Find energy eigenvalue by bracketing sign changes of u(r_max).

    n_r = radial node count (n_r = n - l - 1).
    Searches in [1.3 * E_exact, 0.7 * E_exact] around the known exact energy.
    """
    n = n_r + l + 1
    E_exact = -kappa**2 / (2.0 * n**2)
    E_lo = E_exact * 1.3
    E_hi = E_exact * 0.7

    E_scan = np.linspace(E_lo, E_hi, n_scan)
    vals = [shoot(E, l, kappa) for E in E_scan]

    for i in range(len(E_scan) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if not (np.isnan(v1) or np.isnan(v2)) and v1 * v2 < 0:
            return brentq(lambda E: shoot(E, l, kappa),
                          E_scan[i], E_scan[i + 1],
                          xtol=1e-10)
    raise ValueError(f"No eigenvalue for n_r={n_r}, l={l}")


class TestNumericalVerification:
    """Cross-check RSG/WKB against direct numerical TISE (solve_ivp)."""

    def test_numerical_n1_l0(self):
        """Numerical TISE: E(n=1, l=0) matches Bohr formula."""
        E_exact = bohr_energy_exact(1)
        E_num = find_eigenvalue(0, l=0)
        rel_err = abs(E_num - E_exact) / abs(E_exact)
        assert rel_err < 0.001, (
            f"Numerical E_1={E_num:.6f}, exact={E_exact:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_numerical_n2_l0(self):
        """Numerical TISE: E(n=2, l=0) matches Bohr formula."""
        E_exact = bohr_energy_exact(2)
        E_num = find_eigenvalue(1, l=0)
        rel_err = abs(E_num - E_exact) / abs(E_exact)
        assert rel_err < 0.001, (
            f"Numerical E_2={E_num:.6f}, exact={E_exact:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_numerical_n3_l1(self):
        """Numerical TISE: E(n=3, l=1) -- n_r=1, l=1."""
        E_exact = bohr_energy_exact(3)
        E_num = find_eigenvalue(1, l=1)
        rel_err = abs(E_num - E_exact) / abs(E_exact)
        assert rel_err < 0.001, (
            f"Numerical E(n=3,l=1)={E_num:.6f}, exact={E_exact:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_numerical_n4_l0(self):
        """Numerical TISE: E(n=4, l=0) matches Bohr formula."""
        E_exact = bohr_energy_exact(4)
        E_num = find_eigenvalue(3, l=0)
        rel_err = abs(E_num - E_exact) / abs(E_exact)
        assert rel_err < 0.001, (
            f"Numerical E_4={E_num:.6f}, exact={E_exact:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_wkb_vs_numerical_n1(self):
        """RSG WKB and numerical TISE agree for n=1, l=0."""
        V = lambda r: coulomb_potential(r)
        E_exact = bohr_energy_exact(1)
        E_wkb = bohr_sommerfeld_energy(
            0, 0, V, E_min=E_exact * 1.5, E_max=E_exact * 0.5,
            use_langer=True
        )
        E_num = find_eigenvalue(0, l=0)
        # Both should be within 0.1% of the exact value
        assert abs(E_wkb - E_exact) / abs(E_exact) < 0.001
        assert abs(E_num - E_exact) / abs(E_exact) < 0.001
        # And agree with each other within 0.2%
        assert abs(E_wkb - E_num) / abs(E_exact) < 0.002, (
            f"WKB={E_wkb:.6f} vs Numerical={E_num:.6f}: disagree"
        )

    def test_wkb_vs_numerical_n2(self):
        """RSG WKB and numerical TISE agree for n=2, l=0."""
        V = lambda r: coulomb_potential(r)
        E_exact = bohr_energy_exact(2)
        E_wkb = bohr_sommerfeld_energy(
            1, 0, V, E_min=E_exact * 1.5, E_max=E_exact * 0.5,
            use_langer=True
        )
        E_num = find_eigenvalue(1, l=0)
        assert abs(E_wkb - E_exact) / abs(E_exact) < 0.001
        assert abs(E_num - E_exact) / abs(E_exact) < 0.001
        assert abs(E_wkb - E_num) / abs(E_exact) < 0.002

    def test_numerical_spectrum_monotone(self):
        """Numerically found eigenvalues E_1 < E_2 < E_3 < 0."""
        energies = [find_eigenvalue(n_r, l=0) for n_r in range(3)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1] < 0, (
                f"Spectrum not ordered at i={i}: {energies}"
            )
