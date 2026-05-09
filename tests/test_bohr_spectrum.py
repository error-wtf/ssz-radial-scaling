"""test_bohr_spectrum.py
Test: RSG + WKB + Langer reproduces exact Bohr energy spectrum.

Central validation from:
  Wrede, Casu, Bingsi -- Radial Scaling Gauge in Quantum Mechanics (2025)

Verifies that WITHOUT the TDSE, using only TISE + RSG + WKB,
we recover E_n = -1/(2n^2) exactly (atomic units).
"""

import numpy as np
import pytest
from rsg_core import bohr_energy_exact
from rsg_coulomb import solve_bohr_spectrum_rsg

WKB_TOL = 0.001  # 0.1% relative error tolerance


class TestBohrSpectrum:
    """Bohr spectrum via RSG WKB quantization."""

    def test_exact_energy_formula(self):
        """Analytical Bohr formula E_n = -1/(2n^2) in atomic units."""
        for n in range(1, 6):
            E = bohr_energy_exact(n)
            expected = -1.0 / (2.0 * n**2)
            assert abs(E - expected) < 1e-12, f"E_{n} = {E}, expected {expected}"

    def test_hydrogen_ground_state_ev(self):
        """Ground state energy = -13.6 eV (= -0.5 Hartree)."""
        E_hartree = bohr_energy_exact(1)
        assert abs(E_hartree - (-0.5)) < 1e-12
        E_eV = E_hartree * 27.2114
        assert abs(E_eV - (-13.6057)) < 0.001

    def test_wkb_langer_l0(self):
        """RSG WKB with Langer, l=0: n=1,2,3."""
        results = solve_bohr_spectrum_rsg(n_max=3, l=0)
        for n, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"WKB failed for n={n}, l=0"
            assert rel_err < WKB_TOL, (
                f"n={n}, l=0: E_wkb={E_wkb:.6f}, E_exact={E_exact:.6f}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_wkb_langer_l1(self):
        """RSG WKB with Langer, l=1: n=2,3,4."""
        results = solve_bohr_spectrum_rsg(n_max=3, l=1)
        for n, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"WKB failed for n={n}, l=1"
            assert rel_err < WKB_TOL, (
                f"n={n}, l=1: E_wkb={E_wkb:.6f}, E_exact={E_exact:.6f}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_wkb_langer_l2(self):
        """RSG WKB with Langer, l=2: n=3,4,5."""
        results = solve_bohr_spectrum_rsg(n_max=3, l=2)
        for n, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"WKB failed for n={n}, l=2"
            assert rel_err < WKB_TOL, (
                f"n={n}, l=2: E_wkb={E_wkb:.6f}, E_exact={E_exact:.6f}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_energy_ordering(self):
        """E_1 < E_2 < E_3 < ... < 0."""
        energies = [bohr_energy_exact(n) for n in range(1, 6)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1] < 0
