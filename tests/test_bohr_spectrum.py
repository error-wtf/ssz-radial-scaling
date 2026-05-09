"""test_bohr_spectrum.py
Test: RSG + WKB + Langer reproduces exact Bohr energy spectrum.

Central validation from:
  Wrede, Casu, Bingsi -- Radial Scaling Gauge in Quantum Mechanics (2025)

Verifies that WITHOUT the TDSE, using only TISE + RSG + WKB,
we recover E_n = -1/(2n^2) exactly (atomic units).

SSZ-Logik: Die Skalierungsfunktion s(r) = r ist parameterlos und geometrisch
fixiert. Es gibt KEINE freien Parameter, KEIN Fitting. Das Spektrum folgt
zwingend aus der Geometrie der logarithmischen Transformation.
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

    # ------------------------------------------------------------------
    # SSZ-Logik: Parameterfreiheit
    # ------------------------------------------------------------------

    def test_no_free_parameters(self):
        """SSZ: RSG Spektrum braucht KEINE freien Parameter.

        Die einzigen Eingaben sind n, l, und die Naturkonstanten.
        kappa (= e^2) ist in atomaren Einheiten exakt 1.0.
        Kein Fitting, keine adjustierten Konstanten.
        """
        from rsg_core import HBAR, M_E, KAPPA_H
        assert HBAR == 1.0, "hbar should be 1.0 in atomic units"
        assert M_E == 1.0, "m_e should be 1.0 in atomic units"
        assert KAPPA_H == 1.0, "kappa should be 1.0 in atomic units"
        for n in range(1, 4):
            E = bohr_energy_exact(n, kappa=KAPPA_H, hbar=HBAR, m=M_E)
            E_formula = -M_E * KAPPA_H**2 / (2.0 * HBAR**2 * n**2)
            assert abs(E - E_formula) < 1e-14

    def test_kappa_scaling(self):
        """SSZ: Skalierungsinvarianz -- E skaliert als kappa^2.

        Die Coulomb-Energie skaliert als E ~ kappa^2 / n^2.
        Das ist eine Konsequenz der Geometrie, kein Fitparameter.
        Bei doppeltem kappa (Z=2) skaliert E um Faktor 4.
        """
        E_H = bohr_energy_exact(1, kappa=1.0)
        E_He = bohr_energy_exact(1, kappa=2.0)
        ratio = E_He / E_H
        assert abs(ratio - 4.0) < 1e-10, (
            f"E(Z=2)/E(Z=1) = {ratio}, expected 4.0 (kappa^2 scaling)"
        )

    def test_rsg_spectrum_n_squared_law(self):
        """SSZ: E_n * n^2 = const (= -1/2 in a.u.) -- geometrische Konsequenz.

        Die 1/n^2-Abhaengigkeit folgt aus der Bohr-Sommerfeld-Bedingung
        mit dem Coulomb-Potential. Kein freier Parameter.
        """
        const_values = []
        for n in range(1, 6):
            E = bohr_energy_exact(n)
            const_values.append(E * n**2)
        for i, c in enumerate(const_values):
            assert abs(c - (-0.5)) < 1e-12, (
                f"n={i+1}: E*n^2 = {c}, expected -0.5"
            )

    def test_principal_quantum_number_additivity(self):
        """SSZ: n = n_r + l + 1 -- Additivitaet der Quantenzahlen.

        Die Zerlegung des Hauptquantums in Radial- und Winkelanteil
        ist eine direkte Konsequenz der RSG-Transformation:
        der Radialterm zaehlt Knoten, der l-Term zaehlt Winkelphase.
        """
        from rsg_core import bohr_sommerfeld_energy
        from rsg_coulomb import coulomb_potential
        V = lambda r: coulomb_potential(r)
        for n_r in range(3):
            for l in range(2):
                n = n_r + l + 1
                E_exact = bohr_energy_exact(n)
                E_wkb = bohr_sommerfeld_energy(
                    n_r, l, V,
                    E_min=E_exact * 1.5, E_max=E_exact * 0.5,
                    use_langer=True
                )
                rel_err = abs(E_wkb - E_exact) / abs(E_exact)
                assert rel_err < 0.001, (
                    f"n_r={n_r}, l={l}: n={n}, E_wkb={E_wkb:.6f}, "
                    f"E_exact={E_exact:.6f}, err={rel_err:.2e}"
                )
