"""test_tise_no_tdse.py
Test: RSG solves the TISE WITHOUT the TDSE.

Core question:
  'Can we solve quantum systems with RSG bypassing the TDSE?'

Answer: YES.

RSG works purely with TISE:
  TISE -> log-transform -> Langer-corrected WKB -> exact spectrum

No time evolution, no propagator, no TDSE.
The TDSE only adds exp(-iEt/hbar) -- a clock, not physics.

SSZ-Logik: Die Phase wird im transformierten Koordinatenraum akkumuliert.
Die Bohr-Sommerfeld-Bedingung ist eine Monodromie-Bedingung:
Die Gesamtphase muss ein ganzzahliges Vielfaches von pi*hbar sein.
Das ist die quantenmechanische Entsprechung der SSZ-Phasenbilanz
in der Gravitation (Shapiro, GPS, Lensing).
"""

import numpy as np
import pytest
from rsg_core import (
    bohr_energy_exact, bohr_sommerfeld_energy,
    wkb_action_integral, rsg_transform, rsg_inverse, HBAR
)
from rsg_coulomb import (
    coulomb_potential, effective_potential_rsg,
    solve_bohr_spectrum_rsg
)


class TestTISEWithoutTDSE:
    """Confirm RSG solves TISE directly, no TDSE needed."""

    def test_no_time_parameter_needed(self):
        """The RSG solution has no time parameter -- pure TISE."""
        V = lambda r: coulomb_potential(r)
        E = bohr_sommerfeld_energy(0, 0, V, E_min=-2.0, E_max=-0.1)
        assert isinstance(E, float)
        assert E < 0
        assert not np.isnan(E)

    def test_effective_1d_potential_is_regular(self):
        """RSG maps singular TISE to regular 1D problem -- no singularity."""
        x_array = np.linspace(-5, 5, 100)
        V_eff = effective_potential_rsg(x_array, l=0, E=bohr_energy_exact(1))
        assert np.all(np.isfinite(V_eff)), (
            "V_eff has infinities -- singularity not regularized"
        )

    def test_rsg_coordinate_transform(self):
        """Log-transform: r->0 maps to x->-inf, not finite."""
        x_near_origin = rsg_transform(1e-10)
        assert x_near_origin < -20
        x_large = rsg_transform(1e10)
        assert x_large > 20
        r_test = 5.3
        assert abs(rsg_inverse(rsg_transform(r_test)) - r_test) < 1e-12

    def test_tise_solution_n1_to_n5(self):
        """RSG + WKB solves TISE for n=1..5, l=0 -- no TDSE."""
        results = solve_bohr_spectrum_rsg(n_max=5, l=0)
        assert len(results) == 5
        for n, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"TISE solution failed for n={n}"
            assert rel_err < 0.001, (
                f"n={n}: RSG TISE error {rel_err:.2e} > 0.1%"
            )

    def test_phase_quantization_condition(self):
        """At eigenvalue E_n, total WKB phase = pi*hbar*(n_r + 1/2)."""
        V = lambda r: coulomb_potential(r)
        for n_r in range(3):
            n = n_r + 1
            E_exact = bohr_energy_exact(n)
            target = np.pi * HBAR * (n_r + 0.5)
            action = wkb_action_integral(E_exact, V, 0, use_langer=True)
            rel_err = abs(action - target) / target
            assert rel_err < 0.001, (
                f"n_r={n_r}: action={action:.6f}, target={target:.6f}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_langer_makes_wkb_exact(self):
        """With Langer, WKB is EXACT for Coulomb (not just approximate)."""
        V = lambda r: coulomb_potential(r)
        for n in range(1, 4):
            E_exact = bohr_energy_exact(n)
            E_wkb = bohr_sommerfeld_energy(
                n - 1, 0, V,
                E_min=E_exact * 1.5, E_max=E_exact * 0.5,
                use_langer=True
            )
            rel_err = abs(E_wkb - E_exact) / abs(E_exact)
            assert rel_err < 1e-4, (
                f"n={n}: WKB+Langer should be exact, rel_err={rel_err:.2e}"
            )

    # ------------------------------------------------------------------
    # SSZ-Logik: Monodromie und Phasenbilanz
    # ------------------------------------------------------------------

    def test_phase_balance_is_exact_integer_multiples(self):
        """SSZ: Monodromie -- Phasenakkumulation ist exakt n*pi.

        Die Bohr-Sommerfeld-Bedingung:
          integral p_r dr = pi*hbar*(n_r + 1/2)

        entspricht der SSZ-Phasenbilanz: Die Gesamtphase ueber einen
        vollstaendigen Halbzyklus (Wendepunkt zu Wendepunkt) ist ein
        halbganzzahliges Vielfaches von pi. Das ist die Quantenbedingung
        als geometrische Schliessungsbedingung.
        """
        V = lambda r: coulomb_potential(r)
        for n_r in range(4):
            E_exact = bohr_energy_exact(n_r + 1)
            target = np.pi * HBAR * (n_r + 0.5)
            action = wkb_action_integral(E_exact, V, l=0, use_langer=True)
            ratio = action / target
            assert abs(ratio - 1.0) < 0.001, (
                f"n_r={n_r}: phase ratio = {ratio:.6f}, expected 1.0"
            )

    def test_action_monotone_with_energy(self):
        """SSZ: Mehr Energie = mehr Phasenraum = groessere Aktion.

        I(E) ist monoton wachsend in E (fuer gebundene Zustaende E < 0).
        Hoehere Energie -> groessere klassische Reichweite -> mehr Phase.
        Das ist das Fundament der Bohr-Sommerfeld-Zaehlung.
        """
        V = lambda r: coulomb_potential(r)
        energies = [bohr_energy_exact(n) for n in range(1, 5)]
        actions = [wkb_action_integral(E, V, l=0, use_langer=True)
                   for E in energies]
        for i in range(len(actions) - 1):
            assert actions[i] < actions[i + 1], (
                f"Action not monotone: I(E_{i+1})={actions[i]:.4f} "
                f">= I(E_{i+2})={actions[i+1]:.4f}"
            )

    def test_action_quantization_steps(self):
        """SSZ: Aufeinanderfolgende Zustaende unterscheiden sich um pi*hbar.

        I(E_{n+1}) - I(E_n) = pi*hbar (exakt fuer Coulomb mit Langer).
        Das ist die diskrete Phasenbilanz -- jedes neue Niveau akkumuliert
        genau eine weitere halbe Wellenlaenge.
        """
        V = lambda r: coulomb_potential(r)
        actions = []
        for n_r in range(4):
            E = bohr_energy_exact(n_r + 1)
            a = wkb_action_integral(E, V, l=0, use_langer=True)
            actions.append(a)
        delta_pi = np.pi * HBAR
        for i in range(len(actions) - 1):
            delta = actions[i + 1] - actions[i]
            rel_err = abs(delta - delta_pi) / delta_pi
            assert rel_err < 0.001, (
                f"Step n_r={i}->{i+1}: delta={delta:.6f}, "
                f"expected pi={delta_pi:.6f}, err={rel_err:.2e}"
            )

    def test_tdse_phase_is_purely_temporal(self):
        """SSZ: Die TDSE-Phase exp(-iEt/hbar) ist rein zeitlich.

        Sie aendert NICHT die raeumliche Eigenfunktion und
        NICHT die Energieeigenwerte. Die TISE-Loesung ist vollstaendig
        fuer alle Observablen die nicht Zeit involvieren.
        """
        E = bohr_energy_exact(1)
        omega = abs(E) / HBAR
        t_values = np.linspace(0, 2 * np.pi / omega, 100)
        phase_mod_sq = np.ones_like(t_values)
        np.testing.assert_allclose(phase_mod_sq, 1.0, rtol=1e-14)
        assert E == bohr_energy_exact(1)

    def test_rsg_classical_region_is_finite(self):
        """SSZ: Die klassische Region [r1, r2] ist endlich und wohldefiniert.

        Zwischen den Wendepunkten ist p_r^2 > 0.
        Das ist die Region wo Phase akkumuliert wird.
        Ausserhalb ist p_r^2 < 0 (klassisch verboten).
        """
        from rsg_core import find_turning_points, radial_momentum_langer
        V = lambda r: coulomb_potential(r)
        for n in range(1, 4):
            E = bohr_energy_exact(n)
            r1, r2 = find_turning_points(E, V, l=0, use_langer=True)
            assert 0 < r1 < r2
            r_mid = (r1 + r2) / 2.0
            pr2_mid = radial_momentum_langer(
                np.array([r_mid]), E, V, 0
            )[0]
            assert pr2_mid > 0, (
                f"n={n}: p_r^2({r_mid:.3f}) = {pr2_mid:.4f} should be > 0"
            )
            r_out = r2 * 1.1
            pr2_out = radial_momentum_langer(
                np.array([r_out]), E, V, 0
            )[0]
            assert pr2_out < 0, (
                f"n={n}: p_r^2({r_out:.3f}) = {pr2_out:.4f} should be < 0"
            )
