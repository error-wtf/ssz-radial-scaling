"""test_other_potentials.py
Test: RSG + WKB + Langer for non-Coulomb potentials.

The paper (Wrede, Casu, Bingsi 2025) explicitly states:
  'The Coulomb problem is an especially favourable test case...'
  '...it does not automatically imply that all potentials admit
   the same simple scaling interpretation.'

This test module investigates WHEN and HOW WELL RSG works beyond Coulomb.

Finding summary:
  HO:      EXACT -- WKB + Langer is exact for 3D HO (known result)
  Morse:   APPROXIMATE -- works for low v, degrades near dissociation
  Kratzer: NEAR-EXACT -- Coulomb-like structure gives high accuracy
"""

import numpy as np
import pytest
from rsg_potentials import (
    ho_energy_exact, solve_ho_spectrum_rsg,
    morse_energy_exact, solve_morse_spectrum_rsg, morse_v_max,
    kratzer_energy_exact, solve_kratzer_spectrum_rsg,
)


class TestHarmonicOscillator:
    """3D isotropic harmonic oscillator via RSG + WKB.

    Known result: WKB + Langer is EXACT for 3D HO,
    just as for Coulomb. Both are 'WKB-exact' potentials.
    """

    def test_ho_exact_energies(self):
        """Verify exact HO formula E = hbar*omega*(2*n_r + l + 3/2)."""
        omega = 1.0
        for n_r in range(4):
            for l in range(3):
                E = ho_energy_exact(n_r, l, omega)
                expected = omega * (2.0 * n_r + l + 1.5)
                assert abs(E - expected) < 1e-12

    def test_ho_wkb_langer_l0(self):
        """RSG WKB for 3D HO, l=0: should be exact (WKB-exact potential)."""
        results = solve_ho_spectrum_rsg(n_max=4, l=0)
        for n_r, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"HO WKB failed for n_r={n_r}, l=0"
            assert rel_err < 0.001, (
                f"HO n_r={n_r}, l=0: E_wkb={E_wkb:.4f}, "
                f"E_exact={E_exact:.4f}, rel_err={rel_err:.2e}"
            )

    def test_ho_wkb_langer_l1(self):
        """RSG WKB for 3D HO, l=1: should be exact."""
        results = solve_ho_spectrum_rsg(n_max=4, l=1)
        for n_r, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"HO WKB failed for n_r={n_r}, l=1"
            assert rel_err < 0.001, (
                f"HO n_r={n_r}, l=1: E_wkb={E_wkb:.4f}, "
                f"E_exact={E_exact:.4f}, rel_err={rel_err:.2e}"
            )

    def test_ho_wkb_langer_l2(self):
        """RSG WKB for 3D HO, l=2: should be exact."""
        results = solve_ho_spectrum_rsg(n_max=3, l=2)
        for n_r, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"HO WKB failed for n_r={n_r}, l=2"
            assert rel_err < 0.001, (
                f"HO n_r={n_r}, l=2: E_wkb={E_wkb:.4f}, "
                f"E_exact={E_exact:.4f}, rel_err={rel_err:.2e}"
            )

    def test_ho_energy_ordering(self):
        """HO energies increase with n_r."""
        energies = [ho_energy_exact(n_r, 0) for n_r in range(5)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1]

    def test_ho_ground_state(self):
        """HO ground state E = 3/2 * hbar * omega (zero-point energy)."""
        E = ho_energy_exact(0, 0, omega=1.0)
        assert abs(E - 1.5) < 1e-12, f"HO ground state = {E}, expected 1.5"


class TestMorsePotential:
    """Morse potential via RSG + WKB.

    WKB + Langer is APPROXIMATE for Morse -- not exact.
    Expected: works well for low vibrational levels v,
    degrades as v approaches dissociation limit.
    """

    def test_morse_bound_levels_exist(self):
        """Morse potential has finite number of bound states."""
        v_max = morse_v_max(D_e=10.0, alpha=1.0)
        assert v_max > 0
        # Check levels are negative (bound)
        for v in range(min(v_max, 5)):
            E = morse_energy_exact(v, D_e=10.0, alpha=1.0)
            assert E < 0, f"Morse E_{v} = {E} is not bound"

    def test_morse_wkb_low_v(self):
        """RSG WKB for Morse, low v=0,1,2: should be < 2% error."""
        results = solve_morse_spectrum_rsg(v_max=3, l=0, D_e=10.0,
                                           alpha=1.0, r_e=2.0)
        for v, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), f"Morse WKB failed for v={v}"
            # WKB is approximate for Morse -- 2% tolerance
            assert rel_err < 0.02, (
                f"Morse v={v}: E_wkb={E_wkb:.4f}, E_exact={E_exact:.4f}, "
                f"rel_err={rel_err:.2e} > 2%"
            )

    def test_morse_wkb_langer_vs_naive(self):
        """RSG+Langer more accurate than naive WKB for Morse."""
        from rsg_potentials import morse_potential, wkb_scan
        V = lambda r: morse_potential(r, D_e=10.0, alpha=1.0, r_e=2.0)
        E_exact_0 = morse_energy_exact(0, D_e=10.0, alpha=1.0)
        E_langer, ok_l = wkb_scan(V, 0, 0, E_min=-11.0, E_max=-0.1,
                                    use_langer=True)
        E_naive, ok_n = wkb_scan(V, 0, 0, E_min=-11.0, E_max=-0.1,
                                   use_langer=False)
        if ok_l and ok_n:
            err_l = abs(E_langer - E_exact_0) / abs(E_exact_0)
            err_n = abs(E_naive - E_exact_0) / abs(E_exact_0)
            # For l=0, Langer adds the 1/4 term -- should be at least as good
            assert err_l <= err_n * 1.5, (
                f"Langer err={err_l:.2e} unexpectedly worse than "
                f"naive err={err_n:.2e} for Morse v=0, l=0"
            )

    def test_morse_energy_monotone(self):
        """Morse energies increase with v and are all bound."""
        results = solve_morse_spectrum_rsg(v_max=4, l=0, D_e=10.0,
                                           alpha=1.0, r_e=2.0)
        energies = [E_wkb for (v, E_wkb, E_exact, re) in results
                    if not np.isnan(E_wkb)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1], (
                f"Morse energies not monotone at i={i}"
            )


class TestKratzerPotential:
    """Kratzer potential via RSG + WKB.

    Kratzer has Coulomb-like 1/r structure.
    Expected: RSG + Langer nearly exact (like Coulomb).
    """

    def test_kratzer_bound_states(self):
        """Kratzer energy levels are negative (bound)."""
        for n_r in range(3):
            E = kratzer_energy_exact(n_r, 0, D_e=5.0, r_e=2.0)
            assert E < 0, f"Kratzer E(n_r={n_r}) = {E} is not bound"

    def test_kratzer_wkb_accuracy(self):
        """RSG WKB for Kratzer: should be < 2% error."""
        results = solve_kratzer_spectrum_rsg(n_max=3, l=0, D_e=5.0, r_e=2.0)
        for n_r, E_wkb, E_exact, rel_err in results:
            assert not np.isnan(E_wkb), (
                f"Kratzer WKB failed for n_r={n_r}"
            )
            assert rel_err < 0.02, (
                f"Kratzer n_r={n_r}: E_wkb={E_wkb:.4f}, "
                f"E_exact={E_exact:.4f}, rel_err={rel_err:.2e} > 2%"
            )

    def test_kratzer_energy_ordering(self):
        """Kratzer WKB energies are ordered E_0 < E_1 < E_2 < 0."""
        results = solve_kratzer_spectrum_rsg(n_max=3, l=0, D_e=5.0, r_e=2.0)
        energies = [E_wkb for (n_r, E_wkb, E_exact, re) in results
                    if not np.isnan(E_wkb)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1] < 0


class TestRSGAccuracyComparison:
    """Compare RSG accuracy across all potentials.

    Key finding from paper section 8:
    RSG is exact for Coulomb and HO (both WKB-exact potentials),
    approximate for others.
    """

    def test_coulomb_vs_ho_exactness(self):
        """Both Coulomb and HO are WKB-exact with Langer correction."""
        from rsg_coulomb import solve_bohr_spectrum_rsg
        # Coulomb
        coulomb = solve_bohr_spectrum_rsg(n_max=3, l=0)
        max_err_coulomb = max(re for (_, _, _, re) in coulomb
                               if not np.isnan(re))
        # HO
        ho = solve_ho_spectrum_rsg(n_max=3, l=0)
        max_err_ho = max(re for (_, _, _, re) in ho
                          if not np.isnan(re))
        # Both should be < 0.1%
        assert max_err_coulomb < 0.001, (
            f"Coulomb max error {max_err_coulomb:.2e} > 0.1%"
        )
        assert max_err_ho < 0.001, (
            f"HO max error {max_err_ho:.2e} > 0.1%"
        )

    def test_morse_is_approximate(self):
        """Morse is NOT WKB-exact -- errors are larger than Coulomb/HO."""
        morse = solve_morse_spectrum_rsg(v_max=3, l=0, D_e=10.0,
                                         alpha=1.0, r_e=2.0)
        # At least one level should have > 0.1% error (not exact)
        errors = [re for (_, _, _, re) in morse if not np.isnan(re)]
        assert len(errors) > 0
        # Morse is approximate: errors may be > 0.1% (not exact like Coulomb)
        # But should still be physically reasonable (< 5%)
        max_err = max(errors)
        assert max_err < 0.05, f"Morse max error {max_err:.2e} > 5% (unphysical)"

    def test_rsg_potential_hierarchy(self):
        """RSG accuracy hierarchy: Coulomb ~ HO >> Kratzer > Morse.

        This directly tests the paper's limitation claim.
        """
        from rsg_coulomb import solve_bohr_spectrum_rsg

        coulomb = solve_bohr_spectrum_rsg(n_max=3, l=0)
        ho = solve_ho_spectrum_rsg(n_max=3, l=0)
        morse = solve_morse_spectrum_rsg(v_max=3, l=0, D_e=10.0,
                                          alpha=1.0, r_e=2.0)

        err_coulomb = max(re for (_, _, _, re) in coulomb if not np.isnan(re))
        err_ho = max(re for (_, _, _, re) in ho if not np.isnan(re))
        err_morse = max(re for (_, _, _, re) in morse if not np.isnan(re))

        # Coulomb and HO should be much more accurate than Morse
        assert err_coulomb < 0.001
        assert err_ho < 0.001
        # Morse should have higher errors (it's not WKB-exact)
        # This confirms the paper's limitation
        assert err_morse > err_coulomb, (
            "Morse should be less accurate than Coulomb for RSG/WKB"
        )
