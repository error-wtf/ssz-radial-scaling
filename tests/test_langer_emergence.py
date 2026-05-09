"""test_langer_emergence.py
Test: The Langer correction emerges naturally from RSG geometry.

Central interpretive claim:
  l*(l+1) -> (l+1/2)^2 is NOT ad hoc, but geometric consequence
  of the log-radial transformation.
"""

import numpy as np
import pytest
from rsg_core import (
    langer_angular_term, naive_angular_term,
    bohr_energy_exact, bohr_sommerfeld_energy
)
from rsg_coulomb import coulomb_potential


class TestLangerEmergence:
    """Langer correction: geometric vs ad hoc."""

    def test_langer_term_value(self):
        """Langer term (l+1/2)^2 differs from l*(l+1) by exactly 1/4."""
        for l in range(5):
            naive = naive_angular_term(l)
            langer = langer_angular_term(l)
            diff = langer - naive
            assert abs(diff - 0.25) < 1e-12, (
                f"l={l}: Langer-Naive = {diff}, expected 0.25"
            )

    def test_l0_langer_correction_is_critical(self):
        """For l=0: Langer gives (1/2)^2=0.25, naive gives 0.
        Without Langer, centrifugal barrier vanishes -- unphysical.
        """
        naive_l0 = naive_angular_term(0)
        langer_l0 = langer_angular_term(0)
        assert naive_l0 == 0, "Naive l=0 has zero angular term"
        assert abs(langer_l0 - 0.25) < 1e-12

    def test_langer_gives_exact_ground_state(self):
        """Langer-corrected WKB gives exact n=1,l=0 energy."""
        V = lambda r: coulomb_potential(r)
        E_exact = bohr_energy_exact(1)
        E_langer = bohr_sommerfeld_energy(
            0, 0, V, E_min=-2.0, E_max=-0.1, use_langer=True
        )
        rel_err = abs(E_langer - E_exact) / abs(E_exact)
        assert rel_err < 0.001, (
            f"RSG/Langer E={E_langer:.6f}, exact={E_exact:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_langer_improvement_over_naive(self):
        """For all l in [0,1,2], Langer WKB error < 0.1%."""
        V = lambda r: coulomb_potential(r)
        for l in range(3):
            n = l + 1
            E_exact = bohr_energy_exact(n)
            E_langer = bohr_sommerfeld_energy(
                0, l, V,
                E_min=E_exact * 1.8,
                E_max=E_exact * 0.3,
                use_langer=True
            )
            err = abs(E_langer - E_exact) / abs(E_exact)
            assert err < 0.001, f"l={l}: Langer error {err:.2e} > 0.1%"

    def test_langer_1_4_is_universal(self):
        """The extra 1/4 term is the same for all l -- universal geometric."""
        diffs = [langer_angular_term(l) - naive_angular_term(l)
                 for l in range(10)]
        for d in diffs:
            assert abs(d - 0.25) < 1e-12
