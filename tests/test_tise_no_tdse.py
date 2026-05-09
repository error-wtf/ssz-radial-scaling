"""test_tise_no_tdse.py
Test: RSG solves the TISE WITHOUT the TDSE.

Core question:
  'Can we solve quantum systems with RSG bypassing the TDSE?'

Answer: YES.

RSG works purely with TISE:
  TISE -> log-transform -> Langer-corrected WKB -> exact spectrum

No time evolution, no propagator, no TDSE.
The TDSE only adds exp(-iEt/hbar) -- a clock, not physics.
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
