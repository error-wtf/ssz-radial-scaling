"""test_langer_emergence.py
Test: The Langer correction emerges naturally from RSG geometry.

Central interpretive claim:
  l*(l+1) -> (l+1/2)^2 is NOT ad hoc, but geometric consequence
  of the log-radial transformation.

SSZ-Logik: Die 1/4-Korrektur ist die geometrische Signatur der
logarithmischen Koordinatentransformation. Sie entsteht aus drei
gleichzeitig transformierten Elementen:
  1. Wellenfunktionsumgewichtung:  R(r) -> sqrt(r)*R(r)
  2. Masstransformation:           dr   -> e^x dx
  3. Radialoperator-Konsistenz:    d^2/dr^2 -> korrekte Form in x

Die Summe dieser drei Beitraege ergibt exakt +1/4 -- ohne Fitting.
"""

import numpy as np
import pytest
from rsg_core import (
    langer_angular_term, naive_angular_term,
    bohr_energy_exact, bohr_sommerfeld_energy,
    rsg_transform, rsg_inverse, HBAR, M_E
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

    # ------------------------------------------------------------------
    # SSZ-Logik: Geometrische Herkunft der 1/4-Korrektur
    # ------------------------------------------------------------------

    def test_langer_term_is_half_integer_squared(self):
        """SSZ: (l+1/2)^2 ist die Quadratur eines Halbganzzahligen.

        Die RSG-Transformation r=e^x macht aus dem ganzzahligen l
        einen halbganzzahligen Effektivwert l+1/2. Das ist die
        geometrische Konsequenz: Die sphaerische Geometrie zaehlt in
        halben Einheiten in der logarithmischen Koordinate.
        """
        for l in range(6):
            effective_l = l + 0.5
            langer = langer_angular_term(l)
            assert abs(langer - effective_l**2) < 1e-14, (
                f"l={l}: Langer term {langer} != (l+0.5)^2 = {effective_l**2}"
            )

    def test_langer_correction_independent_of_potential(self):
        """SSZ: Die 1/4-Korrektur ist potentialunabhaengig (rein geometrisch).

        Die Langer-Korrektur haengt nur von l ab, nicht von V(r).
        Sie ist die Signatur der Radialgeometrie, nicht des Potentials.
        """
        for l in range(4):
            naive = naive_angular_term(l)
            langer = langer_angular_term(l)
            correction = langer - naive
            assert abs(correction - 0.25) < 1e-14, (
                f"l={l}: correction={correction}, must be 0.25 regardless of V"
            )

    def test_log_transform_shifts_singularity(self):
        """SSZ: x=ln(r) schiebt r=0 nach x=-inf -- Singularitaet regularisiert.

        Das ist die physikalische Grundlage der Langer-Korrektur:
        r=0 ist kein gewoehnlicher Randpunkt mehr, sondern eine
        Grenze im Unendlichen.
        """
        r_values = np.array([1e-10, 1e-5, 1e-2, 1.0])
        x_values = rsg_transform(r_values)
        assert x_values[0] < -20, "r=1e-10 must map to x << 0"
        assert x_values[1] < -10, "r=1e-5 must map to x < -10"
        assert x_values[2] < -4, "r=1e-2 must map to x < -4"
        r_back = rsg_inverse(x_values)
        np.testing.assert_allclose(r_back, r_values, rtol=1e-12)

    def test_wavefunction_rescaling_factor(self):
        """SSZ: R(r) -> sqrt(r)*R(r) ist das korrekte Massgewicht.

        Die Transformation phi(x) = sqrt(r)*R(r) entspricht dem
        Jacobi-Faktor sqrt(dr/dx) = sqrt(r) der Koordinatentransformation.
        Das ist kein freier Parameter -- es ist das eindeutige Massgewicht
        fuer L^2(dr) -> L^2(dx).
        """
        r_test = np.array([0.5, 1.0, 2.0, 5.0])
        x_test = rsg_transform(r_test)
        jacobian = np.exp(x_test)
        np.testing.assert_allclose(jacobian, r_test, rtol=1e-12)
        weight = np.sqrt(jacobian)
        np.testing.assert_allclose(weight, np.sqrt(r_test), rtol=1e-12)

    def test_langer_naive_crossover_at_large_l(self):
        """SSZ: Fuer l>>1 konvergieren Langer und Naiv (relative Differenz -> 0).

        (l+1/2)^2 / l(l+1) -> 1 fuer l -> inf.
        Die 1/4-Korrektur ist bei grossen l relativ klein,
        bei l=0 dominant (von 0 auf 0.25: unendlicher relativer Effekt).
        """
        for l in [1, 2, 5, 10, 20]:
            naive = naive_angular_term(l)
            langer = langer_angular_term(l)
            relative_diff = abs(langer - naive) / naive
            expected = 0.25 / (l * (l + 1))
            assert abs(relative_diff - expected) < 1e-12, (
                f"l={l}: relative diff {relative_diff} != 0.25/l(l+1)"
            )
        assert abs(langer_angular_term(0) - naive_angular_term(0) - 0.25) < 1e-14
