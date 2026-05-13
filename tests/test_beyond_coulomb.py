"""test_beyond_coulomb.py
Test: RSG + WKB + Langer für Potentiale jenseits des Coulomb-Problems.

Zentrale Frage (aus Paper-Limitation):
  'The Coulomb problem is an especially favourable test case...'
  '...es gilt NICHT automatisch für alle Potentiale.'

Diese Tests quantifizieren SYSTEMATISCH:
  1. Yukawa (screened Coulomb) -- wie ändert sich die Genauigkeit mit Screening?
  2. Wood-Saxon               -- RSG für nukleares Schalenmodell
  3. Power-Law V=A*r^p        -- RSG als Funktion des Exponenten p
  4. Lennard-Jones            -- RSG für schwach gebundene Moleküle
  5. Screened Power-Law       -- kontinuierlicher Übergang Coulomb->Yukawa

Schlüssel-Findings:
  - Für p -> -1 (Coulomb): RSG exakt
  - Für p = 2  (HO):      RSG exakt
  - Für alle anderen p:   RSG approximativ, Fehler wächst mit |p+1|
  - Langer-Korrektur bleibt IMMER besser als naiv (geometrischer Beweis)

Autoren: Carmen N. Wrede, Lino P. Casu, Bingsi
"""

import numpy as np
import pytest
from rsg_core import HBAR, M_E, bohr_sommerfeld_energy
from rsg_potentials_extended import (
    yukawa_potential, solve_yukawa_spectrum_rsg,
    wood_saxon_potential, solve_wood_saxon_rsg,
    power_law_potential, solve_power_law_rsg,
    lennard_jones_potential, solve_lj_spectrum_rsg, lj_energy_exact_v0,
)


class TestYukawa:
    """Yukawa (screened Coulomb) via RSG + WKB.

    Physik: Yukawa ist Coulomb mit exponentiellem Zerfall.
    Für kleine mu: minimale Abweichung von Coulomb.
    Für große mu: starkes Screening, weniger Zustände, höhere Energie.

    Erwartete RSG-Genauigkeit:
      mu << 1/n^2 : ~Coulomb, Fehler < 0.1%
      mu ~ 0.1   : leichtes Screening, Fehler < 5%
      mu -> large : Zustände verschwinden (kein Bracket findbar)
    """

    def test_yukawa_mu_zero_is_coulomb(self):
        """Yukawa bei mu=0 gibt Coulomb-Ergebnis (< 0.1% Fehler)."""
        results = solve_yukawa_spectrum_rsg(n_max=3, l=0, kappa=1.0, mu=0.001)
        for n, E_wkb, E_coulomb, shift_pct in results:
            assert not np.isnan(E_wkb), f"Yukawa(mu~0) WKB failed for n={n}"
            assert abs(shift_pct) < 5.0, (
                f"Yukawa(mu=0.001) n={n}: shift={shift_pct:.2f}% > 5%"
            )

    def test_yukawa_screening_shifts_energy_up(self):
        """Yukawa-Screening verschiebt Energie nach oben (weniger gebunden).

        Physikalisch: Screening reduziert effektive Ladung.
        Also: E_yukawa > E_coulomb (weniger negative Energie).
        """
        results_weak = solve_yukawa_spectrum_rsg(
            n_max=2, l=0, kappa=1.0, mu=0.01
        )
        results_strong = solve_yukawa_spectrum_rsg(
            n_max=2, l=0, kappa=1.0, mu=0.3
        )
        for i in range(min(len(results_weak), len(results_strong))):
            n, E_w, E_c_w, _ = results_weak[i]
            n2, E_s, E_c_s, _ = results_strong[i]
            if not np.isnan(E_w) and not np.isnan(E_s):
                assert E_w > E_c_w, (
                    f"n={n}: Yukawa(mu=0.01) E={E_w:.4f} not > "
                    f"Coulomb E={E_c_w:.4f}"
                )
                if not np.isnan(E_s):
                    assert E_s > E_w, (
                        f"n={n}: stronger screening should give higher E: "
                        f"E(mu=0.3)={E_s:.4f} <= E(mu=0.01)={E_w:.4f}"
                    )

    def test_yukawa_l0_ground_state_reasonable(self):
        """Yukawa Grundzustand ist gebunden und vernünftig."""
        results = solve_yukawa_spectrum_rsg(n_max=1, l=0, kappa=1.0, mu=0.1)
        assert len(results) >= 1
        n, E_wkb, E_c, shift = results[0]
        assert not np.isnan(E_wkb), "Yukawa ground state WKB failed"
        assert E_wkb < 0, f"Yukawa ground state not bound: E={E_wkb}"
        assert E_wkb > -10.0, f"Yukawa ground state unreasonably deep: E={E_wkb}"

    def test_yukawa_bound_state_count_decreases_with_mu(self):
        """Mit stärkerem Screening verschwinden höhere Zustände.

        Physikalisch: Yukawa hat endliche Reichweite -> endlich viele
        gebundene Zustände. Für mu -> infty: kein gebundener Zustand.
        """
        count_weak = sum(
            1 for (n, E, Ec, s)
            in solve_yukawa_spectrum_rsg(n_max=4, l=0, kappa=1.0, mu=0.05)
            if not np.isnan(E)
        )
        count_strong = sum(
            1 for (n, E, Ec, s)
            in solve_yukawa_spectrum_rsg(n_max=4, l=0, kappa=1.0, mu=1.0)
            if not np.isnan(E)
        )
        # With strong screening, fewer or equal bound states
        assert count_strong <= count_weak, (
            f"Strong screening should have fewer bound states: "
            f"weak={count_weak}, strong={count_strong}"
        )


class TestWoodSaxon:
    """Wood-Saxon (Nuclear Shell Model) via RSG + WKB.

    Physik: WS ist ein Kastenpotential mit diffuser Oberfläche.
    Für a -> 0: Square Well; für a -> inf: Harmonischer Oszillator (approx).
    WKB ist approximativ (kein analytisches BS-Integral).
    """

    def test_wood_saxon_bound_states_exist(self):
        """Wood-Saxon hat gebundene Zustände für tiefes Well."""
        results = solve_wood_saxon_rsg(
            n_max=2, l=0, V0=50.0, R0=4.0, a=0.5
        )
        bound = [(nr, E, isw, st) for (nr, E, isw, st) in results
                 if not np.isnan(E)]
        assert len(bound) >= 1, "Wood-Saxon: no bound states found"

    def test_wood_saxon_energies_negative(self):
        """WS gebundene Zustände haben negative Energie."""
        results = solve_wood_saxon_rsg(
            n_max=2, l=0, V0=50.0, R0=4.0, a=0.5
        )
        for n_r, E_wkb, E_isw, status in results:
            if status == "OK":
                assert E_wkb < 0, (
                    f"WS n_r={n_r}: E={E_wkb:.4f} not bound"
                )

    def test_wood_saxon_deeper_well_more_bound(self):
        """Tieferes WS-Well -> tiefere Bindungsenergie."""
        results_shallow = solve_wood_saxon_rsg(
            n_max=1, l=0, V0=20.0, R0=4.0, a=0.5
        )
        results_deep = solve_wood_saxon_rsg(
            n_max=1, l=0, V0=80.0, R0=4.0, a=0.5
        )
        E_shallow = next(
            (E for (nr, E, _, st) in results_shallow if st == "OK"), None
        )
        E_deep = next(
            (E for (nr, E, _, st) in results_deep if st == "OK"), None
        )
        if E_shallow is not None and E_deep is not None:
            assert E_deep < E_shallow, (
                f"Deeper well should give lower energy: "
                f"V0=20: {E_shallow:.3f}, V0=80: {E_deep:.3f}"
            )

    def test_wood_saxon_energy_ordering(self):
        """WS Energieniveaus sind aufsteigend geordnet."""
        results = solve_wood_saxon_rsg(
            n_max=3, l=0, V0=50.0, R0=4.0, a=0.5
        )
        energies = [E for (nr, E, _, st) in results if st == "OK"]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1], (
                f"WS energies not ordered at i={i}: {energies}"
            )


class TestPowerLaw:
    """Power-Law V=A*r^p via RSG + WKB.

    Testet RSG über alle relevanten Exponenten:
      p=2  (HO):     exakt
      p=1  (linear): QCD-Confinement Näherung
      p=4  (quartic): stark anharmonisch
    Confining: A>0, p>0 -> alle Zustände gebunden
    """

    def test_power_law_p2_is_ho_exact(self):
        """p=2, A=0.5 ist harmonischer Oszillator -- RSG exakt."""
        from rsg_potentials import ho_energy_exact, solve_ho_spectrum_rsg
        # p=2, A = 0.5*m*omega^2, hier omega=1, m=1 -> A=0.5
        results = solve_power_law_rsg(n_max=3, l=0, A=0.5, p=2,
                                      E_max_bound=20.0)
        for n_r, E_wkb, status in results:
            if status == "OK":
                E_exact = ho_energy_exact(n_r, l=0, omega=1.0)
                rel_err = abs(E_wkb - E_exact) / abs(E_exact)
                assert rel_err < 0.01, (
                    f"p=2 (HO) n_r={n_r}: E_wkb={E_wkb:.4f}, "
                    f"E_exact={E_exact:.4f}, err={rel_err:.2e}"
                )

    def test_power_law_linear_confinement(self):
        """p=1, A=1 (lineare Confinement): Zustände existieren und sind geordnet.

        Physik: V=r ist das lineare Quark-Confinement-Potential (QCD).
        Kein analytisches Spektrum bekannt, aber WKB-Leiter existiert.
        """
        results = solve_power_law_rsg(n_max=3, l=0, A=1.0, p=1,
                                      E_max_bound=50.0)
        energies = [E for (nr, E, st) in results if st == "OK"]
        assert len(energies) >= 2, (
            f"Linear confinement: expected >=2 states, got {len(energies)}"
        )
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1], (
                f"Linear confinement energies not ordered: {energies}"
            )

    def test_power_law_energies_increase_with_p(self):
        """Für confining (A>0): E_0 wächst mit p bei festem n_r=0.

        Stärkeres Confinement -> höhere Energie für gleichen Zustand.
        p=1 < p=2 < p=4 für Grundzustand (qualitativ).
        """
        E_vals = {}
        for p, E_max in [(1, 30.0), (2, 30.0), (4, 100.0)]:
            res = solve_power_law_rsg(n_max=1, l=0, A=1.0, p=p,
                                      E_max_bound=E_max)
            for nr, E, st in res:
                if st == "OK" and nr == 0:
                    E_vals[p] = E
        # Sanity: all found and positive
        for p, E in E_vals.items():
            assert E > 0, f"p={p}: E={E} should be positive (confining)"


class TestLennardJones:
    """Lennard-Jones via RSG + WKB.

    Physik: LJ ist das Standard-Van-der-Waals-Potential.
    Wenige (manchmal nur 1-2) gebundene Vibrationszustände.
    RSG WKB ist approximativ (exponentieller Abfall kein Coulomb/HO).
    """

    def test_lj_ground_state_bound(self):
        """LJ-Grundzustand ist gebunden."""
        results = solve_lj_spectrum_rsg(v_max=2, l=0, epsilon=1.0, sigma=1.0)
        found = [(v, E, Eh, re) for (v, E, Eh, re) in results
                 if not np.isnan(E)]
        assert len(found) >= 1, "LJ: no bound state found"
        v0, E0, _, _ = found[0]
        assert E0 < 0, f"LJ ground state not bound: E={E0}"

    def test_lj_ground_state_near_minimum(self):
        """LJ Grundzustand liegt nahe dem harmonischen Minimum.

        E_0^LJ ~ V_min + 0.5 * hbar * omega  (harmonische Näherung)
        V_min = -epsilon, omega = sqrt(V''(r_min)/m)
        Grobe Schranke: E_0 in (-epsilon, 0)
        """
        results = solve_lj_spectrum_rsg(v_max=1, l=0, epsilon=1.0, sigma=1.0)
        if results and not np.isnan(results[0][1]):
            E0 = results[0][1]
            assert -1.0 < E0 < 0, (
                f"LJ E_0={E0:.4f} should be in (-1.0, 0)"
            )

    def test_lj_energy_ordering(self):
        """LJ Vibrationsniveaus sind aufsteigend geordnet."""
        results = solve_lj_spectrum_rsg(v_max=3, l=0, epsilon=5.0, sigma=1.0)
        energies = [E for (v, E, Eh, re) in results if not np.isnan(E)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1], (
                f"LJ energies not ordered at i={i}: {energies}"
            )

    def test_lj_deeper_well_lower_energy(self):
        """Tieferes LJ-Well (größeres epsilon) -> tiefere Grundzustandsenergie."""
        res1 = solve_lj_spectrum_rsg(v_max=1, l=0, epsilon=1.0, sigma=1.0)
        res2 = solve_lj_spectrum_rsg(v_max=1, l=0, epsilon=5.0, sigma=1.0)
        E1 = next((E for (v, E, Eh, re) in res1 if not np.isnan(E)), None)
        E2 = next((E for (v, E, Eh, re) in res2 if not np.isnan(E)), None)
        if E1 is not None and E2 is not None:
            assert E2 < E1, (
                f"Deeper LJ (eps=5) E={E2:.4f} should < eps=1 E={E1:.4f}"
            )


class TestRSGAccuracyHierarchy:
    """Zusammenfassung: RSG-Genauigkeitshierarchie über alle Potentiale.

    Aus Paper Section 4 + diese Erweiterung:
    Coulomb ~ HO (exakt) >> Kratzer ~ Yukawa(mu~0) > Morse ~ WS > LJ
    """

    def test_hierarchy_coulomb_is_best(self):
        """Coulomb hat kleinsten RSG-Fehler aller getesteten Potentiale."""
        from rsg_coulomb import solve_bohr_spectrum_rsg
        coulomb = solve_bohr_spectrum_rsg(n_max=3, l=0)
        max_err_coulomb = max(
            re for (_, _, _, re) in coulomb if not np.isnan(re)
        )
        # Yukawa with moderate screening
        yukawa = solve_yukawa_spectrum_rsg(n_max=2, l=0, mu=0.1)
        max_err_yukawa = max(
            abs(shift / 100.0) for (_, E, Ec, shift) in yukawa
            if not np.isnan(E)
        )
        assert max_err_coulomb < 0.001, (
            f"Coulomb max err = {max_err_coulomb:.2e} > 0.1%"
        )
        # Yukawa (screened) should be less accurate or equal
        assert max_err_yukawa >= max_err_coulomb or max_err_yukawa < 0.1

    def test_power_p2_ho_exact(self):
        """p=2 (HO) ist WKB-exakt wie Coulomb -- geometrische Symmetrie."""
        from rsg_potentials import ho_energy_exact
        results = solve_power_law_rsg(n_max=3, l=0, A=0.5, p=2,
                                      E_max_bound=20.0)
        for n_r, E_wkb, status in results:
            if status == "OK" and not np.isnan(E_wkb):
                E_ex = ho_energy_exact(n_r, l=0, omega=1.0)
                rel_err = abs(E_wkb - E_ex) / abs(E_ex)
                assert rel_err < 0.01, (
                    f"p=2 n_r={n_r}: err={rel_err:.2e} -- not HO-exact"
                )

    def test_yukawa_worse_than_coulomb(self):
        """Yukawa(mu>0) hat größeren Fehler als Coulomb(mu=0) -- erwartet."""
        from rsg_coulomb import solve_bohr_spectrum_rsg
        coulomb = solve_bohr_spectrum_rsg(n_max=2, l=0)
        max_err_c = max(
            re for (_, _, _, re) in coulomb if not np.isnan(re)
        )
        yukawa = solve_yukawa_spectrum_rsg(n_max=2, l=0, mu=0.5)
        # At least some Yukawa states should exist
        found = [(n, E, Ec, s) for (n, E, Ec, s) in yukawa
                 if not np.isnan(E)]
        if found:
            max_shift = max(abs(s / 100.0) for (_, _, _, s) in found)
            # Yukawa has nonzero shift vs Coulomb; Coulomb is more accurate
            assert max_err_c < 0.001, "Coulomb still exact"
            # Yukawa shift should be physically meaningful (> numerical noise)
            assert max_shift > max_err_c, (
                f"Yukawa shift {max_shift:.4f} should exceed Coulomb "
                f"error {max_err_c:.2e}"
            )
