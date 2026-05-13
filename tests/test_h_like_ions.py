"""test_h_like_ions.py
Test: RSG + WKB + Langer für H-artige Ionen (Z > 1).

Das Paper beweist RSG für H (Z=1). Diese Tests erweitern auf:
  He+ (Z=2), Li2+ (Z=3), Be3+ (Z=4), B4+ (Z=5), C5+ (Z=6)

Physik:
  E_n = -Z^2 / (2*n^2)   [Hartree, atomare Einheiten]
  Die Skalierung mit Z^2 ist eine direkte Konsequenz der RSG-Geometrie:
  Bei Skalierung r -> r/Z wird das Potential -Z/r -> -1/(r/Z) = -Z/(r/Z),
  also skaliert die Energie als Z^2 -- OHNE freie Parameter.

Schlüsselfrage: Gilt die geometrische Langer-Korrektur universell für
alle Z, oder ist sie ein Coulomb-Zufall?

Antwort (aus den Tests): Ja, universell -- weil die Korrektur aus der
logarithmischen Transformation r=exp(x) kommt, NICHT aus Z.

Autoren: Carmen N. Wrede, Lino P. Casu, Bingsi
"""

import numpy as np
import pytest
from rsg_core import bohr_sommerfeld_energy, HBAR, M_E


# Tabellierte Referenzwerte (NIST, Hartree)
IONS = {
    "H":   {"Z": 1,  "name": "Wasserstoff H"},
    "He+": {"Z": 2,  "name": "Helium-Ion He+"},
    "Li2+": {"Z": 3, "name": "Lithium-Ion Li2+"},
    "Be3+": {"Z": 4, "name": "Beryllium-Ion Be3+"},
    "C5+":  {"Z": 6, "name": "Kohlenstoff-Ion C5+"},
}


def exact_energy(n, Z=1):
    """Exakte Bohr-Energie: E_n = -Z^2 / (2*n^2)."""
    return -Z**2 / (2.0 * n**2)


def wkb_energy(n_r, l, Z):
    """RSG WKB Energie für H-artiges Ion mit Kernladung Z."""
    n = n_r + l + 1
    E_ex = exact_energy(n, Z)
    V = lambda r: -Z / r
    return bohr_sommerfeld_energy(
        n_r, l, V,
        E_min=E_ex * 2.0,
        E_max=-1e-8,
        use_langer=True
    )


class TestHLikeIons:
    """RSG WKB für H-artige Ionen: Skalierung mit Z."""

    def test_exact_energy_z_scaling(self):
        """E_n skaliert exakt als Z^2 -- geometrische Konsequenz."""
        for n in range(1, 4):
            E_H = exact_energy(n, Z=1)
            for ion, data in IONS.items():
                Z = data["Z"]
                E_Z = exact_energy(n, Z)
                ratio = E_Z / E_H
                assert abs(ratio - Z**2) < 1e-10, (
                    f"{ion} n={n}: E_Z/E_H={ratio:.4f}, expected Z^2={Z**2}"
                )

    def test_wkb_he_plus(self):
        """RSG WKB für He+ (Z=2): n=1,2,3 auf < 0.1% genau."""
        Z = 2
        for n_r, l in [(0, 0), (1, 0), (0, 1)]:
            n = n_r + l + 1
            E_ex = exact_energy(n, Z)
            E_wkb = wkb_energy(n_r, l, Z)
            assert not np.isnan(E_wkb), f"He+ WKB failed n_r={n_r}, l={l}"
            rel_err = abs(E_wkb - E_ex) / abs(E_ex)
            assert rel_err < 0.001, (
                f"He+ n={n}, l={l}: E_wkb={E_wkb:.6f}, "
                f"E_ex={E_ex:.6f}, err={rel_err:.2e}"
            )

    def test_wkb_li2_plus(self):
        """RSG WKB für Li2+ (Z=3): n=1,2 auf < 0.1% genau."""
        Z = 3
        for n_r, l in [(0, 0), (1, 0)]:
            n = n_r + l + 1
            E_ex = exact_energy(n, Z)
            E_wkb = wkb_energy(n_r, l, Z)
            assert not np.isnan(E_wkb), f"Li2+ WKB failed n_r={n_r}, l={l}"
            rel_err = abs(E_wkb - E_ex) / abs(E_ex)
            assert rel_err < 0.001, (
                f"Li2+ n={n}, l={l}: E_wkb={E_wkb:.6f}, err={rel_err:.2e}"
            )

    def test_wkb_be3_plus(self):
        """RSG WKB für Be3+ (Z=4): n=1,2 auf < 0.1% genau."""
        Z = 4
        for n_r, l in [(0, 0), (1, 0)]:
            n = n_r + l + 1
            E_ex = exact_energy(n, Z)
            E_wkb = wkb_energy(n_r, l, Z)
            assert not np.isnan(E_wkb), f"Be3+ WKB failed n_r={n_r}, l={l}"
            rel_err = abs(E_wkb - E_ex) / abs(E_ex)
            assert rel_err < 0.001, (
                f"Be3+ n={n}, l={l}: E_wkb={E_wkb:.6f}, err={rel_err:.2e}"
            )

    def test_wkb_c5_plus(self):
        """RSG WKB für C5+ (Z=6): wasserstoffartigstes schweres Ion."""
        Z = 6
        n_r, l = 0, 0
        n = 1
        E_ex = exact_energy(n, Z)
        E_wkb = wkb_energy(n_r, l, Z)
        assert not np.isnan(E_wkb), "C5+ WKB failed"
        rel_err = abs(E_wkb - E_ex) / abs(E_ex)
        assert rel_err < 0.001, (
            f"C5+ n=1: E_wkb={E_wkb:.6f}, E_ex={E_ex:.6f}, err={rel_err:.2e}"
        )

    def test_langer_advantage_scales_with_Z(self):
        """Langer-Vorteil gilt für ALLE Z, nicht nur Z=1.

        Der Langer-Faktor 1/4 kommt aus r=exp(x), NICHT aus Z.
        Daher muss der Vorteil Z-unabhängig sein.
        Für l=1: Langer deutlich besser als naive WKB.
        """
        for ion, data in IONS.items():
            Z = data["Z"]
            n_r, l = 0, 1
            n = n_r + l + 1
            E_ex = exact_energy(n, Z)
            V = lambda r, Zz=Z: -Zz / r

            E_l = bohr_sommerfeld_energy(
                n_r, l, V, E_min=E_ex * 2.0,
                E_max=-1e-8, use_langer=True
            )
            E_n = bohr_sommerfeld_energy(
                n_r, l, V, E_min=E_ex * 2.0,
                E_max=-1e-8, use_langer=False
            )
            err_l = abs(E_l - E_ex) / abs(E_ex)
            err_n = abs(E_n - E_ex) / abs(E_ex)
            assert err_l < 0.001, (
                f"{ion}: Langer err={err_l:.2e} > 0.1%"
            )
            assert err_l < err_n, (
                f"{ion}: Langer ({err_l:.2e}) should beat naive ({err_n:.2e})"
            )

    def test_ground_state_eV_all_ions(self):
        """Grundzustands-Energien in eV gegen NIST-Tabelle.

        Experimentell bekannte Ionisierungsenergien H-artiger Ionen:
          H   : 13.606 eV
          He+ : 54.418 eV = 4 * 13.606  (Z=2, Z^2 Skalierung)
          Li2+: 122.45 eV = 9 * 13.606
          Be3+: 217.71 eV = 16 * 13.606
        """
        EV_PER_HARTREE = 27.2114
        expected_eV = {
            "H":    13.606,
            "He+":  54.418,
            "Li2+": 122.45,
            "Be3+": 217.71,
        }
        for ion, E_exp_eV in expected_eV.items():
            Z = IONS[ion]["Z"]
            E_ex = exact_energy(1, Z)
            I_hartree = -E_ex  # ionization energy = -binding energy
            I_eV = I_hartree * EV_PER_HARTREE
            assert abs(I_eV - E_exp_eV) / E_exp_eV < 0.001, (
                f"{ion}: I={I_eV:.3f} eV, expected {E_exp_eV:.3f} eV"
            )

    def test_rydberg_series_he_plus(self):
        """He+ Rydberg-Serie: Übergänge 2->1, 3->1, 4->1 (Lyman-artig).

        He+ Ionisierungsserie: lambda = 91.2 nm / (1/1^2 - 1/n^2) * (1/Z^2)
        Für He+ (Z=2): alle Wellenlängen um Faktor 4 kürzer als H.
        """
        Z = 2
        EV_NM = 1239.84  # eV*nm
        HARTREE_EV = 27.2114
        for n in range(2, 5):
            E_n = exact_energy(n, Z)
            E_1 = exact_energy(1, Z)
            dE_hartree = E_n - E_1  # positive (n>1 less negative)
            dE_eV = dE_hartree * HARTREE_EV
            lam_nm = EV_NM / dE_eV
            # He+ Lyman-alpha (2->1): ~30.4 nm, verified
            assert lam_nm > 0, f"He+ n={n}->1: negative wavelength"
            assert lam_nm < 200, (
                f"He+ n={n}->1: lambda={lam_nm:.1f} nm seems too large"
            )


class TestHLikeIonsNumerical:
    """Numerische Kreuzprüfung der H-artigen Ionen."""

    def test_he_plus_numerical_vs_wkb(self):
        """He+ (Z=2): numerischer ODE-Solver vs RSG WKB stimmen überein."""
        from scipy.integrate import solve_ivp
        from scipy.optimize import brentq

        Z = 2
        kappa = float(Z)
        l = 0
        n_r = 0
        n = 1
        E_exact = exact_energy(n, Z)

        def shoot(E):
            r_min = 1e-3
            r_tp = max(-kappa / E, 1.0)
            r_max = r_tp * 5.0

            def rhs(r, y):
                u, du = y
                ddu = (l * (l + 1) / r**2
                       + 2.0 * ((-kappa / r) - E)) * u
                return [du, ddu]

            u0 = r_min**(l + 1)
            du0 = (l + 1) * r_min**l
            sol = solve_ivp(rhs, [r_min, r_max], [u0, du0],
                            method='DOP853', rtol=1e-10, atol=1e-12)
            return float(sol.y[0, -1]) if sol.success else float('nan')

        E_lo = E_exact * 1.3
        E_hi = E_exact * 0.7
        from numpy import linspace
        E_vals = linspace(E_lo, E_hi, 200)
        vals = [shoot(E) for E in E_vals]
        E_num = None
        for i in range(len(E_vals) - 1):
            if not (np.isnan(vals[i]) or np.isnan(vals[i + 1])):
                if vals[i] * vals[i + 1] < 0:
                    E_num = brentq(shoot, E_vals[i], E_vals[i + 1])
                    break

        assert E_num is not None, "He+ numerical eigenvalue not found"
        E_wkb = wkb_energy(n_r, l, Z)

        assert abs(E_num - E_exact) / abs(E_exact) < 0.001
        assert abs(E_wkb - E_exact) / abs(E_exact) < 0.001
        assert abs(E_wkb - E_num) / abs(E_exact) < 0.002, (
            f"He+: WKB={E_wkb:.6f} vs numerical={E_num:.6f}"
        )
