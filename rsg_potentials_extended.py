"""rsg_potentials_extended.py
RSG extension: potentials beyond Coulomb and HO.

Tests whether RSG + WKB + Langer framework works for:
  5. Yukawa (screened Coulomb)     -- tests screening decay
  6. Wood-Saxon                    -- nuclear shell model
  7. Power-Law  V = A * r^p        -- tests exponent sensitivity
  8. Lennard-Jones                 -- molecular van-der-Waals
  9. Inverse-square  V = -A/r^2   -- centrifugal-like, RSG borderline
 10. Linear confinement V = k*r   -- quark model, QCD-inspired

Key question: Does the Langer correction remain beneficial,
and when does WKB break down?

Physical context:
  - Yukawa: screened nuclear or Debye-screened Coulomb in plasma
  - Wood-Saxon: nuclear mean-field (shell structure, magic numbers)
  - Power-Law: interpolates between HO (p=2) and Coulomb (p=-1)
  - Lennard-Jones: van-der-Waals molecules, weakly-bound states
  - Linear: heavy quark confinement (Cornell potential component)

Authors: Carmen N. Wrede, Lino P. Casu, Bingsi
"""

import numpy as np
from rsg_core import bohr_sommerfeld_energy, HBAR, M_E


# ---------------------------------------------------------------------------
# 5. Yukawa / Screened Coulomb  V(r) = -(kappa/r) * exp(-mu*r)
# ---------------------------------------------------------------------------

def yukawa_potential(r, kappa=1.0, mu=0.1):
    """Yukawa (screened Coulomb): V(r) = -(kappa/r) * exp(-mu*r).

    Parameters:
      kappa = coupling strength (= e^2 for Coulomb at mu=0)
      mu    = screening length inverse (Debye length = 1/mu)

    Limits:
      mu -> 0 : pure Coulomb
      mu >> 0 : exponentially screened, fewer bound states
    """
    return -(kappa / r) * np.exp(-mu * r)


def yukawa_energy_approx(n, kappa=1.0, mu=0.1):
    """First-order perturbative energy for Yukawa.

    E_n^Yukawa ~ E_n^Coulomb * (1 - mu * a_0 * n^2 / Z)

    In atomic units (a_0=1, Z=1):
      E_n^Yukawa ~ -1/(2n^2) - mu * <r>_n + O(mu^2)

    where <r>_n = (3n^2 - l(l+1)) / 2 * a_0  (for l=0)
    For l=0: <r>_n ~ 3n^2/2

    First-order shift: delta_E = <V_screening> = kappa*mu * <exp(-mu*r)/1>
    For small mu: delta_E ~ -kappa*mu * n^2
    """
    E_coulomb = -kappa**2 / (2.0 * n**2)
    # First-order shift (perturbative, small mu)
    r_mean = 1.5 * n**2  # <r>_{n,l=0} in atomic units
    delta_E = -kappa * mu * r_mean * np.exp(-mu * r_mean)
    return E_coulomb + delta_E


def solve_yukawa_spectrum_rsg(n_max=3, l=0, kappa=1.0, mu=0.1,
                              hbar=HBAR, m=M_E):
    """Solve Yukawa bound states using RSG + WKB + Langer.

    Returns list of (n, E_wkb, E_coulomb_ref, screening_shift_pct).
    """
    V = lambda r: yukawa_potential(r, kappa, mu)
    results = []
    for n_r in range(n_max):
        n = n_r + l + 1
        E_coulomb = -kappa**2 / (2.0 * n**2)
        # Yukawa has fewer / shallower bound states than Coulomb
        E_min = E_coulomb * 2.0
        E_max = -1e-6
        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            shift_pct = (E_wkb - E_coulomb) / abs(E_coulomb) * 100.0
            results.append((n, E_wkb, E_coulomb, shift_pct))
        except Exception:
            results.append((n, float('nan'), E_coulomb, float('nan')))
    return results


# ---------------------------------------------------------------------------
# 6. Wood-Saxon Potential (nuclear shell model)
# ---------------------------------------------------------------------------

def wood_saxon_potential(r, V0=50.0, R0=4.0, a=0.5):
    """Wood-Saxon: V(r) = -V0 / (1 + exp((r - R0) / a)).

    Parameters (in fm-like atomic units, here rescaled):
      V0 = well depth (positive, so V < 0 inside)
      R0 = nuclear radius
      a  = surface diffuseness

    Used for nuclear shell model mean-field.
    For r << R0: V ~ -V0 (flat well)
    For r >> R0: V ~ 0   (free)
    """
    return -V0 / (1.0 + np.exp((r - R0) / a))


def wood_saxon_energy_approx(n_r, l, V0=50.0, R0=4.0, a=0.5,
                              hbar=HBAR, m=M_E):
    """Approximate WS energy via infinite square well estimate.

    E_{n_r,l} ~ hbar^2 * pi^2 * (n_r+1)^2 / (2*m*R0^2)  [rough guide]

    Only valid for large V0 (deep well). Used to bracket WKB search.
    """
    return hbar**2 * np.pi**2 * (n_r + 1)**2 / (2.0 * m * R0**2)


def solve_wood_saxon_rsg(n_max=3, l=0, V0=50.0, R0=4.0, a=0.5,
                         hbar=HBAR, m=M_E):
    """Solve Wood-Saxon bound states via RSG + WKB + Langer.

    Returns list of (n_r, E_wkb, E_isw_approx, status).
    """
    V = lambda r: wood_saxon_potential(r, V0, R0, a)
    results = []
    for n_r in range(n_max):
        E_isw = wood_saxon_energy_approx(n_r, l, V0, R0, a, hbar, m)
        E_min = -V0 * 1.05
        E_max = -1e-4
        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            results.append((n_r, E_wkb, E_isw, "OK"))
        except Exception:
            results.append((n_r, float('nan'), E_isw, "FAIL"))
    return results


# ---------------------------------------------------------------------------
# 7. Power-Law Potential  V(r) = A * r^p
# ---------------------------------------------------------------------------

def power_law_potential(r, A=1.0, p=1.0):
    """Power-law potential: V(r) = A * r^p.

    Special cases:
      p = -1 : Coulomb (A < 0)
      p =  2 : Harmonic oscillator (A > 0, A = 0.5 * m * omega^2)
      p =  1 : Linear confinement (quark bag model)
      p =  4 : Quartic anharmonic oscillator
    """
    return A * r**p


def power_law_energy_wkb(n_r, l, A=1.0, p=1.0, hbar=HBAR, m=M_E):
    """WKB energy scaling for power-law V = A*r^p.

    Dimensional analysis gives:
      E_n ~ (hbar^2/m)^(p/(p+2)) * A^(2/(p+2)) * n^(2p/(p+2))

    This is an approximate scaling law, not exact (except HO and Coulomb).
    """
    if abs(p + 2.0) < 1e-10:
        raise ValueError("p = -2 is singular (inverse-square)")
    exponent = 2.0 * p / (p + 2.0)
    prefactor = (hbar**2 / m) ** (p / (p + 2.0)) * A ** (2.0 / (p + 2.0))
    n = n_r + 1
    return prefactor * n**exponent


def solve_power_law_rsg(n_max=3, l=0, A=1.0, p=1.0,
                        E_max_bound=200.0, hbar=HBAR, m=M_E):
    """Solve power-law bound states via RSG + WKB + Langer.

    Only for confining (p > 0, A > 0) or attractive (p < 0, A < 0).
    Returns list of (n_r, E_wkb, status).
    """
    if A > 0 and p > 0:
        # Confining: all states bound, energies positive are unphysical
        # Use large positive search range
        E_search_min = 0.01
        E_search_max = E_max_bound
        sign = 1
    elif A < 0 and p < 0:
        # Attractive: bound states negative
        E_search_min = -1000.0
        E_search_max = -1e-6
        sign = -1
    else:
        return [(n_r, float('nan'), "SKIP") for n_r in range(n_max)]

    V = lambda r: power_law_potential(r, A, p)
    results = []
    for n_r in range(n_max):
        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V,
                E_min=E_search_min, E_max=E_search_max,
                use_langer=True, hbar=hbar, m=m
            )
            results.append((n_r, E_wkb, "OK"))
        except Exception:
            results.append((n_r, float('nan'), "FAIL"))
    return results


# ---------------------------------------------------------------------------
# 8. Lennard-Jones Potential
# ---------------------------------------------------------------------------

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """Lennard-Jones 12-6: V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6].

    Parameters:
      epsilon = well depth
      sigma   = zero-crossing radius

    Minimum at r_min = 2^(1/6) * sigma, V_min = -epsilon.
    Weakly bound: few vibrational levels for typical parameters.
    """
    x = sigma / r
    return 4.0 * epsilon * (x**12 - x**6)


def lj_energy_exact_v0(epsilon=1.0, sigma=1.0, m=M_E, hbar=HBAR):
    """Approximate ground-state LJ energy via harmonic expansion at minimum.

    r_min = 2^(1/6) * sigma
    V''(r_min) = 72 * epsilon / (2^(1/3) * sigma^2)
    omega = sqrt(V''(r_min) / m)
    E_0 ~ -epsilon + 0.5 * hbar * omega   (harmonic approx)
    """
    r_min = 2.0**(1.0 / 6.0) * sigma
    v_pp = 72.0 * epsilon / (2.0**(1.0 / 3.0) * sigma**2)
    omega = np.sqrt(v_pp / m)
    return -epsilon + 0.5 * hbar * omega


def solve_lj_spectrum_rsg(v_max=3, l=0, epsilon=1.0, sigma=1.0,
                          hbar=HBAR, m=M_E):
    """Solve LJ bound states via RSG + WKB + Langer.

    Returns list of (v, E_wkb, E_harmonic_approx, rel_err_vs_harmonic).
    """
    V = lambda r: lennard_jones_potential(r, epsilon, sigma)
    E_harm_0 = lj_energy_exact_v0(epsilon, sigma, m, hbar)
    results = []
    for v in range(v_max):
        E_min = -epsilon * 1.05
        E_max = -1e-6
        try:
            E_wkb = bohr_sommerfeld_energy(
                v, l, V,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            # Compare to harmonic ground state shift
            rel_err = (abs(E_wkb - E_harm_0) / abs(E_harm_0)
                       if v == 0 else float('nan'))
            results.append((v, E_wkb, E_harm_0 if v == 0 else float('nan'),
                             rel_err))
        except Exception:
            results.append((v, float('nan'), float('nan'), float('nan')))
    return results


# ---------------------------------------------------------------------------
# 9. Screened Power-Law (interpolates Coulomb -> Yukawa -> free)
# ---------------------------------------------------------------------------

def screened_power_law(r, kappa=1.0, mu=0.0, p=1.0):
    """Generalized screened potential: V(r) = -kappa * r^(-p) * exp(-mu*r).

    Special cases:
      p=1, mu=0 : Coulomb
      p=1, mu>0 : Yukawa
      p=2, mu=0 : -kappa/r^2 (inverse-square, marginal case)
    """
    return -kappa * r**(-p) * np.exp(-mu * r)


# ---------------------------------------------------------------------------
# Helper: generic accuracy survey
# ---------------------------------------------------------------------------

def accuracy_survey(V_func, label, n_r_list, l_list,
                    E_min_list, E_max_list,
                    E_exact_list=None,
                    hbar=HBAR, m=M_E):
    """Run RSG WKB for arbitrary potential and return accuracy table.

    Args:
      V_func       : callable V(r)
      label        : name string for reporting
      n_r_list     : list of radial quantum numbers
      l_list       : list of angular momentum quantum numbers
      E_min_list   : list of lower energy bounds
      E_max_list   : list of upper energy bounds
      E_exact_list : list of exact energies (None if unknown)

    Returns list of dicts with keys:
      label, n_r, l, E_wkb, E_exact, rel_err, status
    """
    results = []
    for i, (n_r, l) in enumerate(zip(n_r_list, l_list)):
        E_min = E_min_list[i]
        E_max = E_max_list[i]
        E_exact = E_exact_list[i] if E_exact_list else None
        try:
            E_wkb = bohr_sommerfeld_energy(
                n_r, l, V_func,
                E_min=E_min, E_max=E_max,
                use_langer=True, hbar=hbar, m=m
            )
            if E_exact is not None and not np.isnan(E_exact):
                rel_err = abs(E_wkb - E_exact) / abs(E_exact)
            else:
                rel_err = float('nan')
            results.append({
                "label": label, "n_r": n_r, "l": l,
                "E_wkb": E_wkb, "E_exact": E_exact,
                "rel_err": rel_err, "status": "OK"
            })
        except Exception as e:
            results.append({
                "label": label, "n_r": n_r, "l": l,
                "E_wkb": float('nan'), "E_exact": E_exact,
                "rel_err": float('nan'), "status": f"FAIL: {e}"
            })
    return results
