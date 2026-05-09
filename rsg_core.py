"""rsg_core.py
Radial Scaling Gauge (RSG) -- Core module.

The RSG maps the radial half-line r in (0,inf) to a flat computational
coordinate x in (-inf,inf) via the logarithmic transformation r = exp(x).

In the transformed coordinate:
 - The singular origin r=0 is shifted to x -> -inf
 - The radial measure becomes: dr = r dx = exp(x) dx
 - The wavefunction rescales: R(r) -> u(x) = sqrt(r) * R(r)
 - The angular momentum barrier gets the Langer correction:
     l(l+1) -> (l + 1/2)^2

This is NOT a modification of quantum mechanics. It is a geometric
reparametrization that makes the WKB treatment exact for radial systems.

Authors: Carmen N. Wrede, Lino P. Casu, Bingsi
"""

import numpy as np
from scipy import integrate
from scipy.optimize import brentq

# Physical constants (atomic units: hbar=1, m_e=1, e=1, a_0=1)
HBAR = 1.0
M_E = 1.0
KAPPA_H = 1.0  # e^2 for hydrogen in atomic units


def rsg_transform(r):
    """Log-radial transformation: x = log(r), so r = exp(x)."""
    return np.log(r)


def rsg_inverse(x):
    """Inverse transformation: r = exp(x)."""
    return np.exp(x)


def langer_angular_term(l):
    """Langer-corrected angular momentum term: (l + 1/2)^2.

    The extra 1/4 arises from consistent transformation of the
    radial measure and operator under log-scaling -- NOT ad hoc.
    """
    return (l + 0.5) ** 2


def naive_angular_term(l):
    """Standard (uncorrected) angular momentum term l*(l+1)."""
    return l * (l + 1)


def radial_momentum_langer(r, E, V_func, l, hbar=HBAR, m=M_E):
    """Langer-corrected radial momentum squared.

    p_r^2(r) = 2m(E - V(r)) - hbar^2*(l+1/2)^2 / r^2
    """
    V = V_func(r)
    ang = langer_angular_term(l)
    return 2.0 * m * (E - V) - hbar**2 * ang / r**2


def radial_momentum_naive(r, E, V_func, l, hbar=HBAR, m=M_E):
    """Naive (uncorrected) radial momentum squared."""
    V = V_func(r)
    ang = naive_angular_term(l)
    return 2.0 * m * (E - V) - hbar**2 * ang / r**2


def find_turning_points(E, V_func, l, r_min=1e-6, r_max=1e4,
                        use_langer=True, hbar=HBAR, m=M_E):
    """Find classical turning points r1 < r2 where p_r^2(r) = 0."""
    if use_langer:
        def pr2(r):
            return radial_momentum_langer(
                np.array([r]), E, V_func, l, hbar, m)[0]
    else:
        def pr2(r):
            return radial_momentum_naive(
                np.array([r]), E, V_func, l, hbar, m)[0]

    r_scan = np.logspace(np.log10(r_min), np.log10(r_max), 10000)
    pr2_vals = np.array([pr2(r) for r in r_scan])

    turning = []
    for i in range(len(r_scan) - 1):
        if pr2_vals[i] * pr2_vals[i + 1] < 0:
            try:
                root = brentq(pr2, r_scan[i], r_scan[i + 1])
                turning.append(root)
            except ValueError:
                pass
        if len(turning) == 2:
            break

    if len(turning) < 2:
        raise ValueError(
            f"Could not find two turning points for E={E:.6f}, l={l}")
    return turning[0], turning[1]


def wkb_action_integral(E, V_func, l, use_langer=True, hbar=HBAR, m=M_E):
    """Compute the WKB radial action integral.

    I = integral p_r(r) dr  (from r1 to r2)
    Bohr-Sommerfeld: I = pi * hbar * (n_r + 1/2)
    """
    r1, r2 = find_turning_points(E, V_func, l,
                                  use_langer=use_langer,
                                  hbar=hbar, m=m)

    def integrand(r):
        if use_langer:
            pr2 = radial_momentum_langer(
                np.array([r]), E, V_func, l, hbar, m)[0]
        else:
            pr2 = radial_momentum_naive(
                np.array([r]), E, V_func, l, hbar, m)[0]
        return np.sqrt(pr2) if pr2 > 0 else 0.0

    result, _ = integrate.quad(integrand, r1, r2, limit=200, epsrel=1e-10)
    return result


def bohr_sommerfeld_energy(n_r, l, V_func, E_min=-10.0, E_max=-1e-6,
                           use_langer=True, hbar=HBAR, m=M_E, tol=1e-9):
    """Find energy eigenvalue via Bohr-Sommerfeld quantization.

    Condition: integral p_r(r) dr = pi * hbar * (n_r + 1/2)
    """
    target = np.pi * hbar * (n_r + 0.5)

    def residual(E):
        try:
            I = wkb_action_integral(E, V_func, l,
                                     use_langer=use_langer,
                                     hbar=hbar, m=m)
            return I - target
        except Exception:
            return -target

    E_scan = np.linspace(E_min, E_max, 500)
    residuals = []
    for E in E_scan:
        try:
            residuals.append(residual(E))
        except Exception:
            residuals.append(float('nan'))

    bracket = None
    for i in range(len(E_scan) - 1):
        r1v, r2v = residuals[i], residuals[i + 1]
        if (not np.isnan(r1v) and not np.isnan(r2v) and r1v * r2v < 0):
            bracket = (E_scan[i], E_scan[i + 1])
            break

    if bracket is None:
        raise ValueError(f"No energy bracket found for n_r={n_r}, l={l}")

    return brentq(residual, bracket[0], bracket[1], xtol=tol)


def bohr_energy_exact(n, kappa=KAPPA_H, hbar=HBAR, m=M_E):
    """Exact Bohr energy: E_n = -m*kappa^2 / (2*hbar^2*n^2).

    In atomic units (hbar=1, m=1, kappa=1): E_n = -1/(2n^2)
    """
    return -m * kappa**2 / (2.0 * hbar**2 * n**2)
