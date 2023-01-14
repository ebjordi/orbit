import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def phase(JD, T0=57880.634, P=29.1333):
    """Calculates the phase of an orbit given the Julian date (JD), the time of periastron passage (T0), the period (P), and a flag to indicate whether to return the mean anomaly or not. Default values are for i Orionis and are updated to Eguren 2021."""
    JD = JD * np.ones_like(JD)
    T0 = T0 * np.ones_like(JD)
    pha = (JD - T0) / P
    pha = pha - pha.astype(int)
    if isinstance(pha, np.float64):
        if pha < 0:
            pha += 1
    else:
        pha[pha < 0.0] += 1.0
    return pha


def eccentric_anomaly(phi, e=0.734):
    """Calculates the eccentric anomaly of an orbit given a phase"""
    phi = np.array(phi) * 2 * np.pi
    E0 = phi
    count = 0
    no_convergence = True
    while no_convergence:
        count += 1
        E = E0 - ((E0 - e * np.sin(E0) - phi) / (1 - e * np.cos(E0)))
        if isinstance(E, np.float64):
            if np.isclose(E, E0):
                if E > 2 * np.pi:
                    E -= 2 * np.pi
                if E < 0:
                    E += 2 * np.pi
                return E
        elif np.allclose(E, E0):
            E[E > 2 * np.pi] -= 2 * np.pi
            E[E < 0] += 2 * np.pi
            return E
        E0 = E
        if count > 10000:
            print(count)
            raise ValueError("Too many iteration")


def true_anomaly(eccentric_anomaly, e=0.734):
    """Calculates the true anomaly of an orbit given the eccentric anomaly and eccentricity."""

    E = np.array(eccentric_anomaly) * np.ones_like(eccentric_anomaly)
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    if isinstance(theta, np.ndarray):
        theta[theta < 0] = theta[theta < 0] + 2 * np.pi
    elif theta < 0:
        theta += 2 * np.pi

    return theta


def rv(true_anomaly, K=108.3, e=0.734, omega=126.3, gamma=34):
    """Calculates the radial velocity of an object in orbit given the true anomaly, and other orbital parameters such as velocity semi-amplitude (K), eccentricity (e), longitude of periastron (omega), and systemic velocity (gamma)."""
    ω = omega * np.pi / 180
    θ = true_anomaly
    Vrad = K * (np.cos(θ + ω) + e * np.cos(ω))
    return Vrad + gamma


def velocity_curve_jd(
    JD, T0=57880.63, P=29.1333, e=0.734, K=108.3, omega=126.3, gamma=34
):
    """Calculates the radial velocity of an object in orbit for a given Julian date (JD) using the `phase`, `eccentric_anomaly` , `true_anomaly` and radial velocity(`rv`) functions."""
    φ = phase(JD, T0=T0, P=P)
    E = eccentric_anomaly(φ, e=e)
    θ = true_anomaly(E, e=e)
    vr = rv(θ, K=K, omega=omega, gamma=gamma)
    return vr


def velocity_curve_from_phase(
    points=1200, a=0, b=1.2, e=0.734, K=108.3, omega=126.3, gamma=34
):
    """Calculates the radial velocity of an object in orbit for a given range of phases using the phase, eccentric anomaly, true anomaly and radial velocity functions. The number of points, the range of the phase and other orbital parameters can be given as input."""
    φ = np.linspace(a, b, points)
    E = eccentric_anomaly(φ, e=e)
    θ = true_anomaly(E, e=e)
    vr = rv(θ, K=K, omega=omega, gamma=gamma)
    return φ, vr


def orbit_function(kepler_file: str):
    """Given a `kepler` output file returns interpolated functions for primary and
    secondary components of a binary system"""
    names = ["fase", "vr-p", "vr-s"]
    df = pd.read_table(
        kepler_file, names=names, sep=r"\s+", skiprows=1, index_col=False
    )
    primary = interp1d(df["fase"], df["vr-p"], kind="cubic")
    secondary = interp1d(df["fase"], df["vr-s"], kind="cubic")
    return primary, secondary
