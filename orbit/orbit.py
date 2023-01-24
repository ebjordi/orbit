import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class Orbit:
    def __init__(
        self,
        JD=None,
        phases=None,
        K1=108.3,
        K2=-192.2,
        T0=57880.634,
        P=29.1333,
        e=0.734,
        omega=126.3,
        gamma=34.0,
        a = 1.
    ):
        if JD is None and phases is None:
            raise ValueError("Both JD and phases cannot be None.")
        if JD is not None:
            self.JD = np.array(JD)
        else:
            self.JD = None
            self._phases = np.array(phases)
        self.K1 = K1
        self.K2 = K2
        self.T0 = T0
        self.P = P
        self.e = e
        self.omega = omega
        self.gamma = gamma
        self.a = a

    @property
    def phases(self):
        if not hasattr(self, "_phases"):
            self._phases = self.calc_phases()
        return self._phases

    @property
    def phi(self):
        return self.phases

    @property
    def eccentric_anomaly(self):
        if not hasattr(self, "_eccentric_anomaly"):
            self._eccentric_anomaly = self.calc_eccentric_anomaly()
        return self._eccentric_anomaly

    @property
    def E(self):
        return self.eccentric_anomaly

    @property
    def true_anomaly(self):
        if not hasattr(self, "_true_anomaly"):
            self._true_anomaly = self.calc_true_anomaly()
        return self._true_anomaly

    @property
    def theta(self):
        return self.true_anomaly


    @property
    def radiovector1(self):
        if not hasattr(self, "_radiovector1"):
            self._radiovector1 = self.calc_radiovector(1)
        return self._radiovector1

    @property
    def r1(self):
        return self.radiovector1

    @property
    def radiovector2(self):
        if not hasattr(self, "_radiovector2"):
            self._radiovector2 = self.calc_radiovector(-1)
        return self._radiovector2

    @property
    def r2(self):
        return self.radiovector2


    @property
    def radiovector2(self):
        if not hasattr(self, "_radiovector2"):
            self._radiovector2 = self.calc_radiovector(2)
        return self._radiovector2        
    
    @property
    def rv1(self):
        if not hasattr(self, "_rv1"):
            self._rv1 = self.calc_rv(self.K1)
        return self._rv1

    @property
    def rv2(self):
        if not hasattr(self, "_rv2"):
            self._rv2 = self.calc_rv(self.K2)
        return self._rv2

    @property
    def rv(self):
        if not hasattr(self, "_rv"):
            self._rv = np.array([self.rv1, self.rv2])
        return self._rv

    @property
    def vr(self):
        return self.rv

    @property
    def vr1(self):
        return self.rv1

    @property
    def vr2(self):
        return self.rv2

    @property
    def functions(self):
        if not hasattr(self, "_functions"):
            self._functions = self.interpolate()
        return self._functions

    def calc_phases(self):
        T0 = self.T0 * np.ones_like(self.JD)
        pha = (self.JD - T0) / self.P
        pha = pha - pha.astype(int)
        pha[pha < 0.0] += 1.0
        return pha

    def calc_eccentric_anomaly(self):
        """Calculates the eccentric anomaly of an orbit given a phase"""
        phi = self.phi * 2 * np.pi
        E0 = phi
        count = 0
        no_convergence = True
        while no_convergence:
            count += 1
            E = E0 - ((E0 - self.e * np.sin(E0) - phi) / (1 - self.e * np.cos(E0)))
            if np.allclose(E, E0):
                E[E > 2 * np.pi] -= 2 * np.pi
                E[E < 0] += 2 * np.pi
                return E
            E0 = E
            if count > 10000:
                print(count)
                raise ValueError("Too many iteration")

    def calc_true_anomaly(self):
        """Calculates the true anomaly of an orbit given the eccentric anomaly and eccentricity."""

        E = self.eccentric_anomaly
        theta = 2 * np.arctan(np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(E / 2))
        theta[theta < 0] = theta[theta < 0] + 2 * np.pi
        return theta

    def calc_radiovector(self,identifier):
        i = identifier
        return (self.a*(1 - i * self.e**2))/(1+ i * self.e * np.cos(self.theta))

    def calc_rv(self, K):

        ω = self.omega * np.pi / 180.0
        θ = self.true_anomaly
        rv = K * (np.cos(θ + ω) + self.e * np.cos(ω))
        return rv + self.gamma

    def __str__(self):
        return f"Orbital parameters for object:\n K1={self.K1},\nK2={self.K2},\nT0={self.T0},\nP={self.P},\ne={self.e},\nomega={self.omega},\ngamma={self.gamma})"

    def __repr__(self):
        return f"orbit.Orbit(JD = {self.JD}, phases={self.phases}, K1={self.K1}, K2={self.K2}, T0={self.T0}, P={self.P}, e={self.e}, omega={self.omega}, gamma={self.gamma})"

    @classmethod
    def from_linspace(
        cls, points=1200, e=0.734, K1=108.3, K2=-192.2, omega=126.3, gamma=34, a = 1
    ):
        T0 = None
        P = None
        phases = np.linspace(0.0, 1.0, points)
        return cls(JD=None, phases=phases, K1=K1, K2=K2, e=e, omega=omega, gamma=gamma, a = a)

    def interpolate(self):
        primary = interp1d(self.phases, self.rv1, kind="cubic")
        secondary = interp1d(self.phases, self.rv2, kind="cubic")
        return primary, secondary


#
#  def orbit_function(kepler_file: str):
#    """Given a `kepler` output file returns interpolated functions for primary and
#    secondary components of a binary system"""
#    names = ["fase", "vr-p", "vr-s"]
#    df = pd.read_table(kepler_file, names=names, sep=r'\s+', skiprows=1, index_col=False)
#    primary = interp1d(df["fase"], df["vr-p"], kind="cubic")
#    secondary = interp1d(df["fase"], df["vr-s"], kind="cubic")
#    return primary, secondary
#
