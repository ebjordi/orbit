import numpy.testing as npt
import numpy as np
from pytest import fixture
from orbit import Orbit


@fixture
def expected_E():
    return np.array(
        [
            0.0,
            1.34412783,
            1.94137176,
            2.38774944,
            2.77593815,
            3.14159265,
            3.50724716,
            3.89543586,
            4.34181355,
            4.93905748,
        ]
    )


@fixture
def expected_true_anom():
    return np.array(
        [
            0.0,
            2.22772102,
            2.61851838,
            2.83435559,
            2.99720062,
            3.14159265,
            3.28598469,
            3.44882972,
            3.66466693,
            4.05546429,
        ]
    )


@fixture
def expected_rv():
    return np.array(
        [
            -77.21392629,
            -43.06122058,
            -1.15821444,
            21.6174233,
            37.78983684,
            51.01612824,
            62.90800252,
            74.41017756,
            86.0445142,
            95.17152749,
        ]
    )


@fixture(scope="module")
def orbit_instance():
    phases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return Orbit(
        phases=phases,
        T0=57880.634,
        P=29.1333,
        e=0.7346,
        K1=108.3,
        K2=-192.2,
        omega=126.3,
        gamma=34.0,
    )


def test_phases_property():
    JD = [57880.634, 57902.4839, 57924.334]
    expected_phases = [0.0, 0.75, 0.5]
    orbit = Orbit(JD=JD, T0=57880.634, P=29.1333)
    npt.assert_allclose(orbit.phases, expected_phases, rtol=1e-5, atol=1e-6)


def test_eccentric_anomaly_property(orbit_instance, expected_E):
    npt.assert_allclose(orbit_instance.eccentric_anomaly, expected_E, atol=1e-6)


def test_true_anomaly_property(orbit_instance, expected_true_anom):
    npt.assert_allclose(orbit_instance.true_anomaly, expected_true_anom, atol=1e-6)


def test_rv1_property(orbit_instance, expected_rv):
    npt.assert_allclose(orbit_instance.rv1, expected_rv, atol=1e-6)
