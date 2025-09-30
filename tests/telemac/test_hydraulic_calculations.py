# tests/telemac/test_hydraulic_calculations.py


import pytest

from telemac.modules.hydraulic_calculations import HydraulicCalculations

# Define a tolerance for comparing floating-point numbers
TOLERANCE = 1e-4


def test_critical_depth_simple():
    """
    Tests the simplified critical depth calculation with a known value.
    Formula: yc = (q^2 / g)^(1/3), where q = Q/W
    """
    flow_rate = 0.01  # m^3/s
    width = 0.3  # m
    q = flow_rate / width  # m^2/s
    expected_yc = (q**2 / 9.81) ** (1 / 3)  # ~0.0483 m

    calculated_yc = HydraulicCalculations.critical_depth_simple(flow_rate, width)

    assert isinstance(calculated_yc, float)
    assert calculated_yc == pytest.approx(expected_yc, abs=TOLERANCE)


def test_normal_depth_simple():
    """
    Tests the simplified normal depth calculation with a known value.
    Formula: yn = (n*q / S^(1/2))^(3/5)
    """
    flow_rate = 0.015  # m^3/s
    width = 0.3  # m
    slope = 0.001  # 0.1%
    n = 0.025  # s/m^(1/3)
    q = flow_rate / width

    expected_yn = ((n * q) / (slope**0.5)) ** (3 / 5)  # ~0.103 m

    calculated_yn = HydraulicCalculations.normal_depth_simple(
        flow_rate, width, slope, n
    )

    assert isinstance(calculated_yn, float)
    assert calculated_yn == pytest.approx(expected_yn, abs=TOLERANCE)


@pytest.mark.parametrize(
    "flow_rate, width, slope, n, expected_subcritical",
    [
        (0.015, 0.3, 0.0001, 0.025, True),  # Mild slope, expect subcritical (yn > yc)
        (0.015, 0.3, 0.01, 0.012, False),  # Steep slope, expect supercritical (yn < yc)
        (0.01, 0.3, 0.0022, 0.015, True),  # A case closer to critical
    ],
)
def test_subcritical_determination(flow_rate, width, slope, n, expected_subcritical):
    """
    Tests if the relationship between normal and critical depth correctly
    determines the flow regime.
    """
    # We test this indirectly within ParameterManager, but here we test the core logic.
    yn = HydraulicCalculations.normal_depth_simple(flow_rate, width, slope, n)
    yc = HydraulicCalculations.critical_depth_simple(flow_rate, width)

    is_subcritical = yn > yc

    assert is_subcritical == expected_subcritical
