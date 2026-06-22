import pytest
from lunar_lander.strategies import (
    ConstantDecelerationStrategy,
    StagedBrakingStrategy,
    PIDStrategy,
    create_strategy,
)


def test_constant_decel_basic():
    strat = ConstantDecelerationStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, target_decel=2.0)
    mass = 1500.0
    thrust = strat.compute_thrust(altitude=1000.0, velocity=-50.0, mass=mass, time=0.0)

    expected = mass * (1.62 + 2.0)
    assert abs(thrust - expected) < 1e-9
    assert 0 <= thrust <= 30000.0


def test_constant_decel_positive_velocity_no_thrust():
    strat = ConstantDecelerationStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0)
    thrust = strat.compute_thrust(altitude=1000.0, velocity=10.0, mass=1500.0, time=0.0)
    assert thrust == 0.0


def test_constant_decel_clamped_to_max():
    strat = ConstantDecelerationStrategy(max_thrust=1000.0, gravity=1.62, mass_dry=1000.0, target_decel=10.0)
    thrust = strat.compute_thrust(altitude=1000.0, velocity=-50.0, mass=2000.0, time=0.0)
    assert thrust == 1000.0


def test_staged_braking_high_alt():
    strat = StagedBrakingStrategy(
        max_thrust=30000.0, gravity=1.62, mass_dry=1000.0,
        high_altitude_thrust_pct=0.8, high_mid_threshold=1000.0,
    )
    thrust = strat.compute_thrust(altitude=1500.0, velocity=-50.0, mass=1500.0, time=0.0)
    assert abs(thrust - 0.8 * 30000.0) < 1e-9


def test_staged_braking_low_alt():
    strat = StagedBrakingStrategy(
        max_thrust=30000.0, gravity=1.62, mass_dry=1000.0,
        low_altitude_thrust_pct=0.4, mid_low_threshold=200.0,
        final_burn_altitude=50.0,
    )
    thrust = strat.compute_thrust(altitude=100.0, velocity=-20.0, mass=1500.0, time=0.0)
    assert abs(thrust - 0.4 * 30000.0) < 1e-9


def test_staged_braking_final_burn():
    strat = StagedBrakingStrategy(
        max_thrust=30000.0, gravity=1.62, mass_dry=1000.0,
        final_burn_altitude=50.0, final_burn_thrust_pct=0.9,
    )
    thrust = strat.compute_thrust(altitude=30.0, velocity=-10.0, mass=1500.0, time=0.0)
    assert abs(thrust - 0.9 * 30000.0) < 1e-9


def test_pid_basic():
    strat = PIDStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, kp=0.5, ki=0.0, kd=0.0)
    mass = 1500.0
    thrust = strat.compute_thrust(altitude=1000.0, velocity=-10.0, mass=mass, time=0.0)
    assert thrust >= 0
    assert thrust <= 30000.0


def test_pid_reset_clears_integral():
    strat = PIDStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, kp=0.0, ki=1.0, kd=0.0)

    for _ in range(100):
        strat.compute_thrust(altitude=1000.0, velocity=-10.0, mass=1500.0, time=0.0)

    thrust_before_reset = strat.compute_thrust(altitude=1000.0, velocity=-10.0, mass=1500.0, time=0.0)

    strat.reset()

    thrust_after_reset = strat.compute_thrust(altitude=1000.0, velocity=-10.0, mass=1500.0, time=0.0)

    assert thrust_after_reset != thrust_before_reset


def test_create_strategy_invalid():
    with pytest.raises(ValueError):
        create_strategy("invalid_strategy", 10000.0, 1.62, 1000.0)


def test_create_strategy_valid():
    for name in ["constant_decel", "staged_braking", "pid"]:
        strat = create_strategy(name, 30000.0, 1.62, 1000.0)
        assert strat is not None
        assert strat.name == name
