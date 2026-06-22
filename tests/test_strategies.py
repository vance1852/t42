import pytest
from lunar_lander.strategies import (
    ConstantDecelerationStrategy,
    StagedBrakingStrategy,
    PIDStrategy,
    create_strategy,
)


def test_constant_decel_braking_when_falling():
    strat = ConstantDecelerationStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, target_decel=2.0)
    mass = 1500.0
    thrust = strat.compute_thrust(altitude=1000.0, velocity=-50.0, mass=mass, time=0.0)
    assert thrust > mass * 1.62
    assert thrust <= 30000.0


def test_constant_decel_reduces_thrust_when_above_target():
    strat = ConstantDecelerationStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, target_decel=2.0)
    mass = 1500.0
    thrust_rising = strat.compute_thrust(altitude=400.0, velocity=5.0, mass=mass, time=0.0)
    hover = mass * 1.62
    assert thrust_rising < hover


def test_constant_decel_zero_thrust_on_ground():
    strat = ConstantDecelerationStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, target_decel=2.0)
    thrust = strat.compute_thrust(altitude=0.0, velocity=-5.0, mass=1500.0, time=0.0)
    assert thrust == 0.0


def test_constant_decel_clamped_to_max():
    strat = ConstantDecelerationStrategy(max_thrust=1000.0, gravity=1.62, mass_dry=1000.0, target_decel=10.0)
    thrust = strat.compute_thrust(altitude=1000.0, velocity=-50.0, mass=2000.0, time=0.0)
    assert thrust == 1000.0


def test_staged_braking_increases_thrust_at_low_alt():
    strat = StagedBrakingStrategy(
        max_thrust=30000.0, gravity=1.62, mass_dry=1000.0,
        high_altitude_thrust_pct=0.3, low_altitude_thrust_pct=0.7,
        mid_low_threshold=200.0, final_burn_altitude=50.0, final_burn_thrust_pct=0.95,
    )
    thrust_high = strat.compute_thrust(altitude=1500.0, velocity=-30.0, mass=1500.0, time=0.0)
    thrust_low = strat.compute_thrust(altitude=100.0, velocity=-10.0, mass=1500.0, time=0.0)
    assert thrust_low > thrust_high


def test_staged_braking_zero_on_ground():
    strat = StagedBrakingStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0)
    thrust = strat.compute_thrust(altitude=0.0, velocity=-5.0, mass=1500.0, time=0.0)
    assert thrust == 0.0


def test_pid_output_bounded():
    strat = PIDStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, kp=1.5, ki=0.05, kd=0.3)
    for v in [-30, -10, -5, 0, 5]:
        thrust = strat.compute_thrust(altitude=500.0, velocity=v, mass=1500.0, time=0.0)
        assert 0 <= thrust <= 30000.0


def test_pid_reset_clears_integral():
    strat = PIDStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0, kp=0.3, ki=1.0, kd=0.1)

    for _ in range(50):
        strat.compute_thrust(altitude=200.0, velocity=-35.0, mass=1500.0, time=0.0)

    thrust_before = strat.compute_thrust(altitude=200.0, velocity=-35.0, mass=1500.0, time=0.0)

    strat.reset()

    thrust_after = strat.compute_thrust(altitude=200.0, velocity=-35.0, mass=1500.0, time=0.0)
    assert abs(thrust_after - thrust_before) > 1.0


def test_pid_zero_on_ground():
    strat = PIDStrategy(max_thrust=30000.0, gravity=1.62, mass_dry=1000.0)
    thrust = strat.compute_thrust(altitude=0.0, velocity=-5.0, mass=1500.0, time=0.0)
    assert thrust == 0.0


def test_create_strategy_invalid():
    with pytest.raises(ValueError):
        create_strategy("invalid_strategy", 10000.0, 1.62, 1000.0)


def test_create_strategy_valid():
    for name in ["constant_decel", "staged_braking", "pid"]:
        strat = create_strategy(name, 30000.0, 1.62, 1000.0)
        assert strat is not None
        assert strat.name == name
