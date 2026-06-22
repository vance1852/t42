import pytest
import numpy as np
from lunar_lander.physics import PhysicsEngine


def test_step_zero_thrust_free_fall():
    physics = PhysicsEngine(gravity=1.62)
    alt0, vel0, mass0 = 100.0, 0.0, 1000.0
    dt = 1.0

    alt1, vel1, mass1 = physics.step(alt0, vel0, mass0, 0.0, dt)

    expected_vel = -1.62
    expected_alt = 100.0 - 0.5 * 1.62

    assert abs(vel1 - expected_vel) < 1e-10
    assert abs(alt1 - expected_alt) < 1e-10
    assert abs(mass1 - mass0) < 1e-10


def test_step_with_thrust_acceleration():
    physics = PhysicsEngine(gravity=1.62)
    alt0, vel0, mass0 = 100.0, 0.0, 1000.0
    thrust = 10000.0
    dt = 1.0

    alt1, vel1, mass1 = physics.step(alt0, vel0, mass0, thrust, dt)

    expected_accel = 10000.0 / 1000.0 - 1.62
    expected_vel = expected_accel * dt
    expected_alt = alt0 + 0.5 * expected_accel * dt * dt

    assert abs(vel1 - expected_vel) < 1e-10
    assert abs(alt1 - expected_alt) < 1e-10
    assert mass1 < mass0


def test_fuel_consumption_increases_with_thrust():
    physics = PhysicsEngine(gravity=1.62)
    rate1 = physics.fuel_consumption_rate(1000.0)
    rate2 = physics.fuel_consumption_rate(2000.0)
    assert rate2 > rate1
    assert rate1 > 0


def test_fuel_consumption_proportional_to_time():
    physics = PhysicsEngine(gravity=1.62)
    alt0, vel0, mass0 = 100.0, 0.0, 1000.0
    thrust = 10000.0

    _, _, mass_short = physics.step(alt0, vel0, mass0, thrust, 0.1)
    _, _, mass_long = physics.step(alt0, vel0, mass0, thrust, 0.2)

    fuel_short = mass0 - mass_short
    fuel_long = mass0 - mass_long

    assert abs(fuel_long - 2 * fuel_short) / fuel_short < 0.01


def test_mass_decreases_monotonically():
    physics = PhysicsEngine(gravity=1.62)
    alt, vel, mass = 1000.0, -10.0, 1500.0
    thrust = 20000.0
    dt = 0.1

    prev_mass = mass
    for _ in range(100):
        alt, vel, mass = physics.step(alt, vel, mass, thrust, dt)
        assert mass <= prev_mass + 1e-10
        prev_mass = mass


def test_gravity_independent_of_motion_direction():
    physics = PhysicsEngine(gravity=1.62)
    alt0 = 500.0
    mass = 1000.0
    thrust = 0.0
    dt = 0.5

    vel_pos = 20.0
    _, vel1, _ = physics.step(alt0, vel_pos, mass, thrust, dt)
    delta1 = vel1 - vel_pos

    vel_neg = -20.0
    _, vel2, _ = physics.step(alt0, vel_neg, mass, thrust, dt)
    delta2 = vel2 - vel_neg

    assert abs(delta1 - delta2) < 1e-10
    assert delta1 < 0
