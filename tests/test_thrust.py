import pytest
from lunar_lander.thrust import ThrustController


def test_initial_state():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.1, max_change_rate=5000.0, dt=0.01)
    assert tc.current_thrust == 0.0
    assert tc.last_commanded == 0.0


def test_step_returns_thrust():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.0, max_change_rate=1e9, dt=0.01)
    tc.set_command(5000.0)
    thrust = tc.step()
    assert thrust > 0
    assert thrust <= 5000.0


def test_thrust_limited_by_max_change_rate():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.0, max_change_rate=1000.0, dt=0.01)
    tc.set_command(5000.0)

    thrust = tc.step()
    max_delta = 1000.0 * 0.01
    assert abs(thrust - 0.0) <= max_delta + 1e-9


def test_response_delay_delays_effect():
    delay = 0.1
    dt = 0.01
    delay_steps = int(round(delay / dt))

    tc = ThrustController(max_thrust=10000.0, response_delay=delay, max_change_rate=1e9, dt=dt)
    tc.set_command(5000.0)

    for _ in range(delay_steps):
        thrust = tc.step()
        assert thrust == 0.0 or abs(thrust) < 1e-9


def test_thrust_clamped_to_zero_and_max():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.0, max_change_rate=1e9, dt=0.01)

    tc.set_command(-100.0)
    tc.step()
    assert tc.current_thrust >= -1e-9

    tc.set_command(20000.0)
    for _ in range(100):
        tc.step()
    assert tc.current_thrust <= 10000.0 + 1e-9


def test_reset_clears_state():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.05, max_change_rate=5000.0, dt=0.01)
    tc.set_command(8000.0)
    for _ in range(10):
        tc.step()

    assert tc.current_thrust > 0

    tc.reset()
    assert tc.current_thrust == 0.0
    assert tc.last_commanded == 0.0


def test_last_commanded_tracks_latest():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.1, max_change_rate=5000.0, dt=0.01)

    tc.set_command(3000.0)
    assert tc.last_commanded == 3000.0

    tc.set_command(7000.0)
    assert tc.last_commanded == 7000.0


def test_thrust_can_decrease():
    tc = ThrustController(max_thrust=10000.0, response_delay=0.0, max_change_rate=2000.0, dt=0.01)

    tc.set_command(5000.0)
    for _ in range(200):
        tc.step()

    high_thrust = tc.current_thrust

    tc.set_command(1000.0)
    for _ in range(50):
        tc.step()

    assert tc.current_thrust < high_thrust
