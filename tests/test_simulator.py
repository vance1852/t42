import pytest
from lunar_lander.config import LanderConfig
from lunar_lander.simulator import LanderSimulator
from lunar_lander.strategies import create_strategy
from lunar_lander.scoring import LandingStatus


def get_default_config():
    return LanderConfig(
        mass_dry=1000.0,
        mass_fuel_initial=800.0,
        max_thrust=40000.0,
        thrust_response_delay=0.1,
        thrust_max_change_rate=30000.0,
        gravity=1.62,
        initial_altitude=500.0,
        initial_velocity=-20.0,
        dt=0.01,
        altimeter_noise_std=0.0,
        velocimeter_noise_std=0.0,
        filter_alpha=1.0,
        max_simulation_time=100.0,
        hover_time_limit=30.0,
        soft_landing_velocity=-2.0,
        hard_landing_velocity=-10.0,
        target_altitude=0.0,
        random_seed=42,
    )


def test_simulator_runs_to_completion():
    config = get_default_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    assert result is not None
    assert len(result.states) > 1
    assert result.status is not None


def test_simulator_altitude_decreases():
    config = get_default_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry, target_decel=1.0)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    initial_alt = result.states[0].altitude
    final_alt = result.states[-1].altitude
    assert final_alt < initial_alt


def test_simulator_fuel_decreases():
    config = get_default_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    initial_fuel = result.states[0].fuel_mass
    final_fuel = result.states[-1].fuel_mass
    assert final_fuel <= initial_fuel + 1e-9


def test_simulator_time_matches_dt():
    config = get_default_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    expected_time = (len(result.states) - 1) * config.dt
    assert abs(result.states[-1].time - expected_time) < 1e-9


def test_simulator_classifies_soft_landing():
    config = get_default_config()
    config.initial_altitude = 5.0
    config.initial_velocity = -1.0
    config.soft_landing_velocity = -5.0
    config.hover_time_limit = 100.0
    config.max_simulation_time = 200.0
    config.max_thrust = 0.0
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    assert result.status == LandingStatus.SOFT_LANDING
    assert result.score.overall_score > 0


def test_simulator_classifies_hard_landing():
    config = get_default_config()
    config.max_thrust = 1000.0
    config.hard_landing_velocity = -5.0
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    assert result.status == LandingStatus.HARD_LANDING
    assert result.score.overall_score == 0.0


def test_simulator_fuel_depleted():
    config = get_default_config()
    config.mass_fuel_initial = 1.0
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    assert result.status == LandingStatus.FUEL_DEPLETED
    assert result.final_state.fuel_mass <= 1e-6


def test_simulator_respects_hover_limit():
    config = get_default_config()
    config.initial_altitude = 100.0
    config.initial_velocity = 0.0
    config.hover_time_limit = 5.0
    config.max_simulation_time = 100.0

    class HoverStrategy:
        name = "hover_test"

        def __init__(self, max_thrust, gravity, mass_dry, **kwargs):
            self.max_thrust = max_thrust
            self.gravity = gravity
            self.mass_dry = mass_dry

        def compute_thrust(self, altitude, velocity, mass, time):
            return mass * self.gravity

        def reset(self):
            pass

    sim = LanderSimulator(config, HoverStrategy(config.max_thrust, config.gravity, config.mass_dry))
    result = sim.run()

    assert result.status == LandingStatus.HOVER_TOO_LONG


def test_simulator_simulation_timeout():
    config = get_default_config()
    config.initial_altitude = 10000.0
    config.initial_velocity = -1.0
    config.max_simulation_time = 1.0
    config.max_thrust = 100000.0
    config.hover_time_limit = 100.0

    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry, target_decel=0.1)
    sim = LanderSimulator(config, strategy)
    result = sim.run()

    assert result.status == LandingStatus.SIMULATION_TIMEOUT


def test_simulator_over_thrust_request():
    config = get_default_config()
    config.max_thrust = 5000.0

    class HighThrustStrategy:
        name = "high_thrust"

        def __init__(self, max_thrust, gravity, mass_dry, **kwargs):
            self.max_thrust = max_thrust
            self.gravity = gravity
            self.mass_dry = mass_dry

        def compute_thrust(self, altitude, velocity, mass, time):
            return 10 * self.max_thrust

        def reset(self):
            pass

    sim = LanderSimulator(config, HighThrustStrategy(config.max_thrust, config.gravity, config.mass_dry))
    result = sim.run()

    assert result.status == LandingStatus.OVER_THRUST_REQUEST


def test_three_strategies_run():
    config = get_default_config()
    config.initial_altitude = 300.0
    config.initial_velocity = -30.0
    config.max_thrust = 40000.0

    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry)
        sim = LanderSimulator(config, strategy)
        result = sim.run()
        assert result is not None
        assert result.status is not None
