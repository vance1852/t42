import pytest
from lunar_lander.config import LanderConfig
from lunar_lander.simulator import LanderSimulator
from lunar_lander.strategies import create_strategy
from lunar_lander.scoring import LandingStatus


def get_workable_config():
    return LanderConfig(
        mass_dry=1000.0,
        mass_fuel_initial=500.0,
        max_thrust=30000.0,
        thrust_response_delay=0.1,
        thrust_max_change_rate=30000.0,
        gravity=1.62,
        initial_altitude=500.0,
        initial_velocity=-20.0,
        dt=0.01,
        altimeter_noise_std=0.0,
        velocimeter_noise_std=0.0,
        filter_alpha=1.0,
        max_simulation_time=300.0,
        hover_time_limit=60.0,
        soft_landing_velocity=-3.0,
        hard_landing_velocity=-10.0,
        target_altitude=0.0,
        random_seed=42,
    )


def test_simulator_constant_decel_lands_successfully():
    config = get_workable_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    assert result.status == LandingStatus.SOFT_LANDING
    assert result.score.overall_score > 0
    assert result.final_state.altitude <= 0.1
    assert result.fuel_used < config.mass_fuel_initial


def test_simulator_pid_lands_successfully():
    config = get_workable_config()
    strategy = create_strategy("pid", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    assert result.status == LandingStatus.SOFT_LANDING
    assert result.score.overall_score > 0


def test_simulator_altitude_decreases():
    config = get_workable_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    assert result.states[-1].altitude < result.states[0].altitude


def test_simulator_fuel_decreases():
    config = get_workable_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    assert result.fuel_used > 0
    assert result.states[-1].fuel_mass < result.states[0].fuel_mass


def test_simulator_time_matches_dt():
    config = get_workable_config()
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    expected_time = (len(result.states) - 1) * config.dt
    assert abs(result.states[-1].time - expected_time) < 1e-9


def test_simulator_classifies_hard_landing():
    config = get_workable_config()
    config.max_thrust = 1000.0
    config.hard_landing_velocity = -5.0
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    assert result.status == LandingStatus.HARD_LANDING
    assert result.score.overall_score == 0.0


def test_simulator_fuel_depleted():
    config = get_workable_config()
    config.mass_fuel_initial = 1.0
    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry)
    result = LanderSimulator(config, strategy).run()

    assert result.status == LandingStatus.FUEL_DEPLETED
    assert result.final_state.fuel_mass <= 1e-6
    assert result.score.overall_score == 0.0


def test_simulator_respects_hover_limit():
    config = get_workable_config()
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

    result = LanderSimulator(config, HoverStrategy(config.max_thrust, config.gravity, config.mass_dry)).run()
    assert result.status == LandingStatus.HOVER_TOO_LONG
    assert result.score.overall_score == 0.0


def test_simulator_simulation_timeout():
    config = get_workable_config()
    config.initial_altitude = 10000.0
    config.initial_velocity = -1.0
    config.max_simulation_time = 1.0
    config.max_thrust = 100000.0
    config.hover_time_limit = 100.0

    strategy = create_strategy("constant_decel", config.max_thrust, config.gravity, config.mass_dry, target_decel=0.1)
    result = LanderSimulator(config, strategy).run()

    assert result.status == LandingStatus.SIMULATION_TIMEOUT
    assert result.score.overall_score == 0.0


def test_simulator_over_thrust_request():
    config = get_workable_config()
    config.max_thrust = 5000.0
    config.mass_fuel_initial = 5.0
    config.initial_altitude = 50.0

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

    result = LanderSimulator(config, HighThrustStrategy(config.max_thrust, config.gravity, config.mass_dry)).run()
    assert result.status == LandingStatus.OVER_THRUST_REQUEST
    assert result.score.overall_score == 0.0


def test_simulator_premature_touchdown():
    config = get_workable_config()
    config.hard_landing_velocity = -5.0

    class LateBrakeStrategy:
        name = "late_brake"

        def __init__(self, max_thrust, gravity, mass_dry, **kwargs):
            self.max_thrust = max_thrust
            self.gravity = gravity
            self.mass_dry = mass_dry

        def compute_thrust(self, altitude, velocity, mass, time):
            if altitude > 10:
                return 0.0
            return self.max_thrust

        def reset(self):
            pass

    result = LanderSimulator(config, LateBrakeStrategy(config.max_thrust, config.gravity, config.mass_dry)).run()
    assert result.status == LandingStatus.PREMATURE_TOUCHDOWN
    assert result.score.overall_score == 0.0


def test_three_strategies_produce_different_scores():
    config = get_workable_config()
    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = result.score.overall_score

    assert all(s > 0 for s in scores.values()), f"All strategies must score > 0, got {scores}"
    assert len(set(round(s, 1) for s in scores.values())) > 1, f"Strategies should produce different scores, got {scores}"
