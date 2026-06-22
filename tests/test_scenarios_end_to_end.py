import os
import pytest
from lunar_lander.config import load_scenario
from lunar_lander.simulator import LanderSimulator
from lunar_lander.strategies import create_strategy
from lunar_lander.scoring import LandingStatus
from lunar_lander.parameter_scan import find_best_strategy


SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "..", "scenarios")

SCENARIO_FILES = [
    "fuel_critical.json",
    "high_initial_velocity.json",
    "high_noise.json",
    "engine_lag.json",
]


@pytest.fixture(params=SCENARIO_FILES)
def scenario(request):
    path = os.path.join(SCENARIOS_DIR, request.param)
    return load_scenario(path)


def test_scenario_loads_and_runs(scenario):
    config = scenario.config
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        assert result is not None
        assert result.status is not None
        assert result.score is not None


def test_scenario_at_least_one_strategy_succeeds(scenario):
    config = scenario.config
    any_success = False
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        if result.status == LandingStatus.SOFT_LANDING:
            any_success = True
            break
    assert any_success, f"Scenario '{scenario.name}': at least one strategy must achieve soft landing"


def test_scenario_not_all_zero_scores(scenario):
    config = scenario.config
    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = result.score.overall_score

    max_score = max(scores.values())
    assert max_score > 0, f"Scenario '{scenario.name}': all strategies scored 0, scores={scores}"


def test_scenario_strategies_have_differentiation(scenario):
    config = scenario.config
    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = result.score.overall_score

    nonzero = {k: v for k, v in scores.items() if v > 0}
    assert len(nonzero) >= 1, f"Scenario '{scenario.name}': need at least 1 nonzero score, got {scores}"

    nonzero_vals = list(nonzero.values())
    if len(nonzero_vals) >= 2:
        spread = max(nonzero_vals) - min(nonzero_vals)
        assert spread > 1.0, (
            f"Scenario '{scenario.name}': nonzero scores should show differentiation "
            f"(spread={spread:.2f}), scores={scores}"
        )


def test_scenario_find_best_can_find_improvement(scenario):
    config = scenario.config
    baseline_strategy = "constant_decel"

    param_ranges = {
        "constant_decel": {"target_decel": [1.0, 2.0, 3.0]},
        "pid": {"kp": [1.0, 2.0, 3.0]},
    }

    best = find_best_strategy(
        base_config=config,
        strategy_names=["constant_decel", "pid"],
        strategy_param_ranges=param_ranges,
        baseline_strategy=baseline_strategy,
        baseline_params=scenario.strategy_params.get(baseline_strategy, {}),
    )

    assert best.best_score > 0, f"Scenario '{scenario.name}': best score must be > 0, got {best.best_score}"


def test_fuel_critical_scenario_specific():
    path = os.path.join(SCENARIOS_DIR, "fuel_critical.json")
    scenario = load_scenario(path)
    config = scenario.config

    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = (result.score.overall_score, result.status)

    max_score = max(v[0] for v in scores.values())
    assert max_score > 0, f"fuel_critical: all strategies scored 0, results={scores}"

    successful = [k for k, v in scores.items() if v[1] == LandingStatus.SOFT_LANDING]
    assert len(successful) >= 1, f"fuel_critical: at least one strategy must succeed, got {scores}"


def test_high_initial_velocity_scenario_specific():
    path = os.path.join(SCENARIOS_DIR, "high_initial_velocity.json")
    scenario = load_scenario(path)
    config = scenario.config

    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = (result.score.overall_score, result.status)

    max_score = max(v[0] for v in scores.values())
    assert max_score > 0, f"high_initial_velocity: all strategies scored 0, results={scores}"

    successful = [k for k, v in scores.items() if v[1] == LandingStatus.SOFT_LANDING]
    assert len(successful) >= 1, f"high_initial_velocity: at least one strategy must succeed, got {scores}"


def test_high_noise_scenario_specific():
    path = os.path.join(SCENARIOS_DIR, "high_noise.json")
    scenario = load_scenario(path)
    config = scenario.config

    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = (result.score.overall_score, result.status)

    max_score = max(v[0] for v in scores.values())
    assert max_score > 0, f"high_noise: all strategies scored 0, results={scores}"

    successful = [k for k, v in scores.items() if v[1] == LandingStatus.SOFT_LANDING]
    assert len(successful) >= 1, f"high_noise: at least one strategy must succeed, got {scores}"


def test_engine_lag_scenario_specific():
    path = os.path.join(SCENARIOS_DIR, "engine_lag.json")
    scenario = load_scenario(path)
    config = scenario.config

    scores = {}
    for strategy_name in ["constant_decel", "staged_braking", "pid"]:
        params = scenario.strategy_params.get(strategy_name, {})
        strategy = create_strategy(strategy_name, config.max_thrust, config.gravity, config.mass_dry, **params)
        result = LanderSimulator(config, strategy).run()
        scores[strategy_name] = (result.score.overall_score, result.status)

    max_score = max(v[0] for v in scores.values())
    assert max_score > 0, f"engine_lag: all strategies scored 0, results={scores}"

    successful = [k for k, v in scores.items() if v[1] == LandingStatus.SOFT_LANDING]
    assert len(successful) >= 1, f"engine_lag: at least one strategy must succeed, got {scores}"
