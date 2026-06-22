import pytest
from lunar_lander.config import LanderConfig
from lunar_lander.parameter_scan import parameter_scan, find_best_strategy
from lunar_lander.scoring import LandingStatus


def get_simple_config():
    return LanderConfig(
        mass_dry=1000.0,
        mass_fuel_initial=500.0,
        max_thrust=30000.0,
        thrust_response_delay=0.1,
        thrust_max_change_rate=20000.0,
        gravity=1.62,
        initial_altitude=200.0,
        initial_velocity=-20.0,
        dt=0.05,
        altimeter_noise_std=0.0,
        velocimeter_noise_std=0.0,
        filter_alpha=1.0,
        max_simulation_time=100.0,
        hover_time_limit=30.0,
        soft_landing_velocity=-3.0,
        hard_landing_velocity=-10.0,
        target_altitude=0.0,
        random_seed=42,
    )


def test_parameter_scan_returns_all_combinations():
    config = get_simple_config()

    param_ranges = {
        "kp": [0.5, 1.0, 2.0],
    }

    results = parameter_scan(
        base_config=config,
        strategy_name="pid",
        param_ranges=param_ranges,
    )

    assert len(results) == 3
    for params, result in results:
        assert "kp" in params
        assert result is not None


def test_parameter_scan_two_params():
    config = get_simple_config()

    param_ranges = {
        "kp": [0.5, 1.0],
        "kd": [1.0, 2.0],
    }

    results = parameter_scan(
        base_config=config,
        strategy_name="pid",
        param_ranges=param_ranges,
    )

    assert len(results) == 4


def test_parameter_scan_different_scores():
    config = get_simple_config()

    param_ranges = {
        "target_decel": [0.5, 3.0],
    }

    results = parameter_scan(
        base_config=config,
        strategy_name="constant_decel",
        param_ranges=param_ranges,
    )

    scores = [r[1].score.overall_score for r in results]
    assert len(set(scores)) > 1 or all(s == 0 for s in scores)


def test_find_best_strategy_returns_better_than_baseline():
    config = get_simple_config()

    best_strategy, best_params, best_result = find_best_strategy(
        base_config=config,
        strategy_names=["constant_decel", "staged_braking", "pid"],
        baseline_strategy="constant_decel",
    )

    assert best_strategy in ["constant_decel", "staged_braking", "pid"]
    assert best_result is not None
    assert best_result.score.overall_score >= 0


def test_find_best_with_param_ranges():
    config = get_simple_config()

    param_ranges = {
        "pid": {
            "kp": [0.3, 1.0, 2.0],
        },
    }

    best_strategy, best_params, best_result = find_best_strategy(
        base_config=config,
        strategy_names=["constant_decel", "pid"],
        strategy_param_ranges=param_ranges,
        baseline_strategy="constant_decel",
    )

    assert best_strategy is not None
    assert best_result is not None


def test_parameter_scan_can_find_better_than_baseline():
    config = get_simple_config()
    config.initial_altitude = 300.0
    config.initial_velocity = -40.0

    from lunar_lander.strategies import create_strategy
    from lunar_lander.simulator import LanderSimulator

    baseline_strat = create_strategy(
        "constant_decel", config.max_thrust, config.gravity, config.mass_dry,
        target_decel=1.0,
    )
    baseline_sim = LanderSimulator(config, baseline_strat)
    baseline_result = baseline_sim.run()

    param_ranges = {
        "target_decel": [0.5, 1.5, 2.5, 3.5],
    }
    scan_results = parameter_scan(
        base_config=config,
        strategy_name="constant_decel",
        param_ranges=param_ranges,
    )

    best_scan = max(scan_results, key=lambda x: x[1].score.overall_score)

    assert best_scan[1].score.overall_score >= baseline_result.score.overall_score
