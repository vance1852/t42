import itertools
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from .config import LanderConfig
from .simulator import LanderSimulator, SimulationResult
from .strategies import create_strategy


def parameter_scan(
    base_config: LanderConfig,
    strategy_name: str,
    param_ranges: Dict[str, List[float]],
    strategy_params: Dict[str, Any] = None,
) -> List[Tuple[Dict[str, float], SimulationResult]]:
    if strategy_params is None:
        strategy_params = {}

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    results = []

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        modified_params = {**strategy_params, **params}

        config = LanderConfig(**{
            **base_config.__dict__,
        })

        strategy = create_strategy(
            strategy_name=strategy_name,
            max_thrust=config.max_thrust,
            gravity=config.gravity,
            mass_dry=config.mass_dry,
            **modified_params,
        )

        simulator = LanderSimulator(config, strategy)
        result = simulator.run()

        results.append((params, result))

    return results


@dataclass
class BestStrategyResult:
    strategy_name: str
    params: Dict[str, Any]
    result: SimulationResult
    improved_over_baseline: bool
    baseline_strategy: str
    baseline_score: float
    best_score: float


def find_best_strategy(
    base_config: LanderConfig,
    strategy_names: List[str],
    strategy_param_ranges: Dict[str, Dict[str, List[float]]] = None,
    baseline_strategy: str = "constant_decel",
    baseline_params: Dict[str, Any] = None,
) -> BestStrategyResult:
    if strategy_param_ranges is None:
        strategy_param_ranges = {}
    if baseline_params is None:
        baseline_params = {}

    baseline_strat = create_strategy(
        strategy_name=baseline_strategy,
        max_thrust=base_config.max_thrust,
        gravity=base_config.gravity,
        mass_dry=base_config.mass_dry,
        **baseline_params,
    )
    baseline_sim = LanderSimulator(base_config, baseline_strat)
    baseline_result = baseline_sim.run()
    baseline_score = baseline_result.score.overall_score

    best_score = baseline_score
    best_strategy = baseline_strategy
    best_params = baseline_params
    best_result = baseline_result

    for s_name in strategy_names:
        param_ranges = strategy_param_ranges.get(s_name, {})

        if not param_ranges:
            strat = create_strategy(
                strategy_name=s_name,
                max_thrust=base_config.max_thrust,
                gravity=base_config.gravity,
                mass_dry=base_config.mass_dry,
            )
            sim = LanderSimulator(base_config, strat)
            result = sim.run()

            if result.score.overall_score > best_score:
                best_score = result.score.overall_score
                best_strategy = s_name
                best_params = {}
                best_result = result
        else:
            scan_results = parameter_scan(
                base_config=base_config,
                strategy_name=s_name,
                param_ranges=param_ranges,
            )

            for params, result in scan_results:
                if result.score.overall_score > best_score:
                    best_score = result.score.overall_score
                    best_strategy = s_name
                    best_params = params
                    best_result = result

    improved = bool(best_score > baseline_score)

    return BestStrategyResult(
        strategy_name=best_strategy,
        params=best_params,
        result=best_result,
        improved_over_baseline=improved,
        baseline_strategy=baseline_strategy,
        baseline_score=baseline_score,
        best_score=best_score,
    )
