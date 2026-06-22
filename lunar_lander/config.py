from dataclasses import dataclass, field
from typing import Dict, Any
import json
import os


@dataclass
class LanderConfig:
    mass_dry: float = 1000.0
    mass_fuel_initial: float = 500.0
    max_thrust: float = 30000.0
    thrust_response_delay: float = 0.5
    thrust_max_change_rate: float = 15000.0
    gravity: float = 1.62
    initial_altitude: float = 2000.0
    initial_velocity: float = -50.0
    dt: float = 0.01
    altimeter_noise_std: float = 1.0
    velocimeter_noise_std: float = 0.5
    filter_alpha: float = 0.1
    max_simulation_time: float = 300.0
    hover_time_limit: float = 30.0
    soft_landing_velocity: float = -2.0
    hard_landing_velocity: float = -10.0
    target_altitude: float = 0.0
    random_seed: int = 42

    @property
    def mass_initial(self) -> float:
        return self.mass_dry + self.mass_fuel_initial


@dataclass
class Scenario:
    name: str
    description: str
    config: LanderConfig
    strategy_params: Dict[str, Any] = field(default_factory=dict)


def load_scenario(filepath: str) -> Scenario:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_data = data.get("config", {})
    config = LanderConfig(**config_data)

    return Scenario(
        name=data.get("name", os.path.basename(filepath)),
        description=data.get("description", ""),
        config=config,
        strategy_params=data.get("strategy_params", {}),
    )


def get_builtin_scenarios_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "scenarios")
