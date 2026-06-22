from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ControlStrategy(ABC):
    name = "base"

    def __init__(self, max_thrust: float, gravity: float, mass_dry: float, **kwargs):
        self.max_thrust = max_thrust
        self.gravity = gravity
        self.mass_dry = mass_dry

    @abstractmethod
    def compute_thrust(
        self,
        altitude: float,
        velocity: float,
        mass: float,
        time: float,
    ) -> float:
        pass

    def reset(self):
        pass


class ConstantDecelerationStrategy(ControlStrategy):
    name = "constant_decel"

    def __init__(self, max_thrust: float, gravity: float, mass_dry: float, target_decel: float = 2.0, **kwargs):
        super().__init__(max_thrust, gravity, mass_dry, **kwargs)
        self.target_decel = target_decel

    def compute_thrust(
        self,
        altitude: float,
        velocity: float,
        mass: float,
        time: float,
    ) -> float:
        if velocity >= 0:
            return 0.0

        required_thrust = mass * (self.gravity + self.target_decel)
        return np.clip(required_thrust, 0.0, self.max_thrust)


class StagedBrakingStrategy(ControlStrategy):
    name = "staged_braking"

    def __init__(
        self,
        max_thrust: float,
        gravity: float,
        mass_dry: float,
        high_altitude_thrust_pct: float = 0.8,
        mid_altitude_thrust_pct: float = 0.6,
        low_altitude_thrust_pct: float = 0.4,
        high_mid_threshold: float = 1000.0,
        mid_low_threshold: float = 200.0,
        final_burn_altitude: float = 50.0,
        final_burn_thrust_pct: float = 0.9,
        **kwargs,
    ):
        super().__init__(max_thrust, gravity, mass_dry, **kwargs)
        self.high_altitude_thrust_pct = high_altitude_thrust_pct
        self.mid_altitude_thrust_pct = mid_altitude_thrust_pct
        self.low_altitude_thrust_pct = low_altitude_thrust_pct
        self.high_mid_threshold = high_mid_threshold
        self.mid_low_threshold = mid_low_threshold
        self.final_burn_altitude = final_burn_altitude
        self.final_burn_thrust_pct = final_burn_thrust_pct

    def compute_thrust(
        self,
        altitude: float,
        velocity: float,
        mass: float,
        time: float,
    ) -> float:
        if velocity >= 0:
            return 0.0

        if altitude > self.high_mid_threshold:
            thrust_pct = self.high_altitude_thrust_pct
        elif altitude > self.mid_low_threshold:
            thrust_pct = self.mid_altitude_thrust_pct
        elif altitude > self.final_burn_altitude:
            thrust_pct = self.low_altitude_thrust_pct
        else:
            thrust_pct = self.final_burn_thrust_pct

        return thrust_pct * self.max_thrust


class PIDStrategy(ControlStrategy):
    name = "pid"

    def __init__(
        self,
        max_thrust: float,
        gravity: float,
        mass_dry: float,
        kp: float = 0.5,
        ki: float = 0.001,
        kd: float = 2.0,
        target_velocity: float = -1.0,
        integral_limit: float = 10000.0,
        **kwargs,
    ):
        super().__init__(max_thrust, gravity, mass_dry, **kwargs)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_velocity = target_velocity
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def compute_thrust(
        self,
        altitude: float,
        velocity: float,
        mass: float,
        time: float,
    ) -> float:
        error = self.target_velocity - velocity

        self._integral += error
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)

        if self._prev_error is not None:
            derivative = error - self._prev_error
        else:
            derivative = 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        gravity_compensation = mass * self.gravity
        thrust = gravity_compensation + output

        return np.clip(thrust, 0.0, self.max_thrust)


def create_strategy(strategy_name: str, max_thrust: float, gravity: float, mass_dry: float, **kwargs) -> ControlStrategy:
    strategies = {
        "constant_decel": ConstantDecelerationStrategy,
        "staged_braking": StagedBrakingStrategy,
        "pid": PIDStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

    return strategies[strategy_name](
        max_thrust=max_thrust,
        gravity=gravity,
        mass_dry=mass_dry,
        **kwargs,
    )
