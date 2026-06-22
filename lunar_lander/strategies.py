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

    def _target_velocity(self, altitude: float) -> float:
        if altitude <= 0:
            return 0.0
        if altitude > 100:
            v_max = -np.sqrt(2.0 * 3.0 * altitude)
            return float(max(v_max, -30.0))
        elif altitude > 10:
            return float(-min(altitude * 0.15, 3.0))
        elif altitude > 1:
            return -1.0
        else:
            return -0.5


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
        if altitude <= 0:
            return 0.0

        target_v = self._target_velocity(altitude)

        if velocity <= target_v:
            needed_decel = 0.3
            if velocity < 0 and altitude > 0:
                v_sq = velocity * velocity
                needed_decel = v_sq / (2.0 * altitude) + 0.3
            decel = max(self.target_decel, needed_decel)
            thrust = mass * (self.gravity + decel)
        else:
            accel_needed = (velocity - target_v) * 2.0
            thrust = mass * (self.gravity - accel_needed)

        return float(np.clip(thrust, 0.0, self.max_thrust))


class StagedBrakingStrategy(ControlStrategy):
    name = "staged_braking"

    def __init__(
        self,
        max_thrust: float,
        gravity: float,
        mass_dry: float,
        high_altitude_thrust_pct: float = 0.3,
        mid_altitude_thrust_pct: float = 0.5,
        low_altitude_thrust_pct: float = 0.7,
        high_mid_threshold: float = 1000.0,
        mid_low_threshold: float = 200.0,
        final_burn_altitude: float = 50.0,
        final_burn_thrust_pct: float = 0.95,
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
        if altitude <= 0:
            return 0.0

        target_v = self._target_velocity(altitude)

        if altitude > self.high_mid_threshold:
            pct = self.high_altitude_thrust_pct
        elif altitude > self.mid_low_threshold:
            pct = self.mid_altitude_thrust_pct
        elif altitude > self.final_burn_altitude:
            pct = self.low_altitude_thrust_pct
        else:
            pct = self.final_burn_thrust_pct

        if velocity < target_v:
            if velocity < 0 and altitude > 0:
                v_sq = velocity * velocity
                needed = v_sq / (2.0 * altitude) + 0.2
                urgency = min(1.0, needed / 5.0)
                pct = pct + urgency * (1.0 - pct)
            thrust = pct * self.max_thrust
        else:
            excess = velocity - target_v
            descent_accel = min(excess * 1.5, 2.0)
            thrust = mass * (self.gravity - descent_accel)

        return float(np.clip(thrust, 0.0, self.max_thrust))


class PIDStrategy(ControlStrategy):
    name = "pid"

    def __init__(
        self,
        max_thrust: float,
        gravity: float,
        mass_dry: float,
        kp: float = 1.5,
        ki: float = 0.05,
        kd: float = 0.3,
        target_velocity: float = -2.0,
        integral_limit: float = 100.0,
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
        if altitude <= 0:
            return 0.0

        target_vel = self._target_velocity(altitude)
        error = target_vel - velocity

        self._integral += error
        self._integral = float(np.clip(self._integral, -self.integral_limit, self.integral_limit))

        if self._prev_error is not None:
            derivative = error - self._prev_error
        else:
            derivative = 0.0
        self._prev_error = error

        accel_cmd = self.kp * error + self.ki * self._integral + self.kd * derivative

        thrust = mass * (self.gravity + accel_cmd)

        return float(np.clip(thrust, 0.0, self.max_thrust))


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
