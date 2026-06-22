from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any
import numpy as np


class LandingStatus(Enum):
    SOFT_LANDING = "soft_landing"
    HARD_LANDING = "hard_landing"
    FUEL_DEPLETED = "fuel_depleted"
    OVER_THRUST_REQUEST = "over_thrust_request"
    HOVER_TOO_LONG = "hover_too_long"
    PREMATURE_TOUCHDOWN = "premature_touchdown"
    SIMULATION_TIMEOUT = "simulation_timeout"

    @property
    def is_success(self) -> bool:
        return self == LandingStatus.SOFT_LANDING

    @property
    def description(self) -> str:
        descriptions = {
            LandingStatus.SOFT_LANDING: "成功软着陆",
            LandingStatus.HARD_LANDING: "硬着陆（撞击速度过大）",
            LandingStatus.FUEL_DEPLETED: "燃料耗尽",
            LandingStatus.OVER_THRUST_REQUEST: "超推力请求",
            LandingStatus.HOVER_TOO_LONG: "悬停时间过长",
            LandingStatus.PREMATURE_TOUCHDOWN: "提前触地",
            LandingStatus.SIMULATION_TIMEOUT: "模拟超时",
        }
        return descriptions.get(self, str(self))


@dataclass
class ScoreResult:
    overall_score: float
    fuel_efficiency: float
    landing_accuracy: float
    landing_smoothness: float
    time_efficiency: float
    status: LandingStatus


class Scorer:
    def __init__(
        self,
        soft_landing_velocity: float = -2.0,
        max_fuel_mass: float = 500.0,
        target_altitude: float = 0.0,
    ):
        self.soft_landing_velocity = soft_landing_velocity
        self.max_fuel_mass = max_fuel_mass
        self.target_altitude = target_altitude

    def compute_score(
        self,
        status: LandingStatus,
        final_altitude: float,
        final_velocity: float,
        fuel_used: float,
        flight_time: float,
        max_thrust_command: float,
        max_available_thrust: float,
    ) -> ScoreResult:
        if not status.is_success:
            return ScoreResult(
                overall_score=0.0,
                fuel_efficiency=0.0,
                landing_accuracy=0.0,
                landing_smoothness=0.0,
                time_efficiency=0.0,
                status=status,
            )

        fuel_efficiency = max(0.0, 1.0 - fuel_used / self.max_fuel_mass) * 100.0

        alt_error = abs(final_altitude - self.target_altitude)
        landing_accuracy = max(0.0, 1.0 - alt_error / 10.0) * 100.0

        vel_magnitude = abs(final_velocity)
        soft_limit = abs(self.soft_landing_velocity)
        landing_smoothness = max(0.0, 1.0 - vel_magnitude / soft_limit) * 100.0

        time_efficiency = max(0.0, 1.0 - flight_time / 300.0) * 100.0

        overall = (
            0.3 * fuel_efficiency
            + 0.25 * landing_accuracy
            + 0.3 * landing_smoothness
            + 0.15 * time_efficiency
        )

        return ScoreResult(
            overall_score=overall,
            fuel_efficiency=fuel_efficiency,
            landing_accuracy=landing_accuracy,
            landing_smoothness=landing_smoothness,
            time_efficiency=time_efficiency,
            status=status,
        )
