import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .config import LanderConfig
from .physics import PhysicsEngine
from .thrust import ThrustController
from .sensors import Sensor
from .strategies import ControlStrategy
from .scoring import LandingStatus, ScoreResult, Scorer


@dataclass
class SimulationState:
    time: float = 0.0
    altitude: float = 0.0
    velocity: float = 0.0
    mass: float = 0.0
    fuel_mass: float = 0.0
    thrust: float = 0.0
    thrust_command: float = 0.0
    filtered_altitude: float = 0.0
    filtered_velocity: float = 0.0
    measured_altitude: float = 0.0
    measured_velocity: float = 0.0


@dataclass
class SimulationResult:
    status: LandingStatus
    states: List[SimulationState]
    score: ScoreResult
    config: LanderConfig
    strategy_name: str

    @property
    def final_state(self) -> SimulationState:
        return self.states[-1]

    @property
    def flight_time(self) -> float:
        return self.states[-1].time - self.states[0].time

    @property
    def fuel_used(self) -> float:
        return self.states[0].fuel_mass - self.states[-1].fuel_mass


class LanderSimulator:
    def __init__(self, config: LanderConfig, strategy: ControlStrategy):
        self.config = config
        self.strategy = strategy
        self.physics = PhysicsEngine(gravity=config.gravity)
        self.thrust_ctrl = ThrustController(
            max_thrust=config.max_thrust,
            response_delay=config.thrust_response_delay,
            max_change_rate=config.thrust_max_change_rate,
            dt=config.dt,
        )
        self.sensor = Sensor(
            altimeter_noise_std=config.altimeter_noise_std,
            velocimeter_noise_std=config.velocimeter_noise_std,
            filter_alpha=config.filter_alpha,
            random_seed=config.random_seed,
        )
        self.scorer = Scorer(
            soft_landing_velocity=config.soft_landing_velocity,
            max_fuel_mass=config.mass_fuel_initial,
            target_altitude=config.target_altitude,
        )

    def reset(self):
        self.strategy.reset()
        self.thrust_ctrl.reset()
        self.sensor.reset(
            self.config.initial_altitude,
            self.config.initial_velocity,
        )

    def run(self) -> SimulationResult:
        self.reset()

        config = self.config
        states = []
        max_thrust_command = 0.0
        over_thrust_requested = False

        altitude = config.initial_altitude
        velocity = config.initial_velocity
        mass = config.mass_initial
        fuel_mass = config.mass_fuel_initial
        time = 0.0
        hover_time = 0.0

        self.sensor.reset(altitude, velocity)
        measured_alt, measured_vel, filtered_alt, filtered_vel = self.sensor.measure(altitude, velocity)

        states.append(
            SimulationState(
                time=time,
                altitude=altitude,
                velocity=velocity,
                mass=mass,
                fuel_mass=fuel_mass,
                thrust=0.0,
                thrust_command=0.0,
                filtered_altitude=filtered_alt,
                filtered_velocity=filtered_vel,
                measured_altitude=measured_alt,
                measured_velocity=measured_vel,
            )
        )

        status = None

        while time < config.max_simulation_time:
            thrust_command = self.strategy.compute_thrust(
                altitude=filtered_alt,
                velocity=filtered_vel,
                mass=mass,
                time=time,
            )

            max_thrust_command = max(max_thrust_command, thrust_command)
            if thrust_command > config.max_thrust + 1e-9:
                over_thrust_requested = True

            self.thrust_ctrl.set_command(thrust_command)
            actual_thrust = self.thrust_ctrl.step()

            new_altitude, new_velocity, new_mass = self.physics.step(
                altitude=altitude,
                velocity=velocity,
                mass=mass,
                thrust=actual_thrust,
                dt=config.dt,
            )

            fuel_consumed = mass - new_mass
            fuel_mass = max(0.0, fuel_mass - fuel_consumed)

            time += config.dt

            measured_alt, measured_vel, filtered_alt, filtered_vel = self.sensor.measure(
                new_altitude, new_velocity
            )

            states.append(
                SimulationState(
                    time=time,
                    altitude=new_altitude,
                    velocity=new_velocity,
                    mass=new_mass,
                    fuel_mass=fuel_mass,
                    thrust=actual_thrust,
                    thrust_command=thrust_command,
                    filtered_altitude=filtered_alt,
                    filtered_velocity=filtered_vel,
                    measured_altitude=measured_alt,
                    measured_velocity=measured_vel,
                )
            )

            altitude = new_altitude
            velocity = new_velocity
            mass = new_mass

            if abs(velocity) < 0.5 and altitude > 10:
                hover_time += config.dt
            else:
                hover_time = max(0.0, hover_time - config.dt)

            if fuel_mass <= 0.0 and actual_thrust > 0:
                status = LandingStatus.FUEL_DEPLETED
                break

            if over_thrust_requested and status is None:
                status = LandingStatus.OVER_THRUST_REQUEST

            if altitude <= config.target_altitude:
                if velocity < config.hard_landing_velocity:
                    status = LandingStatus.HARD_LANDING
                else:
                    if status is None:
                        status = LandingStatus.SOFT_LANDING
                break

            if hover_time >= config.hover_time_limit:
                status = LandingStatus.HOVER_TOO_LONG
                break

        if status is None:
            status = LandingStatus.SIMULATION_TIMEOUT

        final_altitude = states[-1].altitude
        final_velocity = states[-1].velocity
        fuel_used = states[0].fuel_mass - states[-1].fuel_mass
        flight_time = states[-1].time - states[0].time

        score = self.scorer.compute_score(
            status=status,
            final_altitude=final_altitude,
            final_velocity=final_velocity,
            fuel_used=fuel_used,
            flight_time=flight_time,
            max_thrust_command=max_thrust_command,
            max_available_thrust=config.max_thrust,
        )

        return SimulationResult(
            status=status,
            states=states,
            score=score,
            config=config,
            strategy_name=self.strategy.name,
        )
