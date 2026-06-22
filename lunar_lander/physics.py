import numpy as np
from typing import Tuple


class PhysicsEngine:
    def __init__(self, gravity: float, specific_impulse: float = 300.0, g0: float = 9.81):
        self.gravity = gravity
        self.specific_impulse = specific_impulse
        self.g0 = g0

    def step(
        self,
        altitude: float,
        velocity: float,
        mass: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        acceleration = thrust / mass - self.gravity
        new_velocity = velocity + acceleration * dt
        new_altitude = altitude + velocity * dt + 0.5 * acceleration * dt * dt
        fuel_consumed = thrust * dt / (self.specific_impulse * self.g0)
        new_mass = mass - fuel_consumed
        return new_altitude, new_velocity, new_mass

    def fuel_consumption_rate(self, thrust: float) -> float:
        return thrust / (self.specific_impulse * self.g0)
