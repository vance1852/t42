import numpy as np
from collections import deque
from typing import Deque, Tuple


class ThrustController:
    def __init__(
        self,
        max_thrust: float,
        response_delay: float,
        max_change_rate: float,
        dt: float,
    ):
        self.max_thrust = max_thrust
        self.response_delay = response_delay
        self.max_change_rate = max_change_rate
        self.dt = dt
        self.current_thrust = 0.0
        self._delay_steps = int(round(response_delay / dt))
        self._command_queue: Deque[float] = deque()
        self._last_commanded = 0.0
        self.reset()

    def reset(self):
        self.current_thrust = 0.0
        self._command_queue.clear()
        for _ in range(self._delay_steps):
            self._command_queue.append(0.0)
        self._last_commanded = 0.0

    def set_command(self, thrust_command: float):
        clamped = np.clip(thrust_command, 0.0, self.max_thrust)
        self._command_queue.append(clamped)
        self._last_commanded = clamped

    def step(self) -> float:
        if self._command_queue:
            target = self._command_queue.popleft()
        else:
            target = self._last_commanded

        max_delta = self.max_change_rate * self.dt
        delta = target - self.current_thrust
        delta = np.clip(delta, -max_delta, max_delta)
        self.current_thrust += delta
        self.current_thrust = np.clip(self.current_thrust, 0.0, self.max_thrust)
        return self.current_thrust

    @property
    def last_commanded(self) -> float:
        return self._last_commanded
