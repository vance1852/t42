import numpy as np


class Sensor:
    def __init__(
        self,
        altimeter_noise_std: float,
        velocimeter_noise_std: float,
        filter_alpha: float,
        random_seed: int = 42,
    ):
        self.altimeter_noise_std = altimeter_noise_std
        self.velocimeter_noise_std = velocimeter_noise_std
        self.filter_alpha = filter_alpha
        self.rng = np.random.RandomState(random_seed)
        self._filtered_altitude = None
        self._filtered_velocity = None

    def reset(self, initial_altitude: float, initial_velocity: float):
        self._filtered_altitude = initial_altitude
        self._filtered_velocity = initial_velocity

    def measure(self, true_altitude: float, true_velocity: float) -> tuple:
        noisy_alt = true_altitude + self.rng.normal(0, self.altimeter_noise_std)
        noisy_vel = true_velocity + self.rng.normal(0, self.velocimeter_noise_std)

        if self._filtered_altitude is None:
            self._filtered_altitude = noisy_alt
        else:
            alpha = self.filter_alpha
            self._filtered_altitude = alpha * noisy_alt + (1 - alpha) * self._filtered_altitude

        if self._filtered_velocity is None:
            self._filtered_velocity = noisy_vel
        else:
            alpha = self.filter_alpha
            self._filtered_velocity = alpha * noisy_vel + (1 - alpha) * self._filtered_velocity

        return noisy_alt, noisy_vel, self._filtered_altitude, self._filtered_velocity

    @property
    def filtered_altitude(self) -> float:
        return self._filtered_altitude if self._filtered_altitude is not None else 0.0

    @property
    def filtered_velocity(self) -> float:
        return self._filtered_velocity if self._filtered_velocity is not None else 0.0
