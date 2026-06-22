import pytest
import numpy as np
from lunar_lander.sensors import Sensor


def test_initialization():
    sensor = Sensor(altimeter_noise_std=1.0, velocimeter_noise_std=0.5, filter_alpha=0.1, random_seed=42)
    sensor.reset(1000.0, -50.0)
    assert abs(sensor.filtered_altitude - 1000.0) < 1e-9
    assert abs(sensor.filtered_velocity - (-50.0)) < 1e-9


def test_noise_added_to_measurements():
    sensor = Sensor(altimeter_noise_std=0.0, velocimeter_noise_std=0.0, filter_alpha=1.0, random_seed=42)
    sensor.reset(100.0, 10.0)

    noisy_alt, noisy_vel, filt_alt, filt_vel = sensor.measure(100.0, 10.0)
    assert abs(noisy_alt - 100.0) < 1e-9
    assert abs(noisy_vel - 10.0) < 1e-9


def test_filter_smooths_noise():
    np.random.seed(42)
    sensor = Sensor(altimeter_noise_std=10.0, velocimeter_noise_std=5.0, filter_alpha=0.05, random_seed=42)
    sensor.reset(1000.0, -50.0)

    true_alt = 1000.0
    true_vel = -50.0

    noisy_vals = []
    filtered_vals = []

    for _ in range(100):
        noisy_alt, noisy_vel, filt_alt, filt_vel = sensor.measure(true_alt, true_vel)
        noisy_vals.append(noisy_alt)
        filtered_vals.append(filt_alt)

    noisy_std = np.std(noisy_vals)
    filtered_std = np.std(filtered_vals)

    assert filtered_std < noisy_std


def test_filter_alpha_one_no_filter():
    sensor = Sensor(altimeter_noise_std=1.0, velocimeter_noise_std=0.5, filter_alpha=1.0, random_seed=42)
    sensor.reset(100.0, 10.0)

    noisy_alt, noisy_vel, filt_alt, filt_vel = sensor.measure(200.0, 20.0)
    assert abs(filt_alt - noisy_alt) < 1e-9
    assert abs(filt_vel - noisy_vel) < 1e-9


def test_filter_alpha_zero_no_update():
    sensor = Sensor(altimeter_noise_std=1.0, velocimeter_noise_std=0.5, filter_alpha=0.0, random_seed=42)
    sensor.reset(100.0, 10.0)

    _, _, filt_alt, filt_vel = sensor.measure(500.0, 100.0)
    assert abs(filt_alt - 100.0) < 1e-9
    assert abs(filt_vel - 10.0) < 1e-9


def test_random_seed_reproducibility():
    sensor1 = Sensor(altimeter_noise_std=5.0, velocimeter_noise_std=3.0, filter_alpha=0.1, random_seed=123)
    sensor1.reset(100.0, 10.0)

    sensor2 = Sensor(altimeter_noise_std=5.0, velocimeter_noise_std=3.0, filter_alpha=0.1, random_seed=123)
    sensor2.reset(100.0, 10.0)

    for _ in range(50):
        m1 = sensor1.measure(200.0, 20.0)
        m2 = sensor2.measure(200.0, 20.0)
        assert m1 == m2


def test_filter_gradually_converges():
    sensor = Sensor(altimeter_noise_std=0.0, velocimeter_noise_std=0.0, filter_alpha=0.2, random_seed=42)
    sensor.reset(0.0, 0.0)

    target_alt = 100.0
    target_vel = 50.0

    for _ in range(50):
        _, _, filt_alt, filt_vel = sensor.measure(target_alt, target_vel)

    assert abs(filt_alt - target_alt) < 1.0
    assert abs(filt_vel - target_vel) < 1.0
