import pytest
from lunar_lander.scoring import LandingStatus, Scorer, ScoreResult


def test_landing_status_success():
    assert LandingStatus.SOFT_LANDING.is_success is True
    assert LandingStatus.HARD_LANDING.is_success is False
    assert LandingStatus.FUEL_DEPLETED.is_success is False


def test_landing_status_descriptions():
    for status in LandingStatus:
        assert isinstance(status.description, str)
        assert len(status.description) > 0


def test_score_success_full_marks():
    scorer = Scorer(soft_landing_velocity=-2.0, max_fuel_mass=500.0, target_altitude=0.0)
    result = scorer.compute_score(
        status=LandingStatus.SOFT_LANDING,
        final_altitude=0.0,
        final_velocity=0.0,
        fuel_used=0.0,
        flight_time=0.0,
        max_thrust_command=1000.0,
        max_available_thrust=30000.0,
    )

    assert result.overall_score > 0
    assert result.fuel_efficiency == 100.0
    assert result.landing_accuracy == 100.0
    assert result.landing_smoothness == 100.0
    assert result.status == LandingStatus.SOFT_LANDING


def test_score_failure_zero_score():
    scorer = Scorer()
    result = scorer.compute_score(
        status=LandingStatus.HARD_LANDING,
        final_altitude=0.0,
        final_velocity=-20.0,
        fuel_used=100.0,
        flight_time=50.0,
        max_thrust_command=1000.0,
        max_available_thrust=30000.0,
    )

    assert result.overall_score == 0.0
    assert result.status == LandingStatus.HARD_LANDING


def test_score_fuel_efficiency_decreases():
    scorer = Scorer(max_fuel_mass=500.0)

    r1 = scorer.compute_score(
        status=LandingStatus.SOFT_LANDING,
        final_altitude=0.0,
        final_velocity=-1.0,
        fuel_used=100.0,
        flight_time=50.0,
        max_thrust_command=1000.0,
        max_available_thrust=30000.0,
    )

    r2 = scorer.compute_score(
        status=LandingStatus.SOFT_LANDING,
        final_altitude=0.0,
        final_velocity=-1.0,
        fuel_used=400.0,
        flight_time=50.0,
        max_thrust_command=1000.0,
        max_available_thrust=30000.0,
    )

    assert r1.fuel_efficiency > r2.fuel_efficiency


def test_score_landing_smoothness():
    scorer = Scorer(soft_landing_velocity=-2.0)

    r_smooth = scorer.compute_score(
        status=LandingStatus.SOFT_LANDING,
        final_altitude=0.0,
        final_velocity=-0.5,
        fuel_used=100.0,
        flight_time=50.0,
        max_thrust_command=1000.0,
        max_available_thrust=30000.0,
    )

    r_rough = scorer.compute_score(
        status=LandingStatus.SOFT_LANDING,
        final_altitude=0.0,
        final_velocity=-1.9,
        fuel_used=100.0,
        flight_time=50.0,
        max_thrust_command=1000.0,
        max_available_thrust=30000.0,
    )

    assert r_smooth.landing_smoothness > r_rough.landing_smoothness


def test_all_failure_statuses_score_zero():
    scorer = Scorer()
    failures = [
        LandingStatus.HARD_LANDING,
        LandingStatus.FUEL_DEPLETED,
        LandingStatus.OVER_THRUST_REQUEST,
        LandingStatus.HOVER_TOO_LONG,
        LandingStatus.PREMATURE_TOUCHDOWN,
        LandingStatus.SIMULATION_TIMEOUT,
    ]

    for status in failures:
        result = scorer.compute_score(
            status=status,
            final_altitude=0.0,
            final_velocity=-5.0,
            fuel_used=100.0,
            flight_time=50.0,
            max_thrust_command=1000.0,
            max_available_thrust=30000.0,
        )
        assert result.overall_score == 0.0, f"{status} should have zero score"
