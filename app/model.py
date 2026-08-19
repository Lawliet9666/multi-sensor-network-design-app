"""Numerical model for the finite-grid continuous-time clarity lower bound."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt

import numpy as np


@dataclass(frozen=True)
class SpatialSpectrum:
    """Eigenvalues of ``K_gg / N_g`` for a square endpoint-inclusive grid."""

    eigenvalues: np.ndarray
    grid_points: int
    spacing: float


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return value


def compute_spatial_spectrum(
    domain_size: float,
    spacing: float,
    sigma_s: float,
    length_scale_s: float,
) -> SpatialSpectrum:
    """Return the finite-grid spatial spectrum used by the paper's bound.

    The current application is intentionally restricted to a square domain and
    to grid spacings that divide its side length exactly.
    """

    domain_size = _positive("Domain side length", domain_size)
    spacing = _positive("Grid spacing", spacing)
    sigma_s = _positive("Spatial kernel standard deviation", sigma_s)
    length_scale_s = _positive("Spatial length scale", length_scale_s)

    intervals_float = domain_size / spacing
    intervals = round(intervals_float)
    if intervals < 1 or not isclose(intervals_float, intervals, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("Grid spacing must divide the square-domain side length exactly.")

    axis = np.linspace(0.0, domain_size, intervals + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid_points = grid.shape[0]

    differences = grid[:, None, :] - grid[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    kernel = sigma_s**2 * np.exp(-distances / length_scale_s)

    eigenvalues = np.linalg.eigvalsh(kernel / grid_points)[::-1]
    numerical_tolerance = 1e-10 * max(1.0, float(eigenvalues[0]))
    if float(eigenvalues[-1]) < -numerical_tolerance:
        raise FloatingPointError("The spatial kernel produced a negative eigenvalue.")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("The spatial spectrum contains a non-finite value.")

    return SpatialSpectrum(eigenvalues=eigenvalues, grid_points=grid_points, spacing=spacing)


def sensing_parameter(sensor_count: int | float, sigma_c_squared: float) -> float:
    """Return ``theta = N_r / sigma_c^2`` for the revised measurement model."""

    if isinstance(sensor_count, bool) or int(sensor_count) != sensor_count or sensor_count < 1:
        raise ValueError("Number of sensors must be a positive integer.")
    sigma_c_squared = _positive("Continuous-time noise intensity", sigma_c_squared)
    return int(sensor_count) / sigma_c_squared


def discrete_measurement_noise(
    sigma_c_squared: float, sampling_interval: float
) -> tuple[float, float]:
    """Return discrete measurement variance and standard deviation."""

    sigma_c_squared = _positive("Continuous-time noise intensity", sigma_c_squared)
    sampling_interval = _positive("Sampling interval", sampling_interval)
    variance = sigma_c_squared / sampling_interval
    return variance, sqrt(variance)


def continuous_measurement_noise_intensity(
    sigma_m_squared: float, sampling_interval: float
) -> float:
    """Return ``sigma_c^2 = sigma_m^2 * Delta t``."""

    sigma_m_squared = _positive("Measurement noise variance", sigma_m_squared)
    sampling_interval = _positive("Sampling interval", sampling_interval)
    return sigma_m_squared * sampling_interval


def steady_state_clarity_lower_bound(
    sensor_count: int,
    sigma_c_squared: float,
    temporal_length_scale: float,
    temporal_sigma: float,
    spatial_eigenvalues: np.ndarray,
) -> float:
    """Compute the finite-grid continuous-time steady-state clarity lower bound."""

    theta = sensing_parameter(sensor_count, sigma_c_squared)
    temporal_length_scale = _positive("Temporal length scale", temporal_length_scale)
    temporal_sigma = _positive("Temporal kernel standard deviation", temporal_sigma)

    eigenvalues = np.asarray(spatial_eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("Spatial eigenvalues must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0.0):
        raise ValueError("Spatial eigenvalues must be finite and nonnegative.")

    a = -1.0 / temporal_length_scale
    q_c = 1.0
    c0_squared = temporal_sigma**2 * (2.0 / temporal_length_scale)
    information_eigenvalues = theta * c0_squared * eigenvalues
    gamma = -q_c / (a - np.sqrt(a * a + q_c * information_eigenvalues))
    clarity = 1.0 / (1.0 + c0_squared * float(np.dot(eigenvalues, gamma)))

    if not np.isfinite(clarity) or not 0.0 < clarity < 1.0:
        raise FloatingPointError("The clarity lower bound is outside its valid range.")
    return clarity


def minimum_sensor_count(
    target_clarity: float,
    maximum_sensors: int,
    sigma_c_squared: float,
    temporal_length_scale: float,
    temporal_sigma: float,
    spatial_eigenvalues: np.ndarray,
) -> tuple[int, float] | None:
    """Return the first sensor count whose lower bound meets the target."""

    target_clarity = float(target_clarity)
    if not np.isfinite(target_clarity) or not 0.0 < target_clarity < 1.0:
        raise ValueError("Target clarity must lie strictly between zero and one.")
    if isinstance(maximum_sensors, bool) or int(maximum_sensors) != maximum_sensors:
        raise ValueError("Maximum sensors must be a positive integer.")
    maximum_sensors = int(maximum_sensors)
    if maximum_sensors < 1:
        raise ValueError("Maximum sensors must be a positive integer.")

    for sensor_count in range(1, maximum_sensors + 1):
        clarity = steady_state_clarity_lower_bound(
            sensor_count,
            sigma_c_squared,
            temporal_length_scale,
            temporal_sigma,
            spatial_eigenvalues,
        )
        if clarity >= target_clarity:
            return sensor_count, clarity
    return None
