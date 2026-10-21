"""Independent numerical fixtures, never real model accuracy or speed evidence."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from factory_density_math import normalized_log_distance, whitening_from_moments


def test_whitening_matches_independent_regularized_precision_matrix():
    rng = np.random.default_rng(712)
    transform = rng.normal(size=(16, 16))
    covariance = transform @ transform.T + np.eye(16) * .3
    mean = rng.normal(size=(16, 256))
    second = covariance + mean @ mean.T / 256
    whitening, report = whitening_from_moments(mean, second)
    regularized = .9 * covariance + .1 * np.diag(np.diag(covariance))
    measured_precision = whitening.astype(np.float64).T @ whitening
    measured_precision *= report["global_whitening_rescale"] ** 2
    np.testing.assert_allclose(measured_precision, np.linalg.inv(regularized), atol=2e-7, rtol=2e-6)
    assert report["floored_eigenvalues"] == 0


def test_zero_covariance_is_finite_and_whitening_is_bounded():
    whitening, report = whitening_from_moments(np.zeros((16, 256)), np.zeros((16, 16)))
    assert report["floored_eigenvalues"] == 16
    assert report["eigenvalue_floor"] == 1e-6
    assert np.isfinite(whitening).all()
    assert np.abs(whitening).sum(axis=1).max() <= 8


def test_distance_zero_floor_and_large_feature_bounds_are_finite():
    shape = (16, 640, 256)
    zero = normalized_log_distance(np.zeros(shape, np.float32), np.zeros((16, 256)), np.eye(16))
    np.testing.assert_allclose(zero, np.log(1e-6), atol=2e-6)
    high, counts = normalized_log_distance(np.full(shape, 1000, np.float32), np.zeros((16, 256)), np.eye(16) * 8, diagnostics=True)
    np.testing.assert_allclose(high, np.log(32 ** 2), atol=2e-6)
    assert counts["centered_values_clipped"] == np.prod(shape)
    assert counts["whitened_values_clipped"] == np.prod(shape)
    assert np.isfinite(high).all()


def test_invalid_or_nonfinite_moments_fail_before_factorization():
    with pytest.raises(ValueError):
        whitening_from_moments(np.zeros((15, 256)), np.eye(16))
    second = np.eye(16)
    second[0, 1] = np.nan
    with pytest.raises(ValueError):
        whitening_from_moments(np.zeros((16, 256)), second)
