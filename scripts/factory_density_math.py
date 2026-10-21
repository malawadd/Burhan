"""Stable normal-feature density arithmetic; no model or data access."""
import numpy as np


def whitening_from_moments(column_mean, second_moment):
    mean = np.asarray(column_mean, dtype=np.float64)
    second = np.asarray(second_moment, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[0] != 16 or second.shape != (16, 16):
        raise ValueError('Normal-density moment geometry changed')
    if not np.isfinite(mean).all() or not np.isfinite(second).all():
        raise ValueError('Normal-density moments must be finite')
    covariance = second - mean @ mean.T / mean.shape[1]
    covariance = (covariance + covariance.T) * .5
    diagonal = np.maximum(np.diag(covariance), 0.)
    shrunk = .9 * covariance + .1 * np.diag(diagonal)
    shrunk = (shrunk + shrunk.T) * .5
    values, vectors = np.linalg.eigh(shrunk)
    floor = max(1e-6, .001 * float(np.median(diagonal)))
    safe = np.maximum(values, floor)
    whitening = np.diag(1. / np.sqrt(safe)) @ vectors.T
    scale = max(1., float(np.abs(whitening).sum(axis=1).max()) / 8.)
    whitening /= scale
    if not np.isfinite(whitening).all():
        raise ValueError('Invalid finite whitening matrix')
    return whitening.astype(np.float32), {
        'raw_covariance_min_eigenvalue': float(np.linalg.eigvalsh(covariance).min()),
        'shrunk_min_eigenvalue_before_floor': float(values.min()),
        'eigenvalue_floor': floor, 'floored_eigenvalues': int((values < floor).sum()),
        'global_whitening_rescale': scale,
        'maximum_whitening_row_l1': float(np.abs(whitening).sum(axis=1).max()),
        'finite': True}


def normalized_log_distance(features, mean, whitening, diagnostics=False):
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (16, 640, 256) or not np.isfinite(features).all():
        raise ValueError('Invalid native normal features')
    residual = features - np.asarray(mean, dtype=np.float32)[:, None, :]
    centered = np.clip(residual, -32., 32.)
    white = np.asarray(whitening, dtype=np.float32) @ centered.reshape(16, -1)
    clipping = {'centered_values_clipped': int((np.abs(residual) > 32).sum()),
                'whitened_values_clipped': int((np.abs(white) > 32).sum()),
                'channel_values': int(features.size)}
    white = np.clip(white, -32., 32.)
    distance = np.mean(white * white, axis=0)
    result = np.log(np.maximum(distance, 1e-6)).reshape(640, 256)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite log distance')
    return (result, clipping) if diagnostics else result
