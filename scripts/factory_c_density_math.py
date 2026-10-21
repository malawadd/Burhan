"""Stable normal-feature density arithmetic; no model or data access."""
import numpy as np


def dummy_bank():
    """Deterministic dense nonconstant, half-exact initialization; not fitted data."""
    rng=np.random.default_rng(20261026)
    matrix=rng.normal(0,.15,(16,16)).astype(np.float32)+np.eye(16,dtype=np.float32)
    matrix/=max(1.,float(np.abs(matrix).sum(axis=1).max())/8.)
    magnitude=np.abs(matrix);rounded=magnitude.astype(np.float16)
    up=rounded.astype(np.float32)>magnitude
    rounded[up]=np.nextafter(rounded[up],np.float16(0))
    matrix=np.copysign(rounded.astype(np.float32),matrix)
    mean=np.linspace(-.1,.1,256,dtype=np.float32)[None].repeat(16,0).astype(np.float16).astype(np.float32)
    return {'mean':mean,'whitening':matrix,'q50':-.5,'q95':.5,'qspan':1.}


def precision_corrected_moments(mean64,second):
    """Fit covariance around the DEPLOYED half mean, not an unrounded mean."""
    mean64=np.asarray(mean64,np.float64)
    second=np.asarray(second,np.float64)
    if mean64.shape!=(16,256) or second.shape!=(16,16) or not np.isfinite(mean64).all() or not np.isfinite(second).all():
        raise ValueError('Invalid C normal moments')
    mean=mean64.astype(np.float16).astype(np.float32)
    if not np.isfinite(mean).all():raise ValueError('Deployed mean overflow')
    muq=mean.astype(np.float64)
    covariance=second-(mean64@muq.T+muq@mean64.T-muq@muq.T)/256
    whitening,report=whitening_from_moments(np.zeros_like(mean64),covariance)
    magnitude=np.abs(whitening);rounded=magnitude.astype(np.float16)
    up=rounded.astype(np.float32)>magnitude
    rounded[up]=np.nextafter(rounded[up],np.float16(0))
    whitening=np.copysign(rounded.astype(np.float32),whitening)
    report.update(center='Actual residual covariance around deployed nearest-half muq',
                  whitening_quantization='Half magnitudes toward zero, preserving row-L1<=8',
                  quantized_maximum_row_l1=float(np.abs(whitening).sum(1).max()),
                  mean_quantization_max_error=float(np.max(np.abs(mean64-mean))))
    return mean,whitening,report


def quantized_quantiles(rawq50,rawq95):
    if not np.isfinite([rawq50,rawq95]).all() or rawq95<rawq50:
        raise ValueError('Invalid normal-only quantiles')
    q50,q95=float(np.float16(rawq50)),float(np.float16(rawq95))
    qspan=float(np.float16(max(.001,rawq95-rawq50)))
    if not np.isfinite([q50,q95,qspan]).all() or qspan<=0:raise ValueError('Quantized quantile overflow')
    return q50,q95,qspan


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
