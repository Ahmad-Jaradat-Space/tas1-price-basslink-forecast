"""Split-conformal prediction wrapper.

Given a fitted point predictor, calibrate symmetric intervals on a
held-out calibration set so that the realised coverage matches the
nominal level *with finite-sample distribution-free guarantees*.

This is the textbook conformal recipe — see Vovk, Romano-Patterson
& Candes (2019). The implementation here is the simplest version
(absolute residuals as the non-conformity score).
"""

import numpy as np


def conformal_intervals(y_calib, mu_calib, mu_test, alpha=0.20):
    """Return (low, high) intervals at level (1 - alpha).

    `mu_calib`, `mu_test` are the model's point forecasts on the
    calibration and test sets respectively. The interval at each test
    point is mu_test ± q_hat, where q_hat is the empirical
    (1-alpha)-quantile of |y_calib - mu_calib|.
    """
    n = len(y_calib)
    scores = np.abs(y_calib - mu_calib)
    # finite-sample correction:  ceil((n+1)(1-alpha)) / n
    q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_idx = max(0, min(q_idx, n - 1))
    q_hat = float(np.sort(scores)[q_idx])
    low = mu_test - q_hat
    high = mu_test + q_hat
    return low, high, q_hat


def empirical_coverage(y, low, high):
    return float(((y >= low) & (y <= high)).mean())


def cqr_intervals(y_calib, lo_calib, hi_calib, lo_test, hi_test, alpha=0.20):
    """Conformalised Quantile Regression intervals (Romano et al. 2019).

    Uses the *quantile residual* score E_i = max(lo_calib - y, y - hi_calib)
    so the calibrated interval becomes [lo_test - q_hat, hi_test + q_hat].
    Unlike absolute-residual conformal, the resulting band is locally
    adaptive — wide where the q10/q90 model already is, narrow where it
    isn't.
    """
    n = len(y_calib)
    scores = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_idx = max(0, min(q_idx, n - 1))
    q_hat = float(np.sort(scores)[q_idx])
    return lo_test - q_hat, hi_test + q_hat, q_hat


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05,
                        seed=0):
    """Paired-bootstrap 95% CI on a regression metric.

    Returns (point, lo, hi) where lo/hi are the alpha/2 and 1-alpha/2
    percentiles of the bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi
