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
