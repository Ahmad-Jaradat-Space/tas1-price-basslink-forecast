"""Spectral characterisation of TAS1 RRP.

We use a continuous Morlet wavelet transform (via PyWavelets) and a
short-time Fourier transform (numpy.fft) to expose the multi-scale
structure of the price series — daily, weekly and the high-frequency
residual.
"""

import numpy as np
import pywt


def cwt_morlet(prices, dt_minutes=30, n_scales=64,
               freq_low_hz=1 / (14 * 24 * 3600),
               freq_high_hz=1 / (15 * 60)):
    """Continuous Morlet wavelet transform.

    Returns:
        coeffs: complex array, shape (n_scales, len(prices))
        periods_hours: physical period of each scale
    """
    # PyWavelets uses a sampling period — let's express dt in seconds
    dt_seconds = dt_minutes * 60
    central_freq = pywt.central_frequency("cmor1.5-1.0")
    period_low = 1 / freq_high_hz   # smallest physical period
    period_high = 1 / freq_low_hz   # largest physical period
    periods_seconds = np.geomspace(period_low, period_high, n_scales)
    scales = central_freq * periods_seconds / dt_seconds
    coeffs, _ = pywt.cwt(prices - prices.mean(), scales, "cmor1.5-1.0",
                         sampling_period=dt_seconds)
    periods_hours = periods_seconds / 3600
    return coeffs, periods_hours


def stft(prices, dt_minutes=30, window_periods=336, hop=48):
    """Short-time Fourier transform.

    Default window is one week (336 30-min periods); hop is one day.
    Returns the power-spectrum matrix and the period axis in hours.
    """
    n = len(prices)
    win = np.hanning(window_periods)
    starts = np.arange(0, n - window_periods + 1, hop)
    spec = np.empty((len(starts), window_periods // 2 + 1))
    for k, s in enumerate(starts):
        chunk = (prices[s:s + window_periods] - prices[s:s + window_periods].mean()) * win
        F = np.fft.rfft(chunk)
        spec[k] = (np.abs(F) ** 2)

    freqs = np.fft.rfftfreq(window_periods, d=dt_minutes * 60)  # cycles/sec
    # avoid 1/0 for DC; replace with NaN so callers can mask
    with np.errstate(divide="ignore"):
        periods_hours = np.where(freqs > 0, 1 / (freqs * 3600), np.nan)
    return spec, starts, periods_hours
