"""Forecasting models — baselines plus a small Keras stack with a
quantile head for probabilistic forecasts.

The tone here is deliberately scientific: each model is one or two
functions or a thin class. No abstractions for the sake of it.
"""

import numpy as np


# -------------------------------------------------------------- baselines
def naive_seasonal(y_train_full, y_target_idx, period=48):
    """Predict y[t] = y[t - period]. period=48 means 'yesterday-same-time'."""
    return y_train_full[y_target_idx - period]


def vic1_anchor(rrp_vic1):
    """The 'do nothing clever' baseline — TAS1 forecast = VIC1 spot.
    This is the benchmark anything ML has to beat to earn its keep."""
    return rrp_vic1


# -------------------------------------------------------------- quantile loss helpers
def pinball_loss(y_true, y_pred, q):
    """Quantile (pinball) loss for quantile q in (0,1)."""
    e = y_true - y_pred
    return np.where(e >= 0, q * e, (q - 1) * e).mean()


def calibration(y_true, q_low, q_high):
    """Fraction of observations falling inside [q_low, q_high].
    For a well-calibrated 80% interval this should be ≈ 0.80."""
    return float(((y_true >= q_low) & (y_true <= q_high)).mean())


# -------------------------------------------------------------- LSTM with quantile head
def build_lstm_quantile(input_shape, hidden=64, quantiles=(0.1, 0.5, 0.9), seed=0):
    """One-layer LSTM mapping a window of features to three quantile
    forecasts of next-period TAS1 RRP. Trained with summed pinball loss.
    """
    import tensorflow as tf
    tf.random.set_seed(seed)

    quantiles = tuple(quantiles)

    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(hidden, kernel_initializer="glorot_uniform")(inp)
    x = tf.keras.layers.Dropout(0.1, seed=seed)(x)
    outs = tf.keras.layers.Dense(len(quantiles))(x)
    model = tf.keras.Model(inp, outs)

    def quantile_loss(y_true, y_pred):
        # y_true shape (B,1); y_pred shape (B,K)
        y_true = tf.tile(y_true, [1, len(quantiles)])
        e = y_true - y_pred
        q = tf.constant(quantiles, dtype=tf.float32)
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss=quantile_loss)
    return model, quantiles


# -------------------------------------------------------------- windowing
def make_windows(X, y, window=8):
    """Slide a window of length `window` across X, predicting y at the
    end of each window. Used by the LSTM."""
    n = len(X) - window
    Xw = np.stack([X[i:i + window] for i in range(n)], axis=0).astype(np.float32)
    yw = y[window:window + n].astype(np.float32).reshape(-1, 1)
    return Xw, yw


# -------------------------------------------------------------- metrics
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    """Symmetric MAPE — robust to near-zero prices that NEM occasionally has."""
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=1.0)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom))
