"""Bayesian dynamic linear model in numpyro.

Model: TAS1_t = α_t · VIC1_t + β_t + ε_t,  ε_t ~ N(0, σ_ε)
       α_t = α_{t-1} + η^α_t,  η^α ~ N(0, σ_α)
       β_t = β_{t-1} + η^β_t,  η^β ~ N(0, σ_β)

The time-varying intercept β captures Basslink loss + spread; the
slope α captures the regime — when it's near 1 the regions are
coupled, when it drifts the regions have detached.

Inference: NUTS with a small number of chains. Posterior gives full
uncertainty over (α_t, β_t) at every time point.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np


def model(vic1, tas1=None, T=None):
    if T is None:
        T = vic1.shape[0]
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(50.0))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(0.05))
    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(5.0))

    # innovations driving the random walks
    da = numpyro.sample("da", dist.Normal(0, 1).expand([T]))
    db = numpyro.sample("db", dist.Normal(0, 1).expand([T]))

    a0 = numpyro.sample("a0", dist.Normal(1.0, 0.2))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 10.0))

    a = a0 + jnp.cumsum(da * sigma_a)
    b = b0 + jnp.cumsum(db * sigma_b)
    numpyro.deterministic("alpha_t", a)
    numpyro.deterministic("beta_t", b)

    mu = a * vic1 + b
    numpyro.sample("y", dist.Normal(mu, sigma_eps), obs=tas1)


def fit_dlm(vic1, tas1, n_warmup=400, n_samples=400, seed=0,
            sub_every=4):
    """Fit the DLM by NUTS on a subsampled training set (every `sub_every`
    points) so MCMC stays tractable on a year of half-hourly data."""
    from numpyro.infer import NUTS, MCMC

    vic_sub = jnp.asarray(vic1[::sub_every].astype(np.float32))
    tas_sub = jnp.asarray(tas1[::sub_every].astype(np.float32))
    rng = jax.random.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(rng, vic1=vic_sub, tas1=tas_sub, T=len(vic_sub))
    samples = mcmc.get_samples()
    return mcmc, samples, np.arange(len(vic1))[::sub_every]


def posterior_predictive(samples, vic1):
    """Vectorised posterior predictive at every t in vic1, using the
    mean of the posterior over (α_t, β_t) — interpolated to the full
    high-resolution time grid via piecewise constant.
    """
    a_post = np.array(samples["alpha_t"])  # (n_post, T_sub)
    b_post = np.array(samples["beta_t"])   # (n_post, T_sub)
    sigma = np.array(samples["sigma_eps"])  # (n_post,)

    # take posterior median trajectories
    a_med = np.median(a_post, axis=0)
    b_med = np.median(b_post, axis=0)
    a_lo  = np.quantile(a_post, 0.05, axis=0)
    a_hi  = np.quantile(a_post, 0.95, axis=0)
    return {
        "alpha_med": a_med, "alpha_lo": a_lo, "alpha_hi": a_hi,
        "beta_med": b_med,
        "sigma_med": float(np.median(sigma)),
    }
