"""Plot helpers for the notebook. Same conventions as my other capstones."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ACCENT = "#2c5f8d"
WARN = "#c44e52"
CONTEXT = "#7f7f7f"
TAS_COLOUR = "#3a8c5f"
VIC_COLOUR = "#9d62a8"


def apply_style():
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def caption(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def annotate_point(ax, x, y, text, dx=20, dy=20):
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#bbb", lw=0.6, alpha=0.9),
    )


def price_series(df, ax=None):
    """TAS1 vs VIC1 RRP over time, with a Basslink-congestion ribbon."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["SETTLEMENTDATE"], df["rrp_vic1"], color=VIC_COLOUR,
            alpha=0.6, lw=1.0, label="VIC1 RRP")
    ax.plot(df["SETTLEMENTDATE"], df["rrp_tas1"], color=TAS_COLOUR,
            alpha=0.9, lw=1.0, label="TAS1 RRP")
    ax.set_ylabel("RRP ($/MWh)")
    ax.set_xlabel("")
    ax.legend(loc="upper right")
    return ax


def price_scatter(df, ax=None, sample=None):
    """TAS1 vs VIC1 scatter coloured by Basslink congestion."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    d = df.sample(sample, random_state=0) if sample else df
    free = d.loc[~d["bass_congested"]]
    cong = d.loc[d["bass_congested"]]
    ax.scatter(free["rrp_vic1"], free["rrp_tas1"], s=6, alpha=0.3,
               color=ACCENT, label="Basslink free")
    ax.scatter(cong["rrp_vic1"], cong["rrp_tas1"], s=8, alpha=0.6,
               color=WARN, label="Basslink congested")
    lo = min(d["rrp_vic1"].min(), d["rrp_tas1"].min())
    hi = max(d["rrp_vic1"].max(), d["rrp_tas1"].max())
    ax.plot([lo, hi], [lo, hi], "--", color=CONTEXT, lw=1, label="y=x")
    ax.set_xlabel("VIC1 RRP ($/MWh)")
    ax.set_ylabel("TAS1 RRP ($/MWh)")
    ax.legend()
    return ax


def basslink_distribution(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df["bass_metered_mw"], bins=80, color=ACCENT, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    cutoff = df["bass_metered_mw"].abs().quantile(0.90)
    ax.axvline( cutoff, color=WARN, ls="--", lw=1,
               label=f"|flow| 90th pct ≈ {cutoff:.0f} MW")
    ax.axvline(-cutoff, color=WARN, ls="--", lw=1)
    ax.set_xlabel("Basslink metered flow MW (positive = TAS exporting)")
    ax.set_ylabel("count")
    ax.legend()
    return ax


def forecast_overlay(times, y_true, preds, ax=None):
    """Forecast comparison on a slice. preds is a dict of name -> array."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    ax.plot(times, y_true, color="black", lw=1.4, label="actual TAS1")
    palette = [ACCENT, WARN, "#7e57c2", "#2ca25f"]
    for (name, p), col in zip(preds.items(), palette):
        ax.plot(times, p, color=col, lw=1.0, alpha=0.85, label=name)
    ax.set_ylabel("RRP ($/MWh)")
    ax.legend(loc="upper right")
    return ax


def quantile_fan(times, y_true, q10, q50, q90, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(times, q10, q90, color=ACCENT, alpha=0.25,
                    label="10–90% quantile band")
    ax.plot(times, q50, color=ACCENT, lw=1.2, label="median forecast")
    ax.plot(times, y_true, color="black", lw=1.0, label="actual TAS1", alpha=0.8)
    ax.set_ylabel("RRP ($/MWh)")
    ax.legend(loc="upper right")
    return ax


def reliability(prob_pred, y_true, n_bins=10, ax=None):
    """Reliability diagram: predicted probability vs empirical frequency."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(prob_pred, edges) - 1, 0, n_bins - 1)
    obs = np.array([y_true[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                    for i in range(n_bins)])
    ax.plot([0, 1], [0, 1], "--", color=CONTEXT, label="perfectly calibrated")
    ax.plot(centres, obs, "-o", color=ACCENT, label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.legend()
    return ax


def feature_importance(names, importances, top=12, ax=None):
    import matplotlib.pyplot as plt
    order = np.argsort(importances)[::-1][:top]
    names_sorted = np.array(names)[order][::-1]
    vals = importances[order][::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(top), vals, color=ACCENT)
    ax.set_yticks(range(top))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("permutation importance")
    return ax


def quantile_calibration_bar(coverage_dict, ax=None):
    """Show empirical coverage of each nominal interval as a bar chart vs target."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    nominals = list(coverage_dict.keys())
    empirical = list(coverage_dict.values())
    x = np.arange(len(nominals))
    ax.bar(x - 0.2, nominals, 0.4, color=CONTEXT, label="nominal", alpha=0.6)
    ax.bar(x + 0.2, empirical, 0.4, color=ACCENT, label="empirical")
    for i, (n, e) in enumerate(zip(nominals, empirical)):
        ax.text(i + 0.2, e + 0.02, f"{e:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n*100)}%" for n in nominals])
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.1)
    ax.legend()
    return ax


def flow_score_distribution(scores_pos, scores_neg, ax=None):
    """Histogram of model scores split by true flow direction."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(scores_neg, bins=40, alpha=0.5, color=ACCENT, label="true: importing", density=True)
    ax.hist(scores_pos, bins=40, alpha=0.6, color=WARN, label="true: exporting", density=True)
    ax.set_xlabel("predicted P(exporting)")
    ax.set_ylabel("density")
    ax.legend()
    return ax


def model_comparison(df, ax=None):
    """Horizontal bar chart of point-forecast errors per model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))
    df = df.sort_values("RMSE", ascending=False)
    y = np.arange(len(df))
    h = 0.35
    ax.barh(y - h / 2, df["RMSE"], height=h, label="RMSE", color=ACCENT)
    ax.barh(y + h / 2, df["MAE"], height=h, label="MAE", color=WARN)
    for i, (a, b) in enumerate(zip(df["RMSE"], df["MAE"])):
        ax.text(a + 0.5, i - h / 2, f"{a:.1f}", va="center", fontsize=8)
        ax.text(b + 0.5, i + h / 2, f"{b:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("error ($/MWh)")
    ax.legend(loc="lower right")
    return ax
