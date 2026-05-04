"""Plot helpers for the notebook. Production-grade design system shared
across the capstone series."""

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Design palette
PRIMARY = "#0E4F5F"
ACCENT  = "#D88C4A"
GOOD    = "#5C9D7E"
WARN    = "#B0413E"
MUTED   = "#7A8C99"
INK     = "#1A1A1A"
PAPER   = "#FAFAF7"
TAS_COLOUR = PRIMARY
VIC_COLOUR = "#5E548E"
CONTEXT = MUTED


def apply_style():
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.titlepad": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#BFC4CA",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "font.size": 10.5,
    })
    try:
        mpl.rcParams["text.parse_math"] = False
    except KeyError:
        pass


def kpi_card(ax, value, label, sub=None, color=PRIMARY):
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=22, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center",
            fontsize=10, color=INK, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.12, sub, ha="center", va="center",
                fontsize=8.5, color=MUTED, transform=ax.transAxes, style="italic")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, ec="#D8DCE2", lw=1.0))


def kpi_banner(values):
    fig, axes = plt.subplots(1, len(values), figsize=(2.8 * len(values), 1.8))
    if len(values) == 1:
        axes = [axes]
    palette = [PRIMARY, ACCENT, WARN, GOOD, "#5E548E"]
    for ax, v, color in zip(axes, values, palette):
        kpi_card(ax, *v, color=color)
    plt.tight_layout()
    return fig


def business_summary(rows, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1))
    ax.axis("off")
    n = len(rows)
    for i, (lhs, rhs) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax.text(0.02, y, lhs, transform=ax.transAxes,
                fontsize=10.5, color=INK, fontweight="bold", va="center")
        ax.text(0.34, y, rhs, transform=ax.transAxes,
                fontsize=10, color=INK, va="center")
        ax.plot([0.01, 0.99], [1 - i / n, 1 - i / n],
                color="#E0E4EA", lw=0.6, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    return ax


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


def scalogram(coeffs, periods_hours, times, ax=None):
    """Continuous wavelet transform power, log-periods on the y axis."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    power = np.log10(np.abs(coeffs) ** 2 + 1e-9)
    extent = [0, len(times), periods_hours[0], periods_hours[-1]]
    im = ax.imshow(power, aspect="auto", origin="lower",
                   cmap="magma", extent=extent)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 6, 12, 24, 24 * 7])
    ax.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} h")
    )
    ax.set_ylabel("period")
    ax.set_xlabel("time index (30-min)")
    plt.colorbar(im, ax=ax, label="log10 power")
    return ax


def stft_heatmap(spec, starts, periods_hours, ax=None):
    """STFT power vs (time-window, period). Drops DC and aliased bins."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    keep = ~np.isnan(periods_hours) & (periods_hours <= 7 * 24) & (periods_hours >= 0.5)
    img = np.log10(spec[:, keep].T + 1e-6)
    extent = [starts[0], starts[-1],
              periods_hours[keep].min(), periods_hours[keep].max()]
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
                   extent=extent)
    ax.set_yscale("log")
    ax.set_ylabel("period (hours)")
    ax.set_xlabel("window start (30-min index)")
    plt.colorbar(im, ax=ax, label="log10 power")
    return ax


def posterior_alpha(alpha_med, alpha_lo, alpha_hi, sub_idx, ax=None):
    """Time-varying coupling coefficient α_t with credible band."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    ax.fill_between(sub_idx, alpha_lo, alpha_hi, color=ACCENT, alpha=0.25,
                    label="90% credible band")
    ax.plot(sub_idx, alpha_med, color=ACCENT, lw=1.4, label="posterior median")
    ax.axhline(1.0, color=CONTEXT, ls="--", lw=0.8, label="α = 1 (perfect coupling)")
    ax.set_xlabel("training-set index (subsampled)")
    ax.set_ylabel("α_t (TAS1 sensitivity to VIC1)")
    ax.legend()
    return ax


def conformal_panel(times, y_true, mu, intervals, ax=None):
    """Three side-by-side panels: nominal q10/q90, conformal 80%,
    conformal 95%. `intervals` is dict label -> (low, high)."""
    import matplotlib.pyplot as plt
    n = len(intervals)
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(5.5 * n, 4), sharey=True)
    if n == 1:
        ax = [ax]
    for a, (label, (lo, hi)) in zip(ax, intervals.items()):
        a.fill_between(times, lo, hi, color=ACCENT, alpha=0.25, label=label)
        a.plot(times, mu, color=ACCENT, lw=1.0, label="median forecast")
        a.plot(times, y_true, color="black", lw=1.0, alpha=0.85, label="actual")
        a.set_title(label)
        a.set_xlabel("time")
        a.legend(fontsize=8, loc="upper right")
    ax[0].set_ylabel("RRP ($/MWh)")
    return ax


def nem_network(adj, region_names, flows=None, ax=None):
    """Diagram of the NEM regional graph. `flows` is a (n,n) matrix of
    average absolute flow magnitudes used to weight edge widths."""
    import matplotlib.pyplot as plt
    import networkx as nx
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    G = nx.from_numpy_array((adj - np.eye(adj.shape[0])).astype(int))
    pos = {0: (-0.6, 0.7), 1: (0.6, 0.9), 2: (-0.9, 0.0),
           3: (-0.3, -0.9), 4: (0.0, 0.0)}
    pos = {i: pos[i] for i in range(adj.shape[0])}
    nx.draw_networkx_nodes(G, pos, node_color=ACCENT, node_size=900, ax=ax)
    if flows is not None:
        widths = []
        edges = list(G.edges())
        for u, v in edges:
            w = flows[u, v] / (flows.max() + 1e-9) * 6 + 0.5
            widths.append(w)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths,
                               edge_color=CONTEXT, ax=ax, alpha=0.7)
    else:
        nx.draw_networkx_edges(G, pos, width=2, edge_color=CONTEXT, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: r for i, r in enumerate(region_names)},
                            font_color="white", font_weight="bold", ax=ax)
    ax.set_xlim(-1.2, 1.0); ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
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


def model_comparison(df, ax=None, sort_by="MAE", baseline_models=None):
    """Horizontal bar chart of point-forecast errors per model.

    `baseline_models` is a list/set of model names to render with a
    hatched bar so the reader sees which entries are structural baselines
    rather than ML results. CIs (RMSE_lo/RMSE_hi/MAE_lo/MAE_hi columns)
    are drawn as thin black error bars when present.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.6))
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    h = 0.35
    baseline_models = set(baseline_models or [])
    rmse_color = [CONTEXT if m in baseline_models else ACCENT for m in df["model"]]
    mae_color = [CONTEXT if m in baseline_models else WARN for m in df["model"]]
    rmse_hatch = ["///" if m in baseline_models else "" for m in df["model"]]
    bars_rmse = ax.barh(y - h / 2, df["RMSE"], height=h, color=rmse_color,
                        label="RMSE", edgecolor="white")
    for bar, hatch in zip(bars_rmse, rmse_hatch):
        bar.set_hatch(hatch)
    ax.barh(y + h / 2, df["MAE"], height=h, color=mae_color, label="MAE",
            edgecolor="white")
    if {"RMSE_lo", "RMSE_hi"}.issubset(df.columns):
        ax.errorbar(df["RMSE"], y - h / 2,
                    xerr=[df["RMSE"] - df["RMSE_lo"], df["RMSE_hi"] - df["RMSE"]],
                    fmt="none", ecolor="black", capsize=3, lw=0.8)
    if {"MAE_lo", "MAE_hi"}.issubset(df.columns):
        ax.errorbar(df["MAE"], y + h / 2,
                    xerr=[df["MAE"] - df["MAE_lo"], df["MAE_hi"] - df["MAE"]],
                    fmt="none", ecolor="black", capsize=3, lw=0.8)
    for i, (a, b) in enumerate(zip(df["RMSE"], df["MAE"])):
        ax.text(a + 0.5, i - h / 2, f"{a:.1f}", va="center", fontsize=8)
        ax.text(b + 0.5, i + h / 2, f"{b:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("error ($/MWh)")
    legend_lines = [
        plt.Rectangle((0, 0), 1, 1, color=ACCENT, label="ML model — RMSE"),
        plt.Rectangle((0, 0), 1, 1, color=WARN, label="ML model — MAE"),
        plt.Rectangle((0, 0), 1, 1, color=CONTEXT, hatch="///",
                      label="structural baseline"),
    ]
    ax.legend(handles=legend_lines, loc="lower right", fontsize=8)
    return ax
