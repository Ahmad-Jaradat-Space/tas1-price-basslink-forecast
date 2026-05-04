"""Pull AEMO dispatch data for TAS1 and VIC1 via NEMOSIS, aggregate to
30-minute trading intervals, build a tidy dataframe with the columns
the notebook needs.

NEMOSIS caches CSV downloads as feather files under data/cache/ on the
first call; subsequent runs are fast. The cache is gitignored.
"""

import os

import numpy as np
import pandas as pd
from nemosis import dynamic_data_compiler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "cache")

# AEMO identifiers
BASSLINK = "T-V-MNSP1"   # interconnector id for Basslink
TAS_LIMIT_MW = 500       # nominal Basslink capacity, used to flag congestion


def _ensure_cache():
    os.makedirs(CACHE, exist_ok=True)


def _pull_price(start, end, region):
    df = dynamic_data_compiler(
        start, end, "DISPATCHPRICE", CACHE,
        filter_cols=["REGIONID"], filter_values=([region],),
        keep_csv=False,
    )
    df = df[df["INTERVENTION"] == 0]
    return df[["SETTLEMENTDATE", "RRP"]].rename(columns={"RRP": f"rrp_{region.lower()}"})


def _pull_demand(start, end, region):
    df = dynamic_data_compiler(
        start, end, "DISPATCHREGIONSUM", CACHE,
        filter_cols=["REGIONID"], filter_values=([region],),
        keep_csv=False,
    )
    df = df[df["INTERVENTION"] == 0]
    cols = ["SETTLEMENTDATE", "TOTALDEMAND", "AVAILABLEGENERATION"]
    out = df[cols].rename(columns={
        "TOTALDEMAND": f"demand_{region.lower()}",
        "AVAILABLEGENERATION": f"avail_{region.lower()}",
    })
    return out


def _pull_basslink(start, end):
    df = dynamic_data_compiler(
        start, end, "DISPATCHINTERCONNECTORRES", CACHE,
        filter_cols=["INTERCONNECTORID"], filter_values=([BASSLINK],),
        keep_csv=False,
    )
    df = df[df["INTERVENTION"] == 0]
    cols = ["SETTLEMENTDATE", "METEREDMWFLOW", "MWFLOW", "MWLOSSES"]
    return df[cols].rename(columns={
        "METEREDMWFLOW": "bass_metered_mw",
        "MWFLOW": "bass_target_mw",
        "MWLOSSES": "bass_losses_mw",
    })


def load(start="2024/01/01 00:00:00", end="2025/01/01 00:00:00"):
    """Return a 30-minute dataframe with TAS1 RRP, VIC1 RRP, demand, and
    Basslink flow. Sign convention: bass_metered_mw > 0 means TAS exporting."""
    _ensure_cache()

    tas_p = _pull_price(start, end, "TAS1")
    vic_p = _pull_price(start, end, "VIC1")
    tas_d = _pull_demand(start, end, "TAS1")
    bass = _pull_basslink(start, end)

    df = (
        tas_p
        .merge(vic_p, on="SETTLEMENTDATE", how="inner")
        .merge(tas_d, on="SETTLEMENTDATE", how="inner")
        .merge(bass, on="SETTLEMENTDATE", how="inner")
    )
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])

    # aggregate 5-min dispatch to 30-min trading intervals
    df = df.set_index("SETTLEMENTDATE")
    agg = df.resample("30min", label="right", closed="right").mean()
    agg = agg.dropna().reset_index()

    # derived features
    agg["spread"] = agg["rrp_tas1"] - agg["rrp_vic1"]
    # Basslink nominal capacity is ±500 MW; treat the top decile of |flow|
    # as congested-equivalent — empirically this captures the limit-binding
    # intervals without needing the (not-published) explicit limit columns.
    cutoff = agg["bass_metered_mw"].abs().quantile(0.90)
    agg["bass_congested"] = agg["bass_metered_mw"].abs() >= cutoff
    agg["bass_direction"] = np.where(
        agg["bass_metered_mw"] > 5, "TAS_to_VIC",
        np.where(agg["bass_metered_mw"] < -5, "VIC_to_TAS", "idle"),
    )

    # calendar features
    t = agg["SETTLEMENTDATE"]
    agg["hour"] = t.dt.hour
    agg["dow"] = t.dt.dayofweek
    agg["month"] = t.dt.month
    agg["is_weekend"] = (agg["dow"] >= 5).astype(int)
    agg["minute_of_day"] = t.dt.hour * 60 + t.dt.minute

    return agg


def make_features(df, lags=(1, 2, 4, 48, 336)):
    """Build a lagged feature matrix for ML models.

    `lags` are in 30-minute intervals: 1=30min, 48=1d, 336=1wk.
    Returns (X, y_price, y_flow, datetimes) with leading rows dropped to
    cover the longest lag.
    """
    out = df.copy()
    for L in lags:
        out[f"rrp_tas1_lag{L}"] = out["rrp_tas1"].shift(L)
        out[f"rrp_vic1_lag{L}"] = out["rrp_vic1"].shift(L)
        out[f"bass_lag{L}"] = out["bass_metered_mw"].shift(L)
        out[f"demand_lag{L}"] = out["demand_tas1"].shift(L)

    # cyclical encodings of time-of-day and day-of-week
    out["sin_h"] = np.sin(2 * np.pi * out["minute_of_day"] / (24 * 60))
    out["cos_h"] = np.cos(2 * np.pi * out["minute_of_day"] / (24 * 60))
    out["sin_d"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["cos_d"] = np.cos(2 * np.pi * out["dow"] / 7)

    feature_cols = [c for c in out.columns if (
        c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_lag4")
        or c.endswith("_lag48") or c.endswith("_lag336")
        or c in ["sin_h", "cos_h", "sin_d", "cos_d", "is_weekend",
                 "demand_tas1", "rrp_vic1"]
    )]

    out = out.dropna().reset_index(drop=True)
    X = out[feature_cols].values.astype(np.float32)
    y_price = out["rrp_tas1"].values.astype(np.float32)
    y_flow = (out["bass_metered_mw"] > 5).astype(int).values  # 1 = exporting
    return X, y_price, y_flow, out["SETTLEMENTDATE"], feature_cols


def time_split(n, train_frac=0.7, val_frac=0.15):
    """Time-ordered train/val/test split — no shuffling."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


# ===================================================================
# Full-NEM pull for the GNN chapter
# ===================================================================
NEM_REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]

# (from, to) MNSP/regulated interconnectors. Sign convention: positive
# flow on the wire goes from the first region to the second.
INTERCONNECTORS = [
    ("NSW1", "QLD1", "NSW1-QLD1"),
    ("VIC1", "NSW1", "VIC1-NSW1"),
    ("V-S-MNSP1", "VIC1", "SA1"),       # Murraylink, MNSP
    ("VIC1", "SA1",  "V-SA"),            # Heywood
    ("VIC1", "TAS1", "T-V-MNSP1"),       # Basslink (signed by 'from')
    ("NSW1", "QLD1", "N-Q-MNSP1"),       # Terranora, MNSP
]


def load_nem_graph(start="2024/01/01 00:00:00", end="2025/01/01 00:00:00"):
    """Return a wide 30-min DataFrame with RRP for every NEM region.

    For the GNN chapter — each row is one timestamp, columns are
    rrp_{region} for each of the five NEM regions.
    """
    _ensure_cache()
    pieces = []
    for region in NEM_REGIONS:
        df = dynamic_data_compiler(
            start, end, "DISPATCHPRICE", CACHE,
            filter_cols=["REGIONID"], filter_values=([region],),
            keep_csv=False,
        )
        df = df[df["INTERVENTION"] == 0]
        df = df[["SETTLEMENTDATE", "RRP"]].rename(
            columns={"RRP": f"rrp_{region.lower()}"}
        )
        pieces.append(df)
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="SETTLEMENTDATE", how="inner")
    out["SETTLEMENTDATE"] = pd.to_datetime(out["SETTLEMENTDATE"])
    out = out.set_index("SETTLEMENTDATE").sort_index()
    out = out.resample("30min", label="right", closed="right").mean().dropna()
    return out.reset_index()


# Adjacency matrix for the NEM regional graph (5 nodes).
# Entry [i, j] = 1 iff there is a direct interconnector between i and j.
def nem_adjacency():
    A = np.zeros((5, 5), dtype=np.float32)
    idx = {r: i for i, r in enumerate(NEM_REGIONS)}
    edges = [
        ("NSW1", "QLD1"),
        ("NSW1", "VIC1"),
        ("VIC1", "SA1"),
        ("VIC1", "TAS1"),
    ]
    for a, b in edges:
        i, j = idx[a], idx[b]
        A[i, j] = A[j, i] = 1.0
    # add self-loops for stability
    A = A + np.eye(5, dtype=np.float32)
    return A
