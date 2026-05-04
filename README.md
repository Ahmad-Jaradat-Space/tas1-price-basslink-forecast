# TAS1 Price and Basslink Flow Forecasting

Tasmania's wholesale electricity market is structurally weird and that's what makes it a good forecasting target. The state runs ~80% on hydro with reservoir storage that behaves like a giant battery, ~17% on wind, and ties to the rest of the National Electricity Market (NEM) through a single 500 MW HVDC link to Victoria — Basslink. So most of the time TAS1 prices are basically VIC1 prices plus a small spread. Every time Basslink congests, TAS1 detaches and goes wherever local hydro+wind takes it. Those detachments are where forecasting actually matters.

The notebook puts three claims on trial:

1. The "copy VIC1" baseline is hard to beat on average.
2. Where ML earns its keep is in Basslink-congested half-hours.
3. A probabilistic forecast (10/50/90% quantile band) is more honest than a point forecast for a market with this much tail risk.

## How the notebook is laid out

The notebook reads as a short paper with five sections:

1. **Introduction** — why Tasmania's market is unusual and what this notebook actually delivers.
2. **Data** — pulled from AEMO via [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS), aggregated to 30-minute trading intervals; price series, TAS1 vs VIC1 scatter, Basslink flow distribution, and the explicit modelling hypothesis.
3. **Methods** — three baselines (seasonal naïve, VIC1 anchor), histogram gradient boosting on engineered features for the point forecast, and three quantile-GB models (10/50/90 percentile) for the probabilistic forecast band.
4. **Results** — point-forecast errors compared, a representative forecast slice, calibration of the probabilistic band, and a confusion matrix on Basslink flow direction.
5. **Conclusion** — answers to the three claims and what would change with Marinus Link in the picture.

Every plot is read out loud: a one-line setup before the cell, a finding-style title, and a 2–4 sentence takeaway after — *what to look at, what it means, what it indicates next.*

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first run downloads about 12 months of AEMO dispatch CSVs into `data/cache/` (gitignored) — this takes a few minutes the first time and is essentially instant on subsequent runs because NEMOSIS caches as feather files. The committed `notebook.ipynb` is already executed, so GitHub renders all outputs and plots inline without needing to run anything.

## What's where

- `notebook.ipynb` — the whole story, runs top to bottom
- `data.py` — NEMOSIS pulls, 30-min aggregation, lagged feature builder, time-ordered split
- `models.py` — naïve baselines, RMSE/MAE/SMAPE helpers, pinball loss + calibration helpers
- `plots.py` — small matplotlib helpers used by the notebook
