"""Price-sensitivity estimation from randomized price experiments.

Model, per zone:   log(demand) = FE(off-day regime x hour-of-week) + beta * (m - 1) + noise

``m`` is the realised price multiplier. The static operator schedule is a deterministic
function of hour-of-week, so it is absorbed by the fixed effects; the only variation left to
identify ``beta`` is the randomized day-level experiments. By the Frisch-Waugh-Lovell theorem
this is the OLS slope after demeaning ``log(demand)`` and ``(m - 1)`` within hour-of-week cells.
Standard errors are cluster-robust by zone-day because the treatment is assigned per day.

Two outcomes are compared:

* ``demand`` = arrivals (occupied + drivers turned away) is the unconstrained demand;
* ``occupancy`` is censored at capacity. Using it silently biases ``beta`` toward zero in lots
  that are full at peak, which is the classic mistake in parking elasticity work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _slope_fe(y: np.ndarray, x: np.ndarray, cell: np.ndarray, cluster: np.ndarray) -> tuple[float, float, int]:
    """OLS slope of y on x after within-cell demeaning, with cluster-robust (CR1) SE."""
    df = pd.DataFrame({"y": y, "x": x, "cell": cell})
    y_dm = df["y"] - df.groupby("cell")["y"].transform("mean")
    x_dm = df["x"] - df.groupby("cell")["x"].transform("mean")
    sxx = float((x_dm ** 2).sum())
    # No (numerically) usable price variation left after the fixed effects => not identified.
    # Returning float round-off here would look like an estimate with an absurd standard error.
    if not np.isfinite(sxx) or sxx < 1e-9 * max(len(df), 1):
        return float("nan"), float("nan"), len(df)
    beta = float((x_dm * y_dm).sum() / sxx)
    resid = y_dm - beta * x_dm
    g = pd.Series((x_dm * resid).to_numpy()).groupby(cluster).sum()
    G, n = g.shape[0], len(df)
    k = df["cell"].nunique() + 1
    cr1 = (G / (G - 1)) * ((n - 1) / (n - k))
    se = float(np.sqrt(cr1 * (g ** 2).sum()) / sxx)
    return beta, se, n


def estimate(hourly: pd.DataFrame, until: str | pd.Timestamp) -> pd.DataFrame:
    """Per-zone semi-elasticity estimates using only data up to ``until`` (inclusive)."""
    d = hourly[(hourly["date"] <= pd.Timestamp(until)) & hourly["sensor_ok"]].copy()
    # Cell = (off-day regime, hour-of-week). The operator's surcharge keys on "off-day" (weekend OR
    # holiday), so a weekday holiday must not share a cell with ordinary weekdays: holidays shift
    # demand *and* trigger the surcharge, which would confound the price effect.
    d["how"] = d["is_offday"].astype(int) * 168 + d["dow"] * 24 + d["hour"]
    d["cluster"] = d["zone_id"].astype(str) + "_" + d["date"].dt.strftime("%Y%m%d")
    rows = []
    for z in C.ZONES:
        s = d[d["zone_id"] == z.id]
        x = (s["price_mult"] - 1.0).to_numpy()
        cell = s["how"].to_numpy()
        cl = s["cluster"].to_numpy()
        y_dem = np.log(np.maximum(s["arrivals"].to_numpy(), 0.5) / z.capacity)
        y_occ = np.log(np.maximum(s["occupied"].to_numpy(), 0.5) / z.capacity)
        b, se, n = _slope_fe(y_dem, x, cell, cl)
        b_naive, se_naive, _ = _slope_fe(y_occ, x, cell, cl)
        rows.append({"zone_id": z.id, "zone": z.name, "beta": b, "se": se,
                     "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
                     "beta_naive_censored": b_naive, "n_obs": n,
                     "n_experiment_days": int(s.loc[s["is_experiment"], "date"].nunique())})
    return pd.DataFrame(rows)
