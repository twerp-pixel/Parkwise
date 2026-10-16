"""Leakage-safe feature engineering for the day-ahead forecast.

Forecast contract: the forecast for day D is issued at the end of day D-1. A feature for any
hour of day D may therefore use (a) the calendar, (b) weather *forecasts*, (c) the published
event schedule, and (d) demand observed on days <= D-1. Nothing else. ``tests/test_features.py``
proves this by corrupting all outcomes from day D onward and checking features for day D are
unchanged.

Target: ``d0`` = demand at the *base* price (multiplier 1.0), expressed as a fraction of
capacity. Observed arrivals are "deflated" by the estimated price effect:

    d0 = arrivals / capacity * exp(-beta_hat * (m - 1))

so history gathered under different prices (experiments, night discounts) is comparable, and
the pricing engine can ask "what happens at multiplier m?" of one clean demand number.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

LAG_FEATURES = ["lag_24", "lag_168", "lag_336", "mean_7d_same_hour", "mean_4w_same_how",
                "level_1d", "level_7d", "same_type_lag"]
CALENDAR_FEATURES = ["hour", "dow", "is_offday", "is_holiday", "is_pre_holiday", "is_post_holiday", "month"]
WEATHER_FEATURES = ["temp_fc", "rain_fc", "rain_fc_next3", "rain_fc_prev3"]
EVENT_FEATURES = ["ev_local_rel", "ev_local_size", "ev_local_kind", "ev_near_load", "ev_near_rel"]
ZONE_FEATURES = ["zone_idx", "capacity", "base_price", "has_venue"] + [f"poi_{a}" for a in C.ARCHETYPES]
# Zone-agnostic feature set for brand-new lots: no zone identity, no history.
# Raw capacity / base_price are dropped too: unique per zone, trees would use them as zone IDs and
# route a held-out zone to an arbitrary neighbour. Generalisation must come from the POI mix.
COLD_START_FEATURES = [f for f in CALENDAR_FEATURES + WEATHER_FEATURES + EVENT_FEATURES + ZONE_FEATURES
                       if f not in ("zone_idx", "capacity", "base_price")]
ALL_FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES + EVENT_FEATURES + ZONE_FEATURES + LAG_FEATURES


@dataclass
class FeatureSet:
    X: pd.DataFrame          # model inputs, one row per zone-hour (zone-major, time-ascending)
    y: np.ndarray            # d0 target; NaN where unobserved (sensor gap or the live day)
    meta: pd.DataFrame       # zone_id, date, hour, observed occupancy, price multiplier, capacity
    d0: np.ndarray           # [Z, D, 24] deflated demand history (NaN-aware)
    days: pd.DatetimeIndex


def deflate(hourly: pd.DataFrame, beta_hat: np.ndarray) -> np.ndarray:
    z = hourly["zone_id"].to_numpy()
    d = hourly["arrivals"].to_numpy() / hourly["capacity"].to_numpy()
    return d * np.exp(-beta_hat[z] * (hourly["price_mult"].to_numpy() - 1.0))


def _shift(A: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(A, np.nan)
    out[:, k:, :] = A[:, :-k, :]
    return out


def _nanmean_stack(arrs: list[np.ndarray]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(arrs), axis=0)


def _prev_same_type_index(offday: np.ndarray) -> np.ndarray:
    """For each day d, the latest day < d with the same off-day/working-day type (-1 if none)."""
    out = np.full(len(offday), -1, dtype=int)
    last = {False: -1, True: -1}
    for d, t in enumerate(offday):
        out[d] = last[bool(t)]
        last[bool(t)] = d
    return out


def _event_features(days: pd.DatetimeIndex, events: pd.DataFrame) -> dict[str, np.ndarray]:
    """What the platform *plans* from the published event schedule (not the hidden true effect)."""
    Z, T = C.N_ZONES, len(days) * 24
    dist = C.distance_matrix_km()
    t0 = days[0]
    rel_local = np.full((Z, T), np.nan)
    size_local = np.full((Z, T), np.nan)
    kind_local = np.full((Z, T), np.nan)
    near_load = np.zeros((Z, T))
    near_rel = np.full((Z, T), np.nan)
    near_best = np.zeros((Z, T))
    for e in events.itertuples():
        s_idx = int((e.start - t0) / pd.Timedelta(hours=1))
        for tau in range(-6, 9):
            i = s_idx + tau
            if not 0 <= i < T:
                continue
            for z in range(Z):
                if z == e.zone_id:
                    if np.isnan(rel_local[z, i]) or abs(tau) < abs(rel_local[z, i]):
                        rel_local[z, i], size_local[z, i] = tau, e.size
                        kind_local[z, i] = 0.0 if e.kind == "sale" else 1.0
                elif e.kind != "sale" and -4 <= tau <= 6:
                    w = e.size * np.exp(-dist[z, e.zone_id] / 1.5)
                    near_load[z, i] += w
                    if w > near_best[z, i]:
                        near_best[z, i], near_rel[z, i] = w, tau
    return {"ev_local_rel": rel_local, "ev_local_size": size_local, "ev_local_kind": kind_local,
            "ev_near_load": near_load, "ev_near_rel": near_rel}


def build(hourly: pd.DataFrame, events: pd.DataFrame, beta_hat: np.ndarray) -> FeatureSet:
    """Build features for every zone-hour in ``hourly`` (which may include the unobserved live day)."""
    hourly = hourly.sort_values(["zone_id", "ts"]).reset_index(drop=True)
    Z = C.N_ZONES
    days = pd.DatetimeIndex(sorted(hourly["date"].unique()))
    D = len(days)
    assert len(hourly) == Z * D * 24, "hourly frame must be a complete zone x day x hour grid"

    d0_flat = deflate(hourly, beta_hat)
    d0_flat = np.where(hourly["sensor_ok"].to_numpy(), d0_flat, np.nan)
    D0 = d0_flat.reshape(Z, D, 24)

    lag = {
        "lag_24": _shift(D0, 1),
        "lag_168": _shift(D0, 7),
        "lag_336": _shift(D0, 14),
        "mean_7d_same_hour": _nanmean_stack([_shift(D0, k) if k < D else np.full_like(D0, np.nan) for k in range(1, 8)]),
        "mean_4w_same_how": _nanmean_stack([_shift(D0, k) for k in (7, 14, 21, 28)]),
    }
    day_mean = _nanmean_stack([D0[:, :, h:h + 1] for h in range(24)])           # [Z, D, 1]
    lag["level_1d"] = np.broadcast_to(_shift(day_mean, 1), D0.shape)
    lag["level_7d"] = np.broadcast_to(_nanmean_stack([_shift(day_mean, k) for k in range(1, 8)]), D0.shape)
    offday = hourly.loc[hourly["zone_id"] == 0].groupby("date")["is_offday"].first().to_numpy().astype(bool)
    prev = _prev_same_type_index(offday)
    st = np.full_like(D0, np.nan)
    ok = prev >= 0
    st[:, ok, :] = D0[:, prev[ok], :]
    lag["same_type_lag"] = st

    # Weather features: derived from the *forecast* series, rolling over the flat hourly axis.
    w = hourly.loc[hourly["zone_id"] == 0].sort_values("ts")
    rf = w["rain_fc"].to_numpy()
    pad = np.r_[rf, 0, 0]
    next3 = pad[:-2] + pad[1:-1] + pad[2:]
    prev3 = np.r_[0, 0, 0, rf][:-3] + np.r_[0, 0, rf][:-2] + np.r_[0, rf][:-1]
    ev = _event_features(days, events)

    X = hourly[["hour", "dow", "is_offday", "is_holiday", "is_pre_holiday", "is_post_holiday", "month",
                "temp_fc", "rain_fc", "capacity"]].copy()
    for c in ("is_offday", "is_holiday", "is_pre_holiday", "is_post_holiday"):
        X[c] = X[c].astype(int)
    X["rain_fc_next3"] = np.tile(next3, Z)
    X["rain_fc_prev3"] = np.tile(prev3, Z)
    for k, v in ev.items():
        X[k] = v.reshape(-1)
    zones = C.ZONES
    X["zone_idx"] = hourly["zone_id"].to_numpy()
    X["base_price"] = np.array([zones[i].base_price for i in hourly["zone_id"]])
    X["has_venue"] = np.array([int(zones[i].venue) for i in hourly["zone_id"]])
    for j, a in enumerate(C.ARCHETYPES):
        pv = np.array([zones[i].poi_vector()[j] for i in range(Z)])
        X[f"poi_{a}"] = pv[hourly["zone_id"].to_numpy()]
    for k, v in lag.items():
        X[k] = np.asarray(v).reshape(-1)
    X = X[ALL_FEATURES]

    meta = hourly[["zone_id", "date", "hour", "price_mult", "capacity", "occupancy", "turned_away"]].copy()
    return FeatureSet(X=X, y=d0_flat, meta=meta, d0=D0, days=days)
