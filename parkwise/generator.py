"""Synthetic city simulator: the only module that knows the hidden ground truth.

Why synthetic? Real parking telemetry is proprietary. A simulator with *known* demand and
price sensitivity lets us do something real data never allows: grade the elasticity estimator,
the forecaster and the pricing policy against the truth.

The data-generating process is deliberately messy in the ways production data is:

* demand is a mixture of archetype profiles (office / retail / nightlife / ...) per zone;
* weekday vs holiday regimes, festival boosts, monsoon rain, slow growth;
* scheduled events (matches, concerts, sales) that spill into neighbouring zones;
* persistent day-level shocks, autocorrelated hourly noise, Poisson arrival noise, and two
  permanent level shifts (a new office tower, a competitor opening) that a static model misses;
* **censoring**: occupancy is capped at capacity, so full lots hide true demand. We log
  ``turned_away`` (drivers who searched and found the lot full), as a real app could;
* randomized **price experiments** on ~12% of zone-days, which is what makes price
  sensitivity identifiable;
* sensor outages (missing hourly readings) and imperfect weather forecasts.

Nothing outside ``evaluate.py`` and the tests may import ``GROUND_TRUTH``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------------------
# Hidden ground truth (never read by modelling code)
# --------------------------------------------------------------------------------------
# beta: semi-elasticity of demand to the price multiplier m:  demand(m) = demand(1) * exp(beta*(m-1)).
# Elasticity at multiplier m is beta*m, so sensitivity grows as price rises (realistic) and
# revenue-maximising multipliers are interior (m* = 1/|beta|) instead of "always raise".
# scale: peak demand at base price as a multiple of capacity (>1 means the lot overflows).
GROUND_TRUTH = {
    #       id: (beta,  scale)
    0:  (-0.90, 1.25),   # Indiranagar
    1:  (-1.00, 1.35),   # Koramangala
    2:  (-0.60, 1.05),   # MG Road
    3:  (-0.35, 0.85),   # Whitefield
    4:  (-0.30, 0.90),   # Manyata
    5:  (-0.45, 1.10),   # Majestic
    6:  (-0.80, 1.15),   # Mall
    7:  (-1.10, 1.00),   # Jayanagar
    8:  (-0.50, 0.35),   # Stadium (low baseline; events drive it)
    9:  (-0.25, 0.90),   # Hebbal
    10: (-0.30, 0.80),   # Airport
    11: (-0.70, 0.75),   # Sarjapur
}
TRUE_BETA = np.array([GROUND_TRUTH[z.id][0] for z in C.ZONES])
TRUE_SCALE = np.array([GROUND_TRUTH[z.id][1] for z in C.ZONES])
TRUE_SPILL_SHARE = 0.25  # share of displaced drivers who try a nearby lot (used by policy replay)


def _profile(points: dict[int, float]) -> np.ndarray:
    hs = sorted(points)
    xs = np.array(hs + [24.0])
    ys = np.array([points[h] for h in hs] + [points[hs[0]]])
    return np.interp(np.arange(24), xs, ys)


# Hourly demand profiles (fraction of that archetype's peak) for weekday / off-day.
_PROFILES = {
    "office": (
        _profile({0: .05, 6: .06, 7: .25, 8: .65, 9: .95, 10: 1.0, 16: 1.0, 17: .9, 18: .6, 19: .3, 20: .15, 21: .08}),
        _profile({0: .05, 9: .08, 12: .12, 18: .08, 23: .05}),
    ),
    "retail": (
        _profile({0: .03, 8: .03, 10: .30, 12: .50, 14: .55, 17: .65, 19: .85, 20: .80, 21: .55, 22: .20, 23: .05}),
        _profile({0: .04, 9: .05, 10: .40, 12: .75, 15: .90, 18: 1.0, 20: 1.0, 21: .80, 22: .40, 23: .12}),
    ),
    "nightlife": (
        _profile({0: .40, 1: .15, 2: .03, 11: .05, 12: .20, 13: .35, 15: .15, 18: .40, 20: .85, 21: 1.0, 22: .95, 23: .70}),
        _profile({0: .85, 1: .50, 2: .20, 3: .04, 11: .10, 12: .40, 13: .60, 16: .30, 19: .80, 21: 1.0, 23: 1.0}),
    ),
    "residential": (
        _profile({0: .95, 6: .95, 7: .80, 8: .55, 9: .40, 16: .35, 17: .50, 18: .70, 19: .85, 20: .95, 21: 1.0}),
        _profile({0: .95, 8: .95, 9: .80, 11: .80, 12: .65, 17: .65, 18: .80, 19: .95}),
    ),
    "transit": (
        _profile({0: .20, 5: .20, 6: .45, 7: .80, 8: 1.0, 17: .95, 18: .85, 19: .70, 20: .50, 21: .35, 22: .28, 23: .22}),
        _profile({0: .25, 6: .30, 10: .50, 18: .50, 21: .35, 23: .28}),
    ),
    "medical": (
        _profile({0: .20, 6: .20, 7: .40, 8: .70, 9: .95, 12: 1.0, 13: .85, 14: .90, 16: .90, 17: .75, 18: .50, 19: .40, 20: .30, 23: .22}),
        _profile({0: .22, 7: .25, 9: .65, 12: .65, 16: .55, 19: .35, 23: .22}),
    ),
    "travel": (
        _profile({0: .65, 3: .62, 5: .85, 7: .85, 10: .70, 13: .55, 15: .60, 18: .85, 21: .85, 23: .72}),
        _profile({0: .70, 3: .68, 5: .90, 7: .92, 10: .78, 13: .62, 15: .68, 18: .92, 21: .92, 23: .78}),
    ),
}
# Rain effect on demand per archetype at >=4 mm/h (fractional change).
_RAIN_EFFECT = {"office": +0.05, "retail": -0.30, "nightlife": -0.20, "residential": +0.05,
                "transit": +0.12, "medical": 0.0, "travel": 0.0}
_HOLIDAY_ARCH = {"retail": 1.20, "nightlife": 1.15}

_MATCH_KERNEL_X = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
_MATCH_KERNEL_Y = np.array([0, .15, .40, .80, 1.0, 1.0, .9, .8, .7, .5, .25, .10, 0], dtype=float)


@dataclass
class SimData:
    hourly: pd.DataFrame     # one row per zone-hour (incl. the live day with NaN outcomes)
    events: pd.DataFrame     # scheduled events (known in advance by the platform)
    days: pd.DatetimeIndex


def _calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    hol = pd.Series({pd.Timestamp(k): v for k, v in C.HOLIDAYS.items()})
    is_hol = days.isin(hol.index)
    offday = (days.dayofweek >= 5) | is_hol
    nxt = pd.Series(is_hol, index=days).shift(-1, fill_value=False).to_numpy()
    prv = pd.Series(is_hol, index=days).shift(1, fill_value=False).to_numpy()
    return pd.DataFrame({"dow": days.dayofweek, "month": days.month, "is_holiday": is_hol,
                         "is_offday": offday, "is_pre_holiday": nxt, "is_post_holiday": prv}, index=days)


def _weather(days: pd.DatetimeIndex, rng: np.random.Generator) -> dict[str, np.ndarray]:
    D = len(days)
    doy = days.dayofyear.to_numpy()
    base = 27 + 4.5 * np.sin(2 * np.pi * (doy - 100) / 365)
    daily_noise = rng.normal(0, 1.0, D)
    hours = np.arange(24)
    temp = base[:, None] + daily_noise[:, None] + 4 * np.sin(2 * np.pi * (hours[None, :] - 9) / 24)
    p_rain = np.select([np.isin(days.month, [6, 7, 8, 9]), np.isin(days.month, [10, 11])], [0.55, 0.35], 0.08)
    rain = np.zeros((D, 24))
    for d in range(D):
        if rng.random() < p_rain[d]:
            dur = int(rng.integers(2, 6))
            start = int(rng.choice(np.arange(12, 22), p=np.array([1, 2, 3, 3, 3, 3, 2, 2, 1, 1]) / 21))
            for k in range(dur):
                h = start + k
                if h < 24:
                    rain[d, h] = rng.exponential(3.0) * (1.0 if 0 < k < dur - 1 else 0.5)
    # Imperfect forecasts: multiplicative error, some misses and false alarms.
    fc = rain * np.exp(rng.normal(0, 0.4, rain.shape))
    fc[(rain > 0) & (rng.random(rain.shape) < 0.15)] = 0.0
    false_alarm = (rain == 0) & (rng.random(rain.shape) < 0.01)
    fc[false_alarm] = rng.uniform(1, 3, false_alarm.sum())
    return {"temp": temp, "temp_fc": temp + rng.normal(0, 1.0, temp.shape), "rain": rain, "rain_fc": fc}


def _events(days: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    venue = next(z for z in C.ZONES if z.venue)
    months = pd.period_range(days[0], days[-1], freq="M")
    for m in months:
        mdays = days[(days.to_period("M") == m)]
        n = rng.poisson(2.6)
        for _ in range(n):
            wk = mdays[mdays.dayofweek >= 5]
            pick_weekend = rng.random() < 0.6 and len(wk) > 0
            day = rng.choice(wk if pick_weekend else mdays)
            start = int(rng.choice([15, 19])) if pick_weekend else 19
            size = float(rng.choice([0.45, 0.85, 1.4], p=[.4, .4, .2]))
            kind = "match" if rng.random() < 0.65 else "concert"
            s = pd.Timestamp(day) + pd.Timedelta(hours=start)
            rows.append((venue.id, s, s + pd.Timedelta(hours=4), size, kind))
        for z_slug, zid in (("rajajinagar", 6), ("jayanagar", 7)):
            if rng.random() < 0.8 and len(mdays):
                wk = mdays[mdays.dayofweek >= 5]
                day = rng.choice(wk) if len(wk) else rng.choice(mdays)
                s = pd.Timestamp(day) + pd.Timedelta(hours=11)
                rows.append((zid, s, s + pd.Timedelta(hours=10), 0.3, "sale"))
    ev = pd.DataFrame(rows, columns=["zone_id", "start", "end", "size", "kind"])
    ev = ev.sort_values("start").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(len(ev)))
    return ev


def _event_load(days, events, rng) -> np.ndarray:
    """True additive demand (fraction of capacity) from events, shape [Z, T] on the flat hourly axis."""
    T = len(days) * 24
    out = np.zeros((C.N_ZONES, T))
    dist = C.distance_matrix_km()
    t0 = days[0]
    for e in events.itertuples():
        att = float(np.exp(rng.normal(0, 0.25)))  # real attendance differs from the plan
        s_idx = int((e.start - t0) / pd.Timedelta(hours=1))
        if e.kind == "sale":
            taus = np.arange(-1, 11)
            ker = np.r_[0.5, np.ones(10), 0.5]
            amp_local, amp_near = 0.8, 0.0
        else:
            taus = np.arange(-5, 8)
            ker = _MATCH_KERNEL_Y
            amp_local, amp_near = 1.1, 0.28
        for z in range(C.N_ZONES):
            if z == e.zone_id:
                amp = amp_local
            elif amp_near > 0:
                amp = amp_near * np.exp(-dist[z, e.zone_id] / 1.5)
            else:
                continue
            for tau, k in zip(taus, ker):
                i = s_idx + int(tau)
                if 0 <= i < T:
                    out[z, i] += amp * e.size * att * k
    return out


def _ar1(rng, shape, phi=0.85, sigma=0.035):
    eps = rng.normal(0, sigma, shape)
    out = np.zeros(shape)
    out[..., 0] = eps[..., 0] / np.sqrt(1 - phi ** 2)
    for t in range(1, shape[-1]):
        out[..., t] = phi * out[..., t - 1] + eps[..., t]
    return out


def generate(seed: int = 7) -> SimData:
    rng = np.random.default_rng(seed)
    days = pd.date_range(C.DATA_START, C.LIVE_DATE, freq="D")
    D, Z = len(days), C.N_ZONES
    T = D * 24
    cal = _calendar(days)
    wx = _weather(days, rng)
    events = _events(days, rng)

    # --- base demand at multiplier 1.0 (fraction of capacity) ------------------------------
    poi = np.stack([z.poi_vector() for z in C.ZONES])                    # [Z, A]
    prof = np.stack([np.stack(_PROFILES[a]) for a in C.ARCHETYPES])      # [A, 2, 24]
    mix = np.einsum("za,adh->zdh", poi, prof)                            # [Z, 2, 24]
    mix = mix / mix.max(axis=(1, 2), keepdims=True)
    offday = cal["is_offday"].to_numpy().astype(int)
    base = mix[:, offday, :] * TRUE_SCALE[:, None, None]                 # [Z, D, 24]

    hol = cal["is_holiday"].to_numpy()
    hol_arch = np.array([_HOLIDAY_ARCH.get(a, 1.0) for a in C.ARCHETYPES])
    hol_z = (poi * hol_arch).sum(1) / poi.sum(1)                         # [Z]
    base = base * np.where(hol[None, :, None], hol_z[:, None, None], 1.0)

    rain_eff = np.array([_RAIN_EFFECT[a] for a in C.ARCHETYPES])
    z_rain = (poi * rain_eff).sum(1) / poi.sum(1)                        # [Z]
    rain_f = np.minimum(1.0, wx["rain"] / 4.0)                           # [D, 24]
    base = base * (1 + z_rain[:, None, None] * rain_f[None])
    hot = np.clip(wx["temp"] - 28, 0, None)                              # mild heat effect on discretionary trips
    disc = (poi[:, [1, 2]].sum(1))[:, None, None]
    base = base * (1 - 0.006 * disc * hot[None])

    season = np.where(np.isin(days.month, [10, 11]), 1.06, np.where(days.month == 12, 1.10, 1.0))
    growth = 1 + 0.12 * np.arange(D) / 365.0
    base = base * (season * growth)[None, :, None]

    # Day-level shocks persist for a few days (AR(1) across days), and two zones undergo permanent
    # level shifts: a new office tower fills up near Whitefield, a competitor opens near Sarjapur.
    phi, sd = 0.7, 0.07
    day_eff = np.zeros((Z, D))
    day_eff[:, 0] = rng.normal(0, sd, Z)
    for d in range(1, D):
        day_eff[:, d] = phi * day_eff[:, d - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2), Z)
    shift = np.ones((Z, D))
    shift[3, days >= pd.Timestamp("2026-03-01")] = 1.18
    shift[11, days >= pd.Timestamp("2026-06-15")] = 0.85
    base = base * shift[:, :, None]
    hr_eff = _ar1(rng, (Z, T)).reshape(Z, D, 24)
    ev = _event_load(days, events, rng).reshape(Z, D, 24)
    latent_d0 = np.clip(base * np.exp(day_eff[:, :, None] + hr_eff) + ev, 0.01, None)

    # --- prices: static schedule x randomized experiments ----------------------------------
    sq = np.stack([C.status_quo_matrix(bool(o)) for o in offday], axis=1)  # [Z, D, 24]
    exp_mult = np.ones((Z, D))
    in_exp = (rng.random((Z, D)) < 0.12)
    exp_mult[in_exp] = rng.choice([0.8, 0.9, 1.1, 1.25], size=in_exp.sum())
    in_exp[:, -1] = False
    exp_mult[:, -1] = 1.0                                                  # nothing is known about the live day
    m_total = sq * exp_mult[:, :, None]

    # --- outcomes -------------------------------------------------------------------------
    cap = np.array([z.capacity for z in C.ZONES], dtype=float)[:, None, None]
    lam = cap * latent_d0 * np.exp(TRUE_BETA[:, None, None] * (m_total - 1.0))
    arrivals = rng.poisson(lam).astype(float)
    occupied = np.minimum(arrivals, cap)
    turned = np.maximum(arrivals - cap, 0.0)
    price = np.array([z.base_price for z in C.ZONES])[:, None, None] * m_total
    revenue = occupied * price

    ok = np.ones((Z, D, 24), dtype=bool)
    for z in range(Z):                                                     # sensor outage blocks
        for _ in range(rng.poisson(6)):
            s = int(rng.integers(0, T - 12))
            ok.reshape(Z, T)[z, s:s + int(rng.integers(3, 13))] = False
    live = np.zeros((Z, D, 24), dtype=bool)
    live[:, -1, :] = True
    hide = (~ok) | live
    for arr in (arrivals, occupied, turned, revenue):
        arr[hide] = np.nan
    latent_out = latent_d0.copy()
    latent_out[live] = np.nan

    ts = (days.to_numpy()[:, None] + (np.arange(24) * np.timedelta64(1, "h"))[None, :])  # [D, 24]
    zid = np.repeat(np.arange(Z), D * 24)
    df = pd.DataFrame({
        "zone_id": zid,
        "ts": np.tile(ts.reshape(-1), Z),
        "date": np.tile(np.repeat(days.to_numpy(), 24), Z),
        "hour": np.tile(np.tile(np.arange(24), D), Z),
        "dow": np.tile(np.repeat(cal["dow"].to_numpy(), 24), Z),
        "month": np.tile(np.repeat(cal["month"].to_numpy(), 24), Z),
        "is_holiday": np.tile(np.repeat(cal["is_holiday"].to_numpy(), 24), Z),
        "is_offday": np.tile(np.repeat(cal["is_offday"].to_numpy(), 24), Z),
        "is_pre_holiday": np.tile(np.repeat(cal["is_pre_holiday"].to_numpy(), 24), Z),
        "is_post_holiday": np.tile(np.repeat(cal["is_post_holiday"].to_numpy(), 24), Z),
        "temp_c": np.tile(wx["temp"].reshape(-1), Z),
        "temp_fc": np.tile(wx["temp_fc"].reshape(-1), Z),
        "rain_mm": np.tile(wx["rain"].reshape(-1), Z),
        "rain_fc": np.tile(wx["rain_fc"].reshape(-1), Z),
        "capacity": np.repeat(cap.ravel(), D * 24),
        "price_mult_sq": sq.reshape(-1),
        "price_mult_exp": np.repeat(exp_mult.reshape(-1), 24),
        "price_mult": m_total.reshape(-1),
        "is_experiment": np.repeat(in_exp.reshape(-1), 24),
        "price": price.reshape(-1),
        "arrivals": arrivals.reshape(-1),
        "occupied": occupied.reshape(-1),
        "turned_away": turned.reshape(-1),
        "revenue": revenue.reshape(-1),
        "sensor_ok": (~hide).reshape(-1),
        "latent_d0": latent_out.reshape(-1),   # HIDDEN truth: only evaluate.py may read this
    })
    df["occupancy"] = df["occupied"] / df["capacity"]
    df["demand_ratio"] = df["arrivals"] / df["capacity"]
    return SimData(hourly=df, events=events, days=days)
