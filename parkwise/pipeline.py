"""Offline build: simulate -> estimate elasticity -> forecast -> evaluate -> replay pricing -> tickets.

Run with ``python -m parkwise build``. Produces everything the API serves in ``artifacts/``.
This is the only place (besides tests) that is allowed to look at the hidden ground truth,
and it uses it solely to *grade* the system, never to train or tune it.
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import elasticity as E
from . import features as F
from . import forecast as FC
from . import generator as G
from . import pricing as P
from . import tickets as T

log = logging.getLogger("parkwise.pipeline")
Z = C.N_ZONES


def _clean(o):
    """Make an object JSON-safe (numpy scalars -> python, NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def _masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    d = meta["date"]
    warm = d >= pd.Timestamp(C.DATA_START) + pd.Timedelta(days=C.WARMUP_DAYS)
    return {
        "train": (warm & (d <= C.TRAIN_END)).to_numpy(),
        "val": ((d > C.TRAIN_END) & (d <= C.VAL_END)).to_numpy(),
        "test": ((d > C.VAL_END) & (d <= C.DATA_END)).to_numpy(),
        "live": (d > C.DATA_END).to_numpy(),
        "observed": (warm & (d <= C.DATA_END)).to_numpy(),
    }


def _occ_hat(d0_hat: np.ndarray, beta: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    z = meta["zone_id"].to_numpy()
    return np.minimum(1.0, d0_hat * np.exp(beta[z] * (meta["price_mult"].to_numpy() - 1.0)))


def _group_importance(model: FC.ForecastModel, X: pd.DataFrame, y: np.ndarray, rng, n=6000, repeats=3) -> dict:
    idx = rng.choice(np.nonzero(~np.isnan(y))[0], size=min(n, int((~np.isnan(y)).sum())), replace=False)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    base = np.mean(np.abs(model.point.predict(Xs[model.features]) - ys))
    groups = {"History (lags & levels)": F.LAG_FEATURES, "Calendar & holidays": F.CALENDAR_FEATURES,
              "Weather forecast": F.WEATHER_FEATURES, "Event schedule": F.EVENT_FEATURES,
              "Zone profile (POI mix, capacity)": F.ZONE_FEATURES}
    out = {}
    for name, cols in groups.items():
        deltas = []
        for _ in range(repeats):
            Xp = Xs.copy()
            Xp[cols] = Xs[cols].sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy()
            deltas.append(np.mean(np.abs(model.point.predict(Xp[model.features]) - ys)) - base)
        out[name] = float(np.mean(deltas))
    return {"baseline_mae": float(base), "mae_increase_when_shuffled": out}


def _bootstrap_ratio(num: np.ndarray, den: np.ndarray, rng, n=2000) -> tuple[float, float, float]:
    idx = rng.integers(0, len(num), size=(n, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1) - 1.0
    return float(num.sum() / den.sum() - 1.0), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def _replay_policies(p50, naive, lat, beta_hat, offday, days_idx, rng) -> dict:
    """Replay pricing policies on held-out days against the *true* demand environment.

    Two yardsticks: raw revenue, and ``net_value`` = revenue minus the goodwill cost of drivers
    turned away or priced out (what the optimizer actually maximises; the fair basis for regret).
    """
    g = P.Guardrails()
    pol_names = ["status_quo", "flat", "dynamic_model", "dynamic_naive_forecast", "oracle"]
    rows = {k: [] for k in pol_names}
    expected = []
    plans_model = {}
    for i in days_idx:
        off = bool(offday[i])
        plans = {
            "status_quo": C.status_quo_matrix(off), "flat": P.flat_plan(),
            "dynamic_model": P.optimize(p50[:, i], beta_hat, g, is_offday=off),
            "dynamic_naive_forecast": P.optimize(naive[:, i], beta_hat, g, is_offday=off),
            "oracle": P.optimize(lat[:, i], G.TRUE_BETA, g, is_offday=off),
        }
        plans_model[i] = plans["dynamic_model"]
        for k, m in plans.items():
            rows[k].append(P.kpis(P.simulate(lat[:, i], G.TRUE_BETA, m, G.TRUE_SPILL_SHARE), m, guardrails=g))
        m = plans["dynamic_model"]
        expected.append(P.kpis(P.simulate(p50[:, i], beta_hat, m, 0.25), m, guardrails=g))

    out = {"guardrails": g.as_dict(), "n_days": len(days_idx), "policies": {}}
    sq = {c: np.array([r[c] for r in rows["status_quo"]]) for c in ("revenue", "net_value")}
    for k in pol_names:
        df = pd.DataFrame(rows[k])
        up_r = _bootstrap_ratio(df["revenue"].to_numpy(), sq["revenue"], rng)
        up_n = _bootstrap_ratio(df["net_value"].to_numpy(), sq["net_value"], rng)
        out["policies"][k] = {**{c: float(df[c].mean()) for c in df.columns},
                              "revenue_uplift_vs_status_quo": up_r[0], "uplift_ci95": [up_r[1], up_r[2]],
                              "net_value_uplift_vs_status_quo": up_n[0], "net_uplift_ci95": [up_n[1], up_n[2]],
                              "daily_uplift": (df["revenue"].to_numpy() / sq["revenue"] - 1.0).tolist()}
    exp = pd.DataFrame(expected)
    real = pd.DataFrame(rows["dynamic_model"])
    out["expected_vs_realized"] = {
        "expected_revenue_uplift": float(exp["revenue"].sum() / sq["revenue"].sum() - 1),
        "realized_revenue_uplift": float(real["revenue"].sum() / sq["revenue"].sum() - 1),
        "revenue_gap_pct": float(real["revenue"].sum() / exp["revenue"].sum() - 1)}
    o, m_ = out["policies"]["oracle"], out["policies"]["dynamic_model"]
    out["share_of_oracle_net_gain"] = float((m_["net_value"] - out["policies"]["status_quo"]["net_value"]) /
                                            (o["net_value"] - out["policies"]["status_quo"]["net_value"]))

    def totals(plan_fn, spill):
        r = n = rs = ns = 0.0
        for i in days_idx:
            off = bool(offday[i])
            m = plan_fn(i, off)
            k1 = P.kpis(P.simulate(lat[:, i], G.TRUE_BETA, m, spill), m, guardrails=g)
            sqm = C.status_quo_matrix(off)
            k0 = P.kpis(P.simulate(lat[:, i], G.TRUE_BETA, sqm, spill), sqm, guardrails=g)
            r, n, rs, ns = r + k1["revenue"], n + k1["net_value"], rs + k0["revenue"], ns + k0["net_value"]
        return {"revenue": float(r / rs - 1), "net_value": float(n / ns - 1)}

    # Sensitivity 1: does the gain survive different real-world spillover behaviour?
    out["sensitivity_true_spill_share"] = {str(s): totals(lambda i, off: plans_model[i], s) for s in (0.0, 0.25, 0.5)}
    # Sensitivity 2: what if our elasticity estimates are wrong by a common factor?
    out["sensitivity_elasticity_misspecification"] = {
        str(f): totals(lambda i, off, f=f: P.optimize(p50[:, i], beta_hat * f, g, is_offday=off), G.TRUE_SPILL_SHARE)
        for f in (0.6, 0.8, 1.0, 1.25, 1.5)}
    return out


def build(seed: int = 7, out_dir=C.ARTIFACTS) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def step(msg):
        log.info("[%5.1fs] %s", time.time() - t0, msg)

    step("simulating city")
    sim = G.generate(seed)
    hourly = sim.hourly.sort_values(["zone_id", "ts"]).reset_index(drop=True)
    D_all = hourly["date"].nunique()

    step("estimating price sensitivity from randomized experiments")
    est_eval = E.estimate(hourly, C.VAL_END)
    est_prod = E.estimate(hourly, C.DATA_END)
    beta_eval, beta_prod = est_eval["beta"].to_numpy(), est_prod["beta"].to_numpy()

    step("building features + training forecaster")
    fs = F.build(hourly, sim.events, beta_eval)
    mk = _masks(fs.meta)
    model = FC.fit(fs.X, fs.y, mk["train"], mk["val"])
    pred = model.predict(fs.X)
    y, te = fs.y, mk["test"]
    naive, hist = FC.baseline_seasonal_naive(fs.X), FC.baseline_hist_avg(fs.X)

    step("evaluating on held-out test days")
    latent = hourly["latent_d0"].to_numpy()
    occ_true = fs.meta["occupancy"].to_numpy()
    p50 = pred["p50"].to_numpy()
    metrics = {
        "point": {"gbm": FC.regression_metrics(y[te], p50[te]),
                  "seasonal_naive": FC.regression_metrics(y[te], naive[te]),
                  "hist_avg_4w": FC.regression_metrics(y[te], hist[te])},
        "occupancy": {k: FC.regression_metrics(occ_true[te], _occ_hat(v, beta_eval, fs.meta)[te])
                      for k, v in (("gbm", p50), ("seasonal_naive", naive), ("hist_avg_4w", hist))},
        "full_lot_detection": {k: FC.full_lot_detection(occ_true[te], _occ_hat(v, beta_eval, fs.meta)[te])
                               for k, v in (("gbm", p50), ("seasonal_naive", naive))},
        "interval": FC.interval_metrics(y[te], pred["p10"].to_numpy()[te], pred["p90"].to_numpy()[te]),
        "error_vs_truth_mae": float(np.nanmean(np.abs(p50[te] - latent[te]))),
        "poisson_noise_floor_mae": float(np.nanmean(np.abs(y[te] - latent[te]))),
        "model": {"n_iter": model.n_iter, "cqr_adjustment": model.cqr_adj, "n_features": len(model.features),
                  "train_rows": int((mk["train"] & ~np.isnan(y)).sum()), "val_rows": int((mk["val"] & ~np.isnan(y)).sum()),
                  "test_rows": int((te & ~np.isnan(y)).sum())},
    }
    m = fs.meta
    zid = m["zone_id"].to_numpy()
    metrics["by_zone"] = [{"zone_id": z.id, "zone": z.name,
                           "gbm_wmape": FC.regression_metrics(y[te & (zid == z.id)], p50[te & (zid == z.id)])["wmape"],
                           "naive_wmape": FC.regression_metrics(y[te & (zid == z.id)], naive[te & (zid == z.id)])["wmape"],
                           "coverage": FC.interval_metrics(y[te & (zid == z.id)], pred["p10"].to_numpy()[te & (zid == z.id)],
                                                           pred["p90"].to_numpy()[te & (zid == z.id)])["coverage"]}
                          for z in C.ZONES]
    metrics["by_hour"] = [{"hour": h, "gbm_mae": FC.regression_metrics(y[te & (m['hour'] == h).to_numpy()], p50[te & (m['hour'] == h).to_numpy()])["mae"],
                           "naive_mae": FC.regression_metrics(y[te & (m['hour'] == h).to_numpy()], naive[te & (m['hour'] == h).to_numpy()])["mae"]}
                          for h in range(24)]
    ev_flag = (fs.X["ev_local_rel"].notna() | (fs.X["ev_near_load"] > 0)).to_numpy()
    reg = {"Working days": fs.X["is_offday"].to_numpy() == 0, "Weekends & holidays": fs.X["is_offday"].to_numpy() == 1,
           "Event hours": ev_flag, "Rainy hours": (fs.X["rain_fc"] > 1).to_numpy()}
    metrics["by_regime"] = [{"regime": k, "gbm_wmape": FC.regression_metrics(y[te & v], p50[te & v])["wmape"],
                             "naive_wmape": FC.regression_metrics(y[te & v], naive[te & v])["wmape"],
                             "n": int((te & v & ~np.isnan(y)).sum())} for k, v in reg.items()]

    step("ablations")
    base_w = FC.regression_metrics(y[te], FC.fit_fixed(fs.X, y, mk["train"] | mk["val"], F.ALL_FEATURES, model.n_iter)
                                   .predict(fs.X[F.ALL_FEATURES])[te])["wmape"]
    ablate = {"Full model": base_w}
    for name, drop in (("No history features", F.LAG_FEATURES), ("No event schedule", F.EVENT_FEATURES),
                       ("No weather forecast", F.WEATHER_FEATURES),
                       ("No holiday flags", ["is_holiday", "is_pre_holiday", "is_post_holiday"])):
        feats = [f for f in F.ALL_FEATURES if f not in drop]
        mdl = FC.fit_fixed(fs.X, y, mk["train"] | mk["val"], feats, model.n_iter)
        ablate[name] = FC.regression_metrics(y[te], mdl.predict(fs.X[feats])[te])["wmape"]
    metrics["ablation_wmape"] = ablate
    metrics["permutation_importance"] = _group_importance(model, fs.X.loc[te].reset_index(drop=True), y[te], rng)

    step("cold-start (leave-one-zone-out)")
    cold = []
    fit_mask = mk["train"] | mk["val"]
    test_dates = np.sort(fs.meta.loc[te, "date"].unique())
    adapt_days = set(test_dates[:7])                          # first week of the new lot is observed
    is_adapt = fs.meta["date"].isin(adapt_days).to_numpy()
    poi_mat = np.stack([zz.poi_vector() for zz in C.ZONES])
    for z in C.ZONES:
        in_z = zid == z.id
        mdl = FC.fit_fixed(fs.X, y, fit_mask & ~in_z, F.COLD_START_FEATURES, 250)
        p = mdl.predict(fs.X[F.COLD_START_FEATURES])
        prof = (pd.DataFrame({"y": y, "off": fs.X["is_offday"], "h": fs.X["hour"]})[fit_mask & ~in_z & ~np.isnan(y)]
                .groupby(["off", "h"])["y"].mean())
        blind = prof.reindex(pd.MultiIndex.from_arrays([fs.X["is_offday"][in_z], fs.X["hour"][in_z]])).to_numpy()
        blind_full = np.full(len(y), np.nan)
        blind_full[in_z] = blind
        rest = te & in_z & ~is_adapt
        ratio = np.nansum(y[te & in_z & is_adapt]) / np.nansum(p[te & in_z & is_adapt])
        dom = int(np.argmax(poi_mat[z.id]))
        covered = bool(np.delete(poi_mat[:, dom], z.id).max() >= 0.35) and not z.venue
        cold.append({"zone_id": z.id, "zone": z.name, "dominant_archetype": C.ARCHETYPES[dom], "archetype_seen_elsewhere": covered,
                     "cold_start_wmape": FC.regression_metrics(y[rest], p[rest])["wmape"],
                     "cold_start_recalibrated_wmape": FC.regression_metrics(y[rest], p[rest] * ratio)["wmape"],
                     "network_average_wmape": FC.regression_metrics(y[rest], blind_full[rest])["wmape"],
                     "full_model_wmape": FC.regression_metrics(y[rest], p50[rest])["wmape"]})
    metrics["cold_start"] = cold

    step("production refit on all observed data")
    fs_prod = F.build(hourly, sim.events, beta_prod)
    prod = FC.refit(model, fs_prod.X, fs_prod.y, mk["observed"])
    pred_prod = prod.predict(fs_prod.X)

    # Materialise forecasts for backtest days (eval model) and the live day (prod model).
    days = fs.days
    d_test = [i for i, d in enumerate(days) if C.VAL_END < d.strftime("%Y-%m-%d") <= C.DATA_END]
    d_live = len(days) - 1
    d_fc = d_test + [d_live]

    def cube(a):
        return np.array(a, dtype=float).reshape(Z, D_all, 24)   # copy: pandas 3 hands out read-only views

    P10, P50, P90 = cube(pred["p10"]), cube(pred["p50"]), cube(pred["p90"])
    P10[:, d_live], P50[:, d_live], P90[:, d_live] = (cube(pred_prod[c])[:, d_live] for c in ("p10", "p50", "p90"))
    NAIVE = cube(naive)
    LAT = cube(latent)
    offday = hourly[hourly["zone_id"] == 0].groupby("date")["is_offday"].first().to_numpy()

    step("replaying pricing policies on held-out days")
    policy = _replay_policies(P50, NAIVE, LAT, beta_eval, offday, d_test, rng)

    step("tickets: train classifier + simulate stream")
    tk = T.train_and_evaluate()
    stream = T.generate_stream(hourly)
    proba = tk.model.predict_proba(stream["text"])
    stream["pred_category"] = tk.model.clf.classes_[np.argmax(proba, axis=1)]
    stream["confidence"] = proba.max(axis=1)
    tk.metrics["stream_accuracy_vs_true_category"] = float((stream["pred_category"] == stream["true_category"]).mean())
    voc = _voc(stream, hourly)

    step("writing artifacts")
    zones_json = [{"id": z.id, "slug": z.slug, "name": z.name, "lat": z.lat, "lon": z.lon, "capacity": z.capacity,
                   "base_price": z.base_price, "poi": z.poi, "venue": z.venue} for z in C.ZONES]
    keep = [c for c in hourly.columns if c != "latent_d0"]
    hourly[keep].to_csv(out_dir / "hourly.csv.gz", index=False)
    sim.events.to_csv(out_dir / "events.csv", index=False)
    stream.to_csv(out_dir / "tickets.csv.gz", index=False)
    fc_days = [days[i].strftime("%Y-%m-%d") for i in d_fc]
    obs = {k: cube(hourly[k])[:, d_fc] for k in ("occupancy", "price_mult", "turned_away", "revenue", "demand_ratio")}
    np.savez_compressed(
        out_dir / "forecasts.npz", days=np.array(fc_days), p10=P10[:, d_fc], p50=P50[:, d_fc], p90=P90[:, d_fc],
        naive=NAIVE[:, d_fc], beta_eval=beta_eval, beta_prod=beta_prod, is_offday=offday[d_fc],
        **{f"obs_{k}": v for k, v in obs.items()})
    np.savez_compressed(out_dir / "truth.npz", latent=LAT[:, d_fc], beta=G.TRUE_BETA, spill=G.TRUE_SPILL_SHARE)
    joblib.dump({"eval": model, "prod": prod}, out_dir / "forecaster.joblib", compress=3)
    joblib.dump(tk.model, out_dir / "ticket_classifier.joblib", compress=3)

    elast = est_eval.merge(est_prod[["zone_id", "beta", "se"]].rename(columns={"beta": "beta_prod", "se": "se_prod"}), on="zone_id")
    elast["true_beta"] = G.TRUE_BETA
    report = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(), "seed": seed,
        "data": {"start": C.DATA_START, "end": C.DATA_END, "live_date": C.LIVE_DATE, "train_end": C.TRAIN_END,
                 "val_end": C.VAL_END, "rows": int(len(hourly)), "zones": Z,
                 "experiment_share": float(hourly["is_experiment"].mean()),
                 "sensor_gap_share": float((~hourly["sensor_ok"]).mean()),
                 "full_hour_share": float((hourly["occupancy"] >= 0.99).mean()), "events": int(len(sim.events))},
        "forecast": metrics, "elasticity": elast.to_dict("records"), "policy": policy,
        "tickets": {**tk.metrics, "voc": voc},
        "assumptions": [
            "All data is synthetic. Known ground truth lets us grade the estimator, forecaster and pricing policy, which real data cannot.",
            "Weather 'forecasts' are the true weather plus realistic error (misses, false alarms, magnitude noise).",
            "Price experiments are randomized per zone-day; unconstrained demand = occupied + drivers turned away (an app can log this).",
            "Spillover to neighbouring lots is a modelling assumption (25% by default); results are shown across 0-50%.",
            "Pricing replay is an expected-value simulation against the hidden true demand, not a live A/B test.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(_clean(report), indent=1))
    (out_dir / "zones.json").write_text(json.dumps(zones_json, indent=1))
    step("done")
    return report


def _voc(stream: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    end = pd.Timestamp(C.DATA_END)
    parked = hourly.groupby(["zone_id", "date"])["occupied"].sum().reset_index()
    cur = (stream["ts"] > end - pd.Timedelta(days=28)) & (stream["ts"] <= end + pd.Timedelta(days=1))
    prv = (stream["ts"] > end - pd.Timedelta(days=56)) & (stream["ts"] <= end - pd.Timedelta(days=28))
    pk_c = parked[(parked["date"] > end - pd.Timedelta(days=28))].groupby("zone_id")["occupied"].sum()
    pk_p = parked[(parked["date"] > end - pd.Timedelta(days=56)) & (parked["date"] <= end - pd.Timedelta(days=28))].groupby("zone_id")["occupied"].sum()
    by_zone = []
    for z in C.ZONES:
        row = {"zone_id": z.id, "zone": z.name, "current": {}, "previous": {}}
        for c in T.CATEGORIES:
            n_c = int(((stream["zone_id"] == z.id) & cur & (stream["pred_category"] == c)).sum())
            n_p = int(((stream["zone_id"] == z.id) & prv & (stream["pred_category"] == c)).sum())
            row["current"][c] = n_c / max(pk_c.get(z.id, 1), 1) * 10000
            row["previous"][c] = n_p / max(pk_p.get(z.id, 1), 1) * 10000
        by_zone.append(row)
    wk = (stream.assign(week=stream["ts"].dt.to_period("W").dt.start_time)
          .groupby(["week", "pred_category"]).size().unstack(fill_value=0).reindex(columns=T.CATEGORIES, fill_value=0))
    wk = wk.iloc[1:-1]
    # Guardrail evidence: overcharge complaints vs. the day's experimental price multiplier.
    zd = hourly.groupby(["zone_id", "date"]).agg(m=("price_mult_exp", "first"), parked=("occupied", "sum")).reset_index()
    oc = (stream[stream["pred_category"] == "overcharge"].assign(date=lambda d: d["ts"].dt.normalize())
          .groupby(["zone_id", "date"]).size().rename("n").reset_index())
    zd = zd.merge(oc, on=["zone_id", "date"], how="left").fillna({"n": 0})
    g = zd.groupby("m").agg(n=("n", "sum"), parked=("parked", "sum"))
    price_curve = [{"multiplier": float(k), "per_10k_parked_hours": float(v.n / v.parked * 10000)} for k, v in g.iterrows()]
    # Complaint model: overcharge tickets per zone-day ~ a * parked hours + b * price-shock exposure, where
    # shock = sum over hours of max(multiplier - 1.05, 0) * vehicles. Lets the studio project complaints.
    h = hourly[hourly["sensor_ok"]].assign(shock=lambda d: np.clip(d["price_mult"] - 1.05, 0, None) * d["occupied"])
    zs = h.groupby(["zone_id", "date"]).agg(parked=("occupied", "sum"), shock=("shock", "sum")).reset_index()
    zs = zs.merge(oc, on=["zone_id", "date"], how="left").fillna({"n": 0})
    coef, *_ = np.linalg.lstsq(zs[["parked", "shock"]].to_numpy(), zs["n"].to_numpy(), rcond=None)
    complaint_model = {"per_parked_hour": float(coef[0]), "per_shock_unit": float(coef[1]),
                       "note": "OLS on zone-days: predicted overcharge tickets = a*parked + b*sum(max(m-1.05,0)*vehicles)"}
    recent = stream.sort_values("ts").tail(40)
    return {"by_zone": by_zone, "complaint_model": complaint_model,
            "weekly": {"weeks": [w.strftime("%Y-%m-%d") for w in wk.index], "categories": T.CATEGORIES, "counts": wk.to_numpy().tolist()},
            "overcharge_vs_price": price_curve, "labels": T.LABELS,
            "recent": [{"ts": r.ts.isoformat(), "zone_id": int(r.zone_id), "text": r.text, "pred": r.pred_category,
                        "confidence": float(r.confidence)} for r in recent.itertuples()][::-1]}
