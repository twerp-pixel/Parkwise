"""Day-ahead demand forecasting with calibrated uncertainty.

Model family
------------
Histogram gradient boosting (the same algorithm family as LightGBM) in three flavours that
share features: a point model (squared error) and two quantile models (p10 / p90). The
quantile band is then *conformalized* on a held-out validation window (CQR, Romano et al. 2019)
so the nominal 80% interval actually covers ~80% of outcomes.

Time discipline
---------------
Everything is split chronologically. The number of boosting iterations is chosen on the
validation window, then the model is refit on train+validation with that fixed budget.
Test days are never touched until the final report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as F

_HGB_KW = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
               early_stopping=False, random_state=0)
_MAX_ITER = 500
Q_LO, Q_HI = 0.10, 0.90


def _make(loss: str, n_iter: int, features: list[str], quantile: float | None = None):
    kw = dict(_HGB_KW, max_iter=n_iter, loss=loss)
    if quantile is not None:
        kw["quantile"] = quantile
    cat = [f for f in features if f == "zone_idx"]
    return HistGradientBoostingRegressor(categorical_features=cat or None, **kw)


def _best_iter(model, Xva, yva) -> int:
    errs = [np.mean(np.abs(p - yva)) for p in model.staged_predict(Xva)]
    return int(np.argmin(errs)) + 1


@dataclass
class ForecastModel:
    features: list[str]
    point: HistGradientBoostingRegressor
    lo: HistGradientBoostingRegressor
    hi: HistGradientBoostingRegressor
    cqr_adj: float
    n_iter: int

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        p50 = np.clip(self.point.predict(Xf), 0, None)
        lo = self.lo.predict(Xf) - self.cqr_adj
        hi = self.hi.predict(Xf) + self.cqr_adj
        lo = np.clip(np.minimum(lo, p50), 0, None)
        hi = np.maximum(hi, p50)
        return pd.DataFrame({"p10": lo, "p50": p50, "p90": hi}, index=X.index)


def fit(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray,
        features: list[str] | None = None, quantiles: bool = True) -> ForecastModel:
    """Select iterations + calibrate on validation, then refit on train+validation."""
    features = features or F.ALL_FEATURES
    tr = train_mask & ~np.isnan(y)
    va = val_mask & ~np.isnan(y)
    Xtr, ytr, Xva, yva = X.loc[tr, features], y[tr], X.loc[va, features], y[va]

    probe = _make("squared_error", _MAX_ITER, features).fit(Xtr, ytr)
    n_iter = _best_iter(probe, Xva, yva)

    if quantiles:
        lo_t = _make("quantile", n_iter, features, Q_LO).fit(Xtr, ytr)
        hi_t = _make("quantile", n_iter, features, Q_HI).fit(Xtr, ytr)
        lo_v, hi_v = lo_t.predict(Xva), hi_t.predict(Xva)
        scores = np.maximum(lo_v - yva, yva - hi_v)
        alpha = Q_HI - Q_LO
        k = int(np.ceil((len(scores) + 1) * alpha))
        adj = float(np.sort(scores)[min(k, len(scores)) - 1])
    else:
        adj = 0.0

    both = tr | va
    Xall, yall = X.loc[both, features], y[both]
    point = _make("squared_error", n_iter, features).fit(Xall, yall)
    if quantiles:
        lo = _make("quantile", n_iter, features, Q_LO).fit(Xall, yall)
        hi = _make("quantile", n_iter, features, Q_HI).fit(Xall, yall)
    else:
        lo = hi = point
    return ForecastModel(features, point, lo, hi, adj, n_iter)


def refit(model: ForecastModel, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> ForecastModel:
    """Refit on ``mask`` with the iteration budget and conformal adjustment already selected."""
    m = mask & ~np.isnan(y)
    Xm, ym = X.loc[m, model.features], y[m]
    f = model.features
    point = _make("squared_error", model.n_iter, f).fit(Xm, ym)
    lo = _make("quantile", model.n_iter, f, Q_LO).fit(Xm, ym)
    hi = _make("quantile", model.n_iter, f, Q_HI).fit(Xm, ym)
    return ForecastModel(f, point, lo, hi, model.cqr_adj, model.n_iter)


def fit_fixed(X: pd.DataFrame, y: np.ndarray, mask: np.ndarray, features: list[str], n_iter: int = 200):
    """Plain point model with a fixed budget (used for ablations and cold-start experiments)."""
    m = mask & ~np.isnan(y)
    return _make("squared_error", n_iter, features).fit(X.loc[m, features], y[m])


# ------------------------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------------------------
def baseline_seasonal_naive(X: pd.DataFrame) -> np.ndarray:
    """Same hour, same weekday last week (fall back to yesterday's same hour)."""
    return X["lag_168"].fillna(X["lag_24"]).fillna(0.0).to_numpy()


def baseline_hist_avg(X: pd.DataFrame) -> np.ndarray:
    """Mean of the same hour-of-week over the last four weeks."""
    return X["mean_4w_same_how"].fillna(X["lag_168"]).fillna(X["lag_24"]).fillna(0.0).to_numpy()


# ------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------
def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "wmape": float(np.sum(np.abs(e)) / np.sum(np.abs(y))), "bias": float(np.mean(e)), "n": int(m.sum())}


def full_lot_detection(occ_true: np.ndarray, occ_pred: np.ndarray, true_thr=0.95, pred_thr=0.90) -> dict:
    """Operational metric: can we warn the ops team that a lot will be (nearly) full?"""
    m = ~np.isnan(occ_true) & ~np.isnan(occ_pred)
    t, p = occ_true[m] >= true_thr, occ_pred[m] >= pred_thr
    tp = float(np.sum(t & p))
    return {"precision": tp / max(p.sum(), 1), "recall": tp / max(t.sum(), 1), "positives": int(t.sum())}


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    m = ~np.isnan(y)
    y, lo, hi = y[m], lo[m], hi[m]
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo)),
            "nominal": Q_HI - Q_LO}
