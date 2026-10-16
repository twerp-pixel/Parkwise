"""Serving layer: loads the batch-built artifacts and answers product questions.

Architecture note: forecasts are materialised by the offline ``build`` job (as a real platform
would run a nightly day-ahead batch) and stored as arrays. The online path therefore needs no
heavy ML at request time: it combines a stored forecast with the demand curve, guardrails and
the optimizer, which run in milliseconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import pricing as P

Z = C.N_ZONES


class ArtifactsMissing(RuntimeError):
    pass


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for h, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = h
        if not v and start is not None:
            out.append((start, h - 1))
            start = None
    return out


def _fmt_range(a: int, b: int) -> str:
    return f"{a:02d}:00-{(b + 1) % 24:02d}:00"


class Engine:
    def __init__(self, art_dir: Path | str = C.ARTIFACTS):
        d = Path(art_dir)
        needed = ["report.json", "zones.json", "forecasts.npz", "truth.npz", "hourly.csv.gz"]
        missing = [n for n in needed if not (d / n).exists()]
        if missing:
            raise ArtifactsMissing(f"Missing artifacts {missing} in {d}. Run: python -m parkwise build")
        self.report = json.loads((d / "report.json").read_text())
        self.zones = json.loads((d / "zones.json").read_text())
        f = np.load(d / "forecasts.npz", allow_pickle=False)
        self.days = [str(x) for x in f["days"]]
        self.p10, self.p50, self.p90, self.naive = f["p10"], f["p50"], f["p90"], f["naive"]
        self.beta_eval, self.beta_prod = f["beta_eval"], f["beta_prod"]
        self.is_offday = f["is_offday"].astype(bool)
        self.obs = {k: f[f"obs_{k}"] for k in ("occupancy", "price_mult", "turned_away", "revenue", "demand_ratio")}
        t = np.load(d / "truth.npz", allow_pickle=False)
        self.truth_latent, self.truth_beta, self.truth_spill = t["latent"], t["beta"], float(t["spill"])
        self.live_date = self.report["data"]["live_date"]
        hourly = pd.read_csv(d / "hourly.csv.gz", usecols=["zone_id", "date", "occupied", "capacity", "turned_away", "revenue"],
                             parse_dates=["date"])
        self._daily = (hourly.groupby(["zone_id", "date"]).agg(occupied=("occupied", "sum"), cap=("capacity", "first"),
                                                                turned=("turned_away", "sum"), revenue=("revenue", "sum"))
                       .reset_index())
        cm = self.report["tickets"]["voc"]["complaint_model"]
        self.complaint_a, self.complaint_b = cm["per_parked_hour"], cm["per_shock_unit"]

    # ---------------------------------------------------------------- helpers
    def _idx(self, date: str) -> int:
        if date not in self.days:
            raise KeyError(date)
        return self.days.index(date)

    def _beta(self, i: int) -> np.ndarray:
        return self.beta_prod if self.days[i] == self.live_date else self.beta_eval

    def _scenario(self, i: int, scenario: str) -> np.ndarray:
        return {"p10": self.p10, "p50": self.p50, "p90": self.p90}[scenario][:, i, :]

    def _complaints(self, sim: dict, m: np.ndarray) -> float:
        parked = float(sim["served"].sum())
        shock = float((np.clip(m - 1.05, 0, None) * sim["served_own"]).sum())
        return self.complaint_a * parked + self.complaint_b * shock

    # ---------------------------------------------------------------- metadata
    def dates(self) -> list[dict]:
        return [{"date": d, "kind": "live" if d == self.live_date else "backtest",
                 "has_actuals": d != self.live_date, "is_offday": bool(self.is_offday[i])}
                for i, d in enumerate(self.days)]

    def meta(self) -> dict:
        el = {e["zone_id"]: e for e in self.report["elasticity"]}
        zones = []
        for z in self.zones:
            e = el[z["id"]]
            zones.append({**z, "beta": e["beta_prod"], "beta_ci": [e["beta_prod"] - 1.96 * e["se_prod"], e["beta_prod"] + 1.96 * e["se_prod"]]})
        return {"dates": self.dates(), "zones": zones, "live_date": self.live_date,
                "default_guardrails": P.Guardrails().as_dict(), "data": self.report["data"],
                "assumptions": self.report["assumptions"]}

    def history(self, zone_id: int, days: int = 120) -> dict:
        d = self._daily[(self._daily["zone_id"] == zone_id) & (self._daily["date"] < pd.Timestamp(self.live_date))]
        d = d.sort_values("date").tail(days)
        return {"dates": d["date"].dt.strftime("%Y-%m-%d").tolist(),
                "utilization": (d["occupied"] / (d["cap"] * 24)).round(4).tolist(),
                "turned_away": d["turned"].round(0).tolist(), "revenue": d["revenue"].round(0).tolist()}

    # ---------------------------------------------------------------- forecast
    def forecast(self, zone_id: int, date: str) -> dict:
        i = self._idx(date)
        beta = self._beta(i)[zone_id]
        live = date == self.live_date
        sq = C.status_quo_matrix(bool(self.is_offday[i]))[zone_id]
        m = sq if live else self.obs["price_mult"][zone_id, i]

        def occ(d0):
            return np.minimum(1.0, d0 * np.exp(beta * (m - 1.0)))

        out = {"date": date, "zone_id": zone_id, "zone": self.zones[zone_id]["name"], "is_live": live,
               "price_multiplier": np.round(m, 3).tolist(),
               "p10": np.round(occ(self.p10[zone_id, i]), 4).tolist(), "p50": np.round(occ(self.p50[zone_id, i]), 4).tolist(),
               "p90": np.round(occ(self.p90[zone_id, i]), 4).tolist(),
               "naive": np.round(occ(self.naive[zone_id, i]), 4).tolist(),
               "demand_p50": np.round(self.p50[zone_id, i], 4).tolist()}
        if not live:
            act = self.obs["occupancy"][zone_id, i]
            out["actual"] = [None if np.isnan(v) else round(float(v), 4) for v in act]
            ok = ~np.isnan(act)
            if ok.any():
                p, n = np.array(out["p50"]), np.array(out["naive"])
                out["day_metrics"] = {
                    "mae": float(np.mean(np.abs(p[ok] - act[ok]))), "naive_mae": float(np.mean(np.abs(n[ok] - act[ok]))),
                    "coverage": float(np.mean((act[ok] >= np.array(out["p10"])[ok]) & (act[ok] <= np.array(out["p90"])[ok])))}
        return out

    # ---------------------------------------------------------------- pricing
    def _plan(self, i: int, strategy: str, scenario: str, g: P.Guardrails, locked, adjustments) -> np.ndarray:
        off = bool(self.is_offday[i])
        sq = C.status_quo_matrix(off)
        if strategy == "status_quo":
            return sq
        if strategy == "flat":
            return P.flat_plan()
        if strategy == "custom":
            m = sq.copy()
            for z, a in adjustments.items():
                m[z] = np.clip(sq[z] * (1.0 + a), 0.3, 3.0)
            return m
        return P.optimize(self._scenario(i, scenario), self._beta(i), g, locked=locked, is_offday=off)

    @staticmethod
    def _actions(plan: np.ndarray, sq: np.ndarray) -> list[str]:
        out = []
        for z in range(Z):
            diff = plan[z] - sq[z]
            msgs = []
            for a, b in _runs(diff >= 0.08):
                msgs.append(f"Raise {_fmt_range(a, b)} (up to +{diff[a:b + 1].max():.0%})")
            for a, b in _runs(diff <= -0.08):
                msgs.append(f"Discount {_fmt_range(a, b)} (down to {diff[a:b + 1].min():.0%})")
            out.append("; ".join(msgs[:2]) if msgs else "Hold current schedule")
        return out

    def simulate(self, date: str, strategy: str = "optimized", scenario: str = "p50",
                 guardrails: P.Guardrails | None = None, spill_share: float = 0.25,
                 locked: list[int] | None = None, adjustments: dict[int, float] | None = None) -> dict:
        i = self._idx(date)
        g = (guardrails or P.Guardrails()).validate()
        locked, adjustments = locked or [], adjustments or {}
        d0, beta = self._scenario(i, scenario), self._beta(i)
        sq = C.status_quo_matrix(bool(self.is_offday[i]))
        plan = self._plan(i, strategy, scenario, g, locked, adjustments)

        s_plan, s_sq = P.simulate(d0, beta, plan, spill_share), P.simulate(d0, beta, sq, spill_share)
        k_plan, k_sq = P.kpis(s_plan, plan, guardrails=g), P.kpis(s_sq, sq, guardrails=g)
        c_plan, c_sq = self._complaints(s_plan, plan), self._complaints(s_sq, sq)
        parked_plan, parked_sq = float(s_plan["served"].sum()), float(s_sq["served"].sum())

        def pct(a, b):
            return (a / b - 1.0) if b else 0.0

        deltas = {"revenue_pct": pct(k_plan["revenue"], k_sq["revenue"]),
                  "net_value_pct": pct(k_plan["net_value"], k_sq["net_value"]),
                  "utilization_pts": k_plan["utilization"] - k_sq["utilization"],
                  "turned_away_pct": pct(k_plan["turned_away"], k_sq["turned_away"]),
                  "lost_demand_pct": pct(k_plan["lost_demand"], k_sq["lost_demand"]),
                  "avg_multiplier_pct": pct(k_plan["avg_multiplier"], k_sq["avg_multiplier"]),
                  "complaints_pct": pct(c_plan, c_sq)}
        actions = self._actions(plan, sq)
        zones = []
        for z in range(Z):
            peak = int(np.argmax(s_plan["demand"][z]))
            zones.append({
                "id": z, "name": self.zones[z]["name"], "multipliers": np.round(plan[z], 3).tolist(),
                "status_quo_multipliers": np.round(sq[z], 3).tolist(),
                "occupancy": np.round(s_plan["occupancy"][z], 4).tolist(),
                "occupancy_status_quo": np.round(s_sq["occupancy"][z], 4).tolist(),
                "revenue": float(s_plan["revenue"][z].sum()), "revenue_status_quo": float(s_sq["revenue"][z].sum()),
                "revenue_delta_pct": pct(float(s_plan["revenue"][z].sum()), float(s_sq["revenue"][z].sum())),
                "avg_multiplier": float(plan[z].mean()), "turned_away": float(s_plan["turned_away"][z].sum()),
                "turned_away_status_quo": float(s_sq["turned_away"][z].sum()),
                "peak_hour": peak, "peak_occupancy_status_quo": float(s_sq["occupancy"][z].max()),
                "locked": z in locked, "action": actions[z]})
        out = {"date": date, "is_live": date == self.live_date, "strategy": strategy, "scenario": scenario,
               "guardrails": g.as_dict(), "spill_share": spill_share,
               "totals": {"status_quo": {**k_sq, "complaints": c_sq, "complaints_per_10k": c_sq / max(parked_sq, 1) * 1e4},
                          "plan": {**k_plan, "complaints": c_plan, "complaints_per_10k": c_plan / max(parked_plan, 1) * 1e4},
                          "deltas": deltas},
               "zones": zones}
        if date != self.live_date:       # reality check: replay the same plan against the hidden true demand
            lat = self.truth_latent[:, i, :]
            r_plan = P.kpis(P.simulate(lat, self.truth_beta, plan, self.truth_spill), plan, guardrails=g)
            r_sq = P.kpis(P.simulate(lat, self.truth_beta, sq, self.truth_spill), sq, guardrails=g)
            out["reality_check"] = {"revenue_pct": pct(r_plan["revenue"], r_sq["revenue"]),
                                    "net_value_pct": pct(r_plan["net_value"], r_sq["net_value"]),
                                    "expected_revenue_pct": deltas["revenue_pct"],
                                    "note": "Synthetic ground truth: the same plan replayed against the true demand curve."}
        return out

    def overview(self, date: str) -> dict:
        sim = self.simulate(date, "optimized")
        i = self._idx(date)
        d0, beta = self.p50[:, i, :], self._beta(i)
        sq = C.status_quo_matrix(bool(self.is_offday[i]))
        heat_sq = P.simulate(d0, beta, sq, 0.25)["occupancy"]
        at_risk = [{"zone": z["name"], "zone_id": z["id"], "peak_occupancy": z["peak_occupancy_status_quo"], "hour": z["peak_hour"]}
                   for z in sorted(sim["zones"], key=lambda r: -r["peak_occupancy_status_quo"]) if z["peak_occupancy_status_quo"] >= 0.95][:5]
        return {"date": date, "is_live": sim["is_live"], "totals": sim["totals"], "zones": sim["zones"],
                "heatmap_status_quo": np.round(heat_sq, 4).tolist(),
                "heatmap_plan": [z["occupancy"] for z in sim["zones"]], "at_risk": at_risk,
                "reality_check": sim.get("reality_check")}
