"""Dynamic pricing: demand curves, a network simulator and a guardrail-constrained optimizer.

Demand curve (per zone, per hour)
---------------------------------
    demand(m) = d0 * capacity * exp(beta * (m - 1))        m = price multiplier

``d0`` comes from the forecaster and ``beta`` from the experiment-based estimator. Elasticity
at multiplier m is ``beta * m``: drivers get more price-sensitive as the price climbs, so the
revenue-maximising multiplier is finite (about 1/|beta|) rather than "always raise".

Capacity and spillover
----------------------
Served drivers are capped at capacity. Drivers who are turned away or deterred by price may try
a neighbouring lot with probability decaying in distance (single pass, limited by spare
capacity there). The spill share is an *assumption*, not something the data identifies, so
every report shows sensitivity to it.

Optimizer
---------
Per zone, dynamic programming over the 24 hours picks one multiplier per hour from a grid to
maximise  revenue - turn_cost * (turned away) * base_price - deter_cost * (priced out) * base_price,
subject to guardrails (the two cost terms are the "growth vs yield" dial: pure revenue maximisation
quietly sacrifices volume, which a parking platform cannot afford):

* ``min_mult`` / ``max_mult``    bounds on any single hour's price;
* ``max_step``                   limit on hour-to-hour price change (drivers hate jumpy prices);
* ``avg_cap``                    cap on the day's average multiplier, enforced with a Lagrange
                                 multiplier found by bisection;
* ``locked`` zones               stay on the current static schedule.

Nothing here imports the hidden ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config as C

GRID_STEP = 0.05


@dataclass(frozen=True)
class Guardrails:
    min_mult: float = 0.70
    max_mult: float = 1.40
    max_step: float = 0.15
    avg_cap: float = 1.15
    turn_cost: float = 0.50   # goodwill cost of a driver turned away at a full lot (hours of base price)
    deter_cost: float = 0.50  # goodwill cost of a driver priced out (hours of base price): the growth-vs-yield dial

    def validate(self) -> "Guardrails":
        if not (0.3 <= self.min_mult <= 1.0 <= self.max_mult <= 3.0):
            raise ValueError("require 0.3 <= min_mult <= 1 <= max_mult <= 3")
        if not (0.05 <= self.max_step <= 2.0):
            raise ValueError("max_step must be within [0.05, 2]")
        if not (1.0 <= self.avg_cap <= self.max_mult):
            raise ValueError("avg_cap must be within [1, max_mult]")
        if self.turn_cost < 0 or self.deter_cost < 0:
            raise ValueError("turn_cost and deter_cost must be >= 0")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


class Network:
    """Static network arrays: capacities, base prices and the spillover weight matrix."""

    def __init__(self, zones=C.ZONES, ell_km: float = 1.5, radius_km: float = 6.0):
        self.zones = zones
        self.cap = np.array([z.capacity for z in zones], dtype=float)
        self.base = np.array([z.base_price for z in zones], dtype=float)
        dist = C.distance_matrix_km(zones)
        k = np.exp(-dist / ell_km)
        k[dist > radius_km] = 0.0
        np.fill_diagonal(k, 0.0)
        # W[i, j]: probability a displaced driver at i tries j. The "1 +" is the outside option
        # (give up / go elsewhere), so isolated lots lose their displaced demand entirely.
        self.W = k / (1.0 + k.sum(axis=1, keepdims=True))


NET = Network()


def demand_counts(d0: np.ndarray, beta: np.ndarray, m: np.ndarray, net: Network = NET) -> np.ndarray:
    return net.cap[:, None] * d0 * np.exp(beta[:, None] * (m - 1.0))


def simulate(d0: np.ndarray, beta: np.ndarray, m: np.ndarray, spill_share: float = 0.25,
             net: Network = NET) -> dict[str, np.ndarray]:
    """Expected-value simulation of one day. All arrays are shaped [Z, 24]."""
    cap, base = net.cap[:, None], net.base[:, None]
    dem0 = cap * d0
    dem = dem0 * np.exp(beta[:, None] * (m - 1.0))
    served = np.minimum(dem, cap)
    turned = dem - served
    price_deterred = np.maximum(dem0 - dem, 0.0)
    displaced = turned + price_deterred
    inflow = spill_share * (net.W.T @ displaced)
    spill_served = np.minimum(inflow, cap - served)
    total_served = served + spill_served
    price = base * m
    return {"demand": dem, "served_own": served, "turned_away": turned, "price_deterred": price_deterred,
            "spill_in": spill_served, "served": total_served, "price": price,
            "revenue": total_served * price, "occupancy": total_served / cap,
            "lost": np.maximum(displaced - _recovered(displaced, spill_served), 0.0)}


def _recovered(displaced: np.ndarray, spill_served: np.ndarray) -> np.ndarray:
    """Spill actually recaptured, attributed back to origin zones pro rata (for loss accounting)."""
    tot_d = displaced.sum(axis=0, keepdims=True)
    tot_s = spill_served.sum(axis=0, keepdims=True)
    share = np.divide(tot_s, tot_d, out=np.zeros_like(tot_s), where=tot_d > 0)
    return displaced * share


def kpis(sim: dict[str, np.ndarray], m: np.ndarray, net: Network = NET,
         guardrails: "Guardrails | None" = None) -> dict:
    """Network-level KPIs for one simulated day.

    ``net_value`` is revenue minus the goodwill cost of turned-away and priced-out drivers: the
    quantity the optimizer actually maximises, so it is the fair yardstick for "regret".
    """
    g = guardrails or Guardrails()
    cap = net.cap[:, None]
    dm = np.abs(np.diff(m, axis=1))
    goodwill = (g.turn_cost * sim["turned_away"] + g.deter_cost * sim["price_deterred"]) * net.base[:, None]
    return {
        "revenue": float(sim["revenue"].sum()),
        "net_value": float(sim["revenue"].sum() - goodwill.sum()),
        "utilization": float(sim["served"].sum() / (cap.sum() * 24)),
        "turned_away": float(sim["turned_away"].sum()),
        "lost_demand": float(sim["lost"].sum()),
        "full_zone_hours": int((sim["occupancy"] >= 0.95).sum()),
        "avg_multiplier": float(m.mean()),
        "price_volatility": float(dm.mean()),
        "served": float(sim["served"].sum()),
    }


# ------------------------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------------------------
def _grid(g: Guardrails) -> np.ndarray:
    return np.round(np.arange(g.min_mult, g.max_mult + 1e-9, GRID_STEP), 4)


def _dp_zone(J: np.ndarray, grid: np.ndarray, max_step: float) -> np.ndarray:
    """Best path through ``J[h, k]`` with |m_h - m_{h-1}| <= max_step."""
    H, K = J.shape
    ok = np.abs(grid[:, None] - grid[None, :]) <= max_step + 1e-9      # [k, k']
    pen = np.where(ok, 0.0, -np.inf)
    V = np.empty((H, K))
    back = np.zeros((H, K), dtype=int)
    V[0] = J[0]
    for h in range(1, H):
        cand = V[h - 1][None, :] + pen                                   # [k, k']
        back[h] = np.argmax(cand, axis=1)
        V[h] = J[h] + cand[np.arange(K), back[h]]
    k = int(np.argmax(V[-1]))
    path = np.empty(H, dtype=int)
    for h in range(H - 1, -1, -1):
        path[h] = k
        k = back[h, k]
    return grid[path]


def _optimize_zone(d0: np.ndarray, beta: float, cap: float, base: float, g: Guardrails) -> np.ndarray:
    grid = _grid(g)
    dem = cap * d0[:, None] * np.exp(beta * (grid[None, :] - 1.0))       # [24, K]
    served = np.minimum(dem, cap)
    turned = dem - served
    deterred = np.maximum(cap * d0[:, None] - dem, 0.0)
    J = served * base * grid[None, :] - g.turn_cost * turned * base - g.deter_cost * deterred * base
    unit = base * cap * 0.05

    def solve(lam: float) -> np.ndarray:
        return _dp_zone(J - lam * unit * (grid[None, :] - 1.0), grid, g.max_step)

    plan = solve(0.0)
    if plan.mean() <= g.avg_cap + 1e-9:
        return plan
    lo, hi = 0.0, 1.0
    while solve(hi).mean() > g.avg_cap + 1e-9 and hi < 1e4:
        hi *= 2
    for _ in range(30):
        mid = (lo + hi) / 2
        if solve(mid).mean() > g.avg_cap + 1e-9:
            lo = mid
        else:
            hi = mid
    return solve(hi)


def optimize(d0: np.ndarray, beta: np.ndarray, guardrails: Guardrails | None = None,
             locked: tuple[int, ...] | list[int] = (), is_offday: bool = False,
             net: Network = NET) -> np.ndarray:
    """Hourly multipliers [Z, 24] maximising guarded revenue. Locked zones keep the static schedule."""
    g = (guardrails or Guardrails()).validate()
    sq = C.status_quo_matrix(is_offday, net.zones)
    plan = np.empty_like(d0)
    for z in range(d0.shape[0]):
        if z in set(locked):
            plan[z] = sq[z]
        else:
            plan[z] = _optimize_zone(d0[z], float(beta[z]), net.cap[z], net.base[z], g)
    return plan


def flat_plan(n_zones: int = C.N_ZONES) -> np.ndarray:
    return np.ones((n_zones, 24))
