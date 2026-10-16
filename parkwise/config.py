"""Static configuration: the synthetic city, calendar, status-quo pricing rule and data splits.

Everything in here is *observable* information a real platform would have (zone metadata,
the operator's current price schedule, public holidays). Hidden ground truth used only to
synthesise data and to grade the system lives in ``parkwise.generator`` and is never
imported by the modelling code (enforced by a test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Demand archetypes. A zone's hourly profile is a mixture of these, weighted by the share of
# nearby points of interest. That compositional structure is what makes cold-start forecasting
# for a brand-new lot possible.
ARCHETYPES = ("office", "retail", "nightlife", "residential", "transit", "medical", "travel")

DATA_START = "2025-10-01"
DATA_END = "2026-10-14"    # last day with observed outcomes
LIVE_DATE = "2026-10-15"   # "tomorrow": forecast only, no outcomes exist yet
TRAIN_END = "2026-08-15"   # train: DATA_START..TRAIN_END
VAL_END = "2026-09-15"     # validation: (TRAIN_END, VAL_END]; test: (VAL_END, DATA_END]
WARMUP_DAYS = 28           # lag features need history; earlier days are not used for fitting


@dataclass(frozen=True)
class Zone:
    id: int
    slug: str
    name: str
    lat: float
    lon: float
    capacity: int          # parking bays
    base_price: float      # INR per hour at multiplier 1.0
    poi: dict              # share of nearby POIs per archetype (sums to 1)
    venue: bool = False    # hosts large scheduled events

    def poi_vector(self) -> np.ndarray:
        return np.array([self.poi.get(a, 0.0) for a in ARCHETYPES])


def _z(i, slug, name, lat, lon, cap, price, poi, venue=False):
    assert abs(sum(poi.values()) - 1.0) < 1e-9, slug
    return Zone(i, slug, name, lat, lon, cap, price, poi, venue)


# A fictional parking network loosely laid out over a large Indian metro. Locations are
# approximate and demand is entirely synthetic.
ZONES: tuple[Zone, ...] = (
    _z(0, "indiranagar", "Indiranagar High Street", 12.9784, 77.6408, 280, 60,
       {"retail": .40, "nightlife": .35, "residential": .15, "office": .10}),
    _z(1, "koramangala", "Koramangala Food Strip", 12.9352, 77.6245, 220, 50,
       {"nightlife": .55, "retail": .20, "office": .15, "residential": .10}),
    _z(2, "mgroad", "MG Road Commercial", 12.9756, 77.6097, 450, 80,
       {"office": .45, "retail": .35, "transit": .10, "nightlife": .10}),
    _z(3, "whitefield", "Whitefield Tech Park", 12.9855, 77.7270, 900, 40,
       {"office": .90, "retail": .05, "residential": .05}),
    _z(4, "manyata", "Manyata Business Park", 13.0475, 77.6200, 1100, 40,
       {"office": .95, "transit": .05}),
    _z(5, "majestic", "Majestic Transit Hub", 12.9767, 77.5713, 600, 30,
       {"transit": .80, "retail": .15, "residential": .05}),
    _z(6, "rajajinagar", "Rajajinagar Mall", 13.0110, 77.5550, 700, 40,
       {"retail": .80, "nightlife": .10, "residential": .10}),
    _z(7, "jayanagar", "Jayanagar Market", 12.9250, 77.5938, 260, 30,
       {"retail": .70, "residential": .20, "medical": .10}),
    _z(8, "stadium", "Stadium District", 12.9788, 77.5996, 1500, 50,
       {"retail": .20, "residential": .30, "office": .20, "nightlife": .30}, venue=True),
    _z(9, "hebbal", "Hebbal Hospital Cluster", 13.0358, 77.5970, 380, 30,
       {"medical": .80, "residential": .10, "office": .10}),
    _z(10, "airport", "Airport Road Long-Stay", 13.1900, 77.7000, 1200, 45,
       {"travel": .90, "transit": .10}),
    _z(11, "sarjapur", "Sarjapur Residential", 12.9150, 77.6700, 240, 20,
       {"residential": .85, "retail": .10, "office": .05}),
)
N_ZONES = len(ZONES)
ZONE_BY_ID = {z.id: z for z in ZONES}

# Public holidays used by the calendar (approximate dates; the world is synthetic).
HOLIDAYS = {
    "2025-10-02": "Gandhi Jayanti", "2025-10-20": "Diwali", "2025-10-21": "Diwali",
    "2025-11-01": "Rajyotsava", "2025-12-25": "Christmas", "2025-12-31": "New Year's Eve",
    "2026-01-01": "New Year", "2026-01-14": "Sankranti", "2026-01-26": "Republic Day",
    "2026-03-04": "Holi", "2026-03-21": "Eid", "2026-04-03": "Good Friday",
    "2026-04-14": "Ambedkar Jayanti", "2026-05-01": "May Day", "2026-08-15": "Independence Day",
    "2026-09-14": "Ganesh Chaturthi", "2026-10-02": "Gandhi Jayanti",
}


def distance_matrix_km(zones=ZONES) -> np.ndarray:
    """Great-circle distances between zones (haversine)."""
    lat = np.radians([z.lat for z in zones])
    lon = np.radians([z.lon for z in zones])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def status_quo_matrix(is_offday: bool, zones=ZONES) -> np.ndarray:
    """The operator's *current* static price schedule as multipliers on base price, shape [Z, 24].

    Night discount (00-06) everywhere except residential lots, and a weekend/holiday
    afternoon-evening surcharge in retail/nightlife-heavy zones. This is the baseline that
    dynamic pricing has to beat.
    """
    out = np.ones((len(zones), 24))
    for i, z in enumerate(zones):
        if z.poi.get("residential", 0.0) < 0.5:
            out[i, 0:6] *= 0.85
        if is_offday and z.poi.get("retail", 0.0) + z.poi.get("nightlife", 0.0) >= 0.5:
            out[i, 12:22] *= 1.10
    return out
