"""No-leakage guarantees: features for day D never see outcomes from day D onward, and modelling code
never touches the hidden ground truth."""
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from parkwise import config as C
from parkwise import features as F
from parkwise import generator as G

PKG = Path(__file__).resolve().parent.parent / "parkwise"


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = G.generate(seed=5)
        cls.beta = G.TRUE_BETA.copy()

    def test_corrupting_outcomes_from_day_d_does_not_change_day_d_features(self):
        day = pd.Timestamp("2026-06-10")
        base = F.build(self.sim.hourly, self.sim.events, self.beta)
        bad = self.sim.hourly.copy()
        mask = bad["date"] >= day
        rng = np.random.default_rng(0)
        for col in ("arrivals", "occupied", "turned_away", "revenue"):
            bad.loc[mask, col] = rng.uniform(0, 5000, mask.sum())
        bad.loc[mask, "sensor_ok"] = True
        corrupted = F.build(bad, self.sim.events, self.beta)
        rows = (base.meta["date"] == day).to_numpy()
        self.assertGreater(rows.sum(), 0)
        pd.testing.assert_frame_equal(base.X[rows].reset_index(drop=True), corrupted.X[rows].reset_index(drop=True))

    def test_features_do_use_yesterday(self):
        """Sanity check that the test above can fail: changing day D-1 must change day D features."""
        day = pd.Timestamp("2026-06-10")
        base = F.build(self.sim.hourly, self.sim.events, self.beta)
        bad = self.sim.hourly.copy()
        m = bad["date"] == day - pd.Timedelta(days=1)
        bad.loc[m, "arrivals"] = bad.loc[m, "arrivals"] * 3
        changed = F.build(bad, self.sim.events, self.beta)
        rows = (base.meta["date"] == day).to_numpy()
        self.assertFalse(base.X.loc[rows, "lag_24"].equals(changed.X.loc[rows, "lag_24"]))

    def test_live_day_is_featurised_with_no_target(self):
        fs = F.build(self.sim.hourly, self.sim.events, self.beta)
        live = (fs.meta["date"] == C.LIVE_DATE).to_numpy()
        self.assertTrue(np.isnan(fs.y[live]).all())
        self.assertFalse(fs.X.loc[live, "lag_24"].isna().all())


class BoundaryTests(unittest.TestCase):
    FORBIDDEN = re.compile(r"GROUND_TRUTH|TRUE_BETA|TRUE_SCALE|TRUE_SPILL|latent_d0|import generator|from \.generator|from \. import .*generator")

    def test_modelling_code_never_reads_ground_truth(self):
        for name in ("features.py", "forecast.py", "elasticity.py", "pricing.py", "tickets.py"):
            text = (PKG / name).read_text()
            code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
            self.assertIsNone(self.FORBIDDEN.search(code), f"{name} references hidden ground truth")


if __name__ == "__main__":
    unittest.main()
