"""Simulator invariants and the price-sensitivity estimator against known ground truth."""
import unittest

import numpy as np

from parkwise import config as C
from parkwise import elasticity as E
from parkwise import generator as G


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = G.generate(seed=3)
        cls.df = cls.sim.hourly

    def test_deterministic_for_a_seed(self):
        again = G.generate(seed=3).hourly
        np.testing.assert_array_equal(self.df["arrivals"].to_numpy(), again["arrivals"].to_numpy())

    def test_full_grid(self):
        days = self.df["date"].nunique()
        self.assertEqual(len(self.df), C.N_ZONES * days * 24)

    def test_occupancy_never_exceeds_capacity_and_conserves_arrivals(self):
        d = self.df[self.df["sensor_ok"]]
        self.assertTrue((d["occupied"] <= d["capacity"] + 1e-9).all())
        np.testing.assert_allclose(d["occupied"] + d["turned_away"], d["arrivals"])

    def test_live_day_has_no_outcomes(self):
        live = self.df[self.df["date"] == C.LIVE_DATE]
        self.assertTrue(live["arrivals"].isna().all() and live["latent_d0"].isna().all())
        self.assertTrue((live["price_mult_exp"] == 1.0).all())

    def test_sensor_gaps_exist_but_are_rare(self):
        share = (~self.df["sensor_ok"]).mean()
        self.assertTrue(0.001 < share < 0.05)

    def test_some_lots_actually_fill_up(self):
        self.assertGreater((self.df["occupancy"] >= 0.99).mean(), 0.03)


class ElasticityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = G.generate(seed=3)
        cls.est = E.estimate(cls.sim.hourly, C.DATA_END)

    def test_estimates_close_to_truth(self):
        err = np.abs(self.est["beta"].to_numpy() - G.TRUE_BETA)
        self.assertLess(err.mean(), 0.12)
        self.assertLess(err.max(), 0.35)

    def test_confidence_intervals_mostly_cover_truth(self):
        covered = ((G.TRUE_BETA >= self.est["ci_lo"]) & (G.TRUE_BETA <= self.est["ci_hi"])).mean()
        self.assertGreaterEqual(covered, 0.75)

    def test_censored_outcome_understates_sensitivity_in_busy_lots(self):
        busy = self.est["zone_id"].isin([0, 1, 5])    # Indiranagar, Koramangala, Majestic fill up regularly
        proper = self.est.loc[busy, "beta"].abs().mean()
        naive = self.est.loc[busy, "beta_naive_censored"].abs().mean()
        self.assertLess(naive, proper)

    def test_only_experiment_variation_identifies_the_slope(self):
        """Without randomised experiments there is nothing left after the fixed effects."""
        df = self.sim.hourly.copy()
        df["price_mult"] = df["price_mult_sq"]
        est = E.estimate(df, C.DATA_END)
        self.assertTrue(est["beta"].isna().all() or (est["beta"].abs() < 1e-6).all() or est["se"].isna().all()
                        or (est["se"] > 10).all())


if __name__ == "__main__":
    unittest.main()
