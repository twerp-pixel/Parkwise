"""Optimizer guardrails and simulator conservation laws."""
import unittest

import numpy as np

from parkwise import config as C
from parkwise import pricing as P

Z = C.N_ZONES
RNG = np.random.default_rng(1)
BETA = -np.array([0.9, 1.0, 0.6, 0.35, 0.3, 0.45, 0.8, 1.1, 0.5, 0.25, 0.3, 0.7])
# A plausible day: strong demand in the evening, overflowing in some zones.
D0 = (0.4 + 0.9 * np.exp(-((np.arange(24) - 19) ** 2) / 20))[None, :] * RNG.uniform(0.6, 1.4, (Z, 1))


class OptimizerTests(unittest.TestCase):
    def test_respects_every_guardrail(self):
        for g in (P.Guardrails(), P.Guardrails(min_mult=0.9, max_mult=1.2, max_step=0.05, avg_cap=1.05),
                  P.Guardrails(max_mult=2.0, avg_cap=1.5, max_step=0.5, turn_cost=0, deter_cost=0)):
            m = P.optimize(D0, BETA, g)
            self.assertGreaterEqual(m.min(), g.min_mult - 1e-9)
            self.assertLessEqual(m.max(), g.max_mult + 1e-9)
            self.assertLessEqual(np.abs(np.diff(m, axis=1)).max(), g.max_step + 1e-9)
            self.assertLessEqual(m.mean(axis=1).max(), g.avg_cap + 1e-6)

    def test_locked_zones_keep_the_current_schedule(self):
        m = P.optimize(D0, BETA, P.Guardrails(), locked=[2, 5], is_offday=False)
        sq = C.status_quo_matrix(False)
        np.testing.assert_allclose(m[[2, 5]], sq[[2, 5]])

    def test_beats_current_schedule_on_the_objective_it_optimises(self):
        g = P.Guardrails()
        plan = P.optimize(D0, BETA, g)
        sq = C.status_quo_matrix(False)
        a = P.kpis(P.simulate(D0, BETA, plan, 0.0), plan, guardrails=g)["net_value"]
        b = P.kpis(P.simulate(D0, BETA, sq, 0.0), sq, guardrails=g)["net_value"]
        self.assertGreater(a, b)

    def test_valuing_drivers_more_lowers_prices(self):
        lo = P.optimize(D0, BETA, P.Guardrails(deter_cost=0.0, turn_cost=0.0)).mean()
        hi = P.optimize(D0, BETA, P.Guardrails(deter_cost=3.0, turn_cost=3.0)).mean()
        self.assertLess(hi, lo)

    def test_invalid_guardrails_are_rejected(self):
        for bad in (P.Guardrails(max_mult=0.9), P.Guardrails(avg_cap=0.9), P.Guardrails(max_step=0.0), P.Guardrails(deter_cost=-1)):
            with self.assertRaises(ValueError):
                bad.validate()


class SimulatorTests(unittest.TestCase):
    def test_conservation(self):
        for m in (np.ones((Z, 24)), C.status_quo_matrix(True), np.full((Z, 24), 1.4), np.full((Z, 24), 0.7)):
            s = P.simulate(D0, BETA, m, spill_share=0.5)
            cap = P.NET.cap[:, None]
            self.assertTrue((s["served"] <= cap + 1e-6).all())
            self.assertTrue((s["spill_in"] >= -1e-9).all())
            self.assertTrue((s["spill_in"] <= cap - s["served_own"] + 1e-6).all())
            # Recaptured spill can never exceed what was displaced.
            self.assertLessEqual(s["spill_in"].sum(), (s["turned_away"] + s["price_deterred"]).sum() * 0.5 + 1e-6)

    def test_higher_prices_reduce_demand(self):
        lo = P.simulate(D0, BETA, np.full((Z, 24), 0.9), 0.0)["demand"].sum()
        hi = P.simulate(D0, BETA, np.full((Z, 24), 1.3), 0.0)["demand"].sum()
        self.assertLess(hi, lo)

    def test_spillover_never_lowers_revenue(self):
        m = np.full((Z, 24), 1.3)
        self.assertGreaterEqual(P.simulate(D0, BETA, m, 0.5)["revenue"].sum(), P.simulate(D0, BETA, m, 0.0)["revenue"].sum())


if __name__ == "__main__":
    unittest.main()
