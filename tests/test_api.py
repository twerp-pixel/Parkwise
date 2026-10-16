"""API contract tests. They run against the built artifacts (python -m parkwise build)."""
import unittest

from starlette.testclient import TestClient

from parkwise import config as C
from parkwise.api import create_app

HAVE = (C.ARTIFACTS / "report.json").exists()


@unittest.skipUnless(HAVE, "run `python -m parkwise build` first")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(create_app())
        cls.c = cls._cm.__enter__()
        cls.meta = cls.c.get("/api/meta").json()
        cls.live = cls.meta["live_date"]
        cls.backtest = next(d["date"] for d in cls.meta["dates"] if d["kind"] == "backtest")

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_health_and_headers(self):
        r = self.c.get("/api/health")
        self.assertEqual(r.json()["status"], "ok")
        self.assertIn("x-request-id", r.headers)
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")

    def test_meta_lists_all_zones_and_dates(self):
        self.assertEqual(len(self.meta["zones"]), C.N_ZONES)
        self.assertEqual(self.meta["dates"][-1]["kind"], "live")

    def test_overview_shape(self):
        o = self.c.get("/api/overview", params={"date": self.live}).json()
        self.assertEqual(len(o["heatmap_status_quo"]), C.N_ZONES)
        self.assertEqual(len(o["heatmap_status_quo"][0]), 24)
        self.assertIn("deltas", o["totals"])

    def test_forecast_band_is_ordered_and_backtest_has_actuals(self):
        f = self.c.get("/api/forecast", params={"zone_id": 3, "date": self.backtest}).json()
        for lo, mid, hi in zip(f["p10"], f["p50"], f["p90"]):
            self.assertLessEqual(lo, mid + 1e-9)
            self.assertLessEqual(mid, hi + 1e-9)
        self.assertEqual(len(f["actual"]), 24)
        self.assertNotIn("actual", self.c.get("/api/forecast", params={"zone_id": 3, "date": self.live}).json())

    def test_simulate_default_guardrails_hold(self):
        r = self.c.post("/api/simulate", json={"date": self.live}).json()
        g = r["guardrails"]
        for z in r["zones"]:
            self.assertLessEqual(max(z["multipliers"]), g["max_mult"] + 1e-6)
            self.assertLessEqual(z["avg_multiplier"], g["avg_cap"] + 1e-6)

    def test_simulate_strategies(self):
        sq = self.c.post("/api/simulate", json={"date": self.live, "strategy": "status_quo"}).json()
        self.assertAlmostEqual(sq["totals"]["deltas"]["revenue_pct"], 0.0, places=9)
        locked = self.c.post("/api/simulate", json={"date": self.live, "locked_zones": [3]}).json()
        self.assertTrue(locked["zones"][3]["locked"])
        self.assertEqual(locked["zones"][3]["multipliers"], locked["zones"][3]["status_quo_multipliers"])
        custom = self.c.post("/api/simulate", json={"date": self.live, "strategy": "custom", "adjustments": {"0": 0.2}}).json()
        self.assertGreater(custom["zones"][0]["avg_multiplier"], custom["zones"][0]["status_quo_multipliers"][0])

    def test_backtest_includes_reality_check_but_live_does_not(self):
        self.assertIn("reality_check", self.c.post("/api/simulate", json={"date": self.backtest}).json())
        self.assertNotIn("reality_check", self.c.post("/api/simulate", json={"date": self.live}).json())

    def test_validation_errors_are_clear(self):
        for body in ({"date": "nope"}, {"date": self.live, "strategy": "magic"}, {"date": self.live, "spill_share": 3},
                     {"date": self.live, "locked_zones": [99]}, {"date": self.live, "adjustments": {"0": 9}},
                     {"date": self.live, "guardrails": {"max_mult": 1.2, "avg_cap": 1.4}}):
            r = self.c.post("/api/simulate", json=body)
            self.assertEqual(r.status_code, 422, body)
            self.assertEqual(r.json()["error"]["code"], "validation_error")

    def test_not_found_and_bad_json(self):
        self.assertEqual(self.c.post("/api/simulate", json={"date": "2031-01-01"}).status_code, 404)
        self.assertEqual(self.c.get("/api/forecast", params={"zone_id": 99, "date": self.live}).status_code, 404)
        self.assertEqual(self.c.get("/api/forecast", params={"zone_id": "x"}).status_code, 422)
        self.assertEqual(self.c.post("/api/simulate", content=b"{oops", headers={"content-type": "application/json"}).status_code, 400)

    def test_dashboard_is_served(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)


@unittest.skipUnless(HAVE, "run `python -m parkwise build` first")
class ReportSanityTests(unittest.TestCase):
    """Guards on the headline claims in the report, so a regression cannot silently ship."""
    @classmethod
    def setUpClass(cls):
        import json
        cls.r = json.loads((C.ARTIFACTS / "report.json").read_text())

    def test_model_beats_both_baselines(self):
        p = self.r["forecast"]["point"]
        self.assertLess(p["gbm"]["wmape"], p["seasonal_naive"]["wmape"] * 0.8)
        self.assertLess(p["gbm"]["wmape"], p["hist_avg_4w"]["wmape"])

    def test_interval_is_reasonably_calibrated(self):
        self.assertTrue(0.70 <= self.r["forecast"]["interval"]["coverage"] <= 0.88)

    def test_pricing_gain_is_positive_with_confidence_interval_above_zero(self):
        m = self.r["policy"]["policies"]["dynamic_model"]
        self.assertGreater(m["uplift_ci95"][0], 0)
        self.assertLessEqual(m["net_value_uplift_vs_status_quo"], self.r["policy"]["policies"]["oracle"]["net_value_uplift_vs_status_quo"] + 1e-9)

    def test_elasticity_intervals_cover_truth(self):
        el = self.r["elasticity"]
        covered = sum(e["ci_lo"] <= e["true_beta"] <= e["ci_hi"] for e in el)
        self.assertGreaterEqual(covered, len(el) - 2)


if __name__ == "__main__":
    unittest.main()
