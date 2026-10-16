import unittest

import numpy as np
from sklearn.metrics import f1_score

from parkwise import tickets as T


class TicketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = T.train_and_evaluate(n_labeled=1200, seed=2)

    def test_classifier_beats_keyword_baseline(self):
        m = self.report.metrics
        self.assertGreater(m["model"]["macro_f1"], m["keyword_baseline"]["macro_f1"] + 0.08)

    def test_metrics_are_not_suspiciously_perfect(self):
        self.assertLess(self.report.metrics["model"]["accuracy"], 0.98)

    def test_generation_is_deterministic(self):
        a, b = T.generate_labeled(50, seed=9), T.generate_labeled(50, seed=9)
        self.assertTrue(a.equals(b))

    def test_predictions_are_valid_categories_with_probabilities(self):
        data = T.generate_labeled(40, seed=4)
        p = self.report.model.predict_proba(data["text"])
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(set(self.report.model.predict(data["text"])) <= set(T.CATEGORIES))

    def test_handles_empty_and_odd_text(self):
        out = self.report.model.predict(["", "   ", "🚗🚗🚗", "A" * 5000])
        self.assertEqual(len(out), 4)


if __name__ == "__main__":
    unittest.main()
