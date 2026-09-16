import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ensemble


class TestEnsemble(unittest.TestCase):
    def test_weights_from_accuracy_sum_to_one_and_reward_better_method(self):
        w = ensemble.weights_from_accuracy({"ngram": 0.7, "alphabet": 0.6, "neural": 0.95})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["neural"], w["alphabet"])
        self.assertGreater(w["neural"], w["ngram"])

    def test_weights_from_accuracy_never_fully_zeroes_a_method(self):
        w = ensemble.weights_from_accuracy({"a": 0.1, "b": 0.99}, floor=0.1)
        self.assertGreater(w["a"], 0.0)

    def test_combine_prefers_majority_verdict(self):
        results = {
            "ngram": {"best": "ru", "scores": {"ru": 10, "it": 40}, "kind": "distance"},
            "alphabet": {"best": "ru", "scores": {"ru": 0.1, "it": 0.4}, "kind": "distance"},
            "neural": {"best": "it", "scores": {"ru": 0.4, "it": 0.6}, "kind": "probability"},
        }
        weights = {"ngram": 1 / 3, "alphabet": 1 / 3, "neural": 1 / 3}
        out = ensemble.combine(results, weights)
        self.assertEqual(out["best"], "ru")
        self.assertAlmostEqual(sum(out["scores"].values()), 1.0, places=6)

    def test_full_agreement_yields_high_confidence(self):
        results = {
            "ngram": {"best": "it", "scores": {"ru": 50, "it": 5}, "kind": "distance"},
            "alphabet": {"best": "it", "scores": {"ru": 0.5, "it": 0.05}, "kind": "distance"},
            "neural": {"best": "it", "scores": {"ru": 0.02, "it": 0.98}, "kind": "probability"},
        }
        weights = {"ngram": 1 / 3, "alphabet": 1 / 3, "neural": 1 / 3}
        out = ensemble.combine(results, weights)
        self.assertEqual(out["agreement"], 1.0)
        self.assertEqual(out["confidence_level"], "высокая")

    def test_combine_defaults_to_equal_weights_when_none_given(self):
        results = {
            "ngram": {"best": "ru", "scores": {"ru": 1, "it": 100}, "kind": "distance"},
            "alphabet": {"best": "ru", "scores": {"ru": 0.01, "it": 1.0}, "kind": "distance"},
        }
        out = ensemble.combine(results)
        self.assertEqual(out["best"], "ru")


if __name__ == "__main__":
    unittest.main()
