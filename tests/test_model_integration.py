import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CORPUS_DIR, TEST_DIR
from core.model import LanguageModel


class TestModelIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LanguageModel.train(CORPUS_DIR, TEST_DIR)

    def test_all_methods_and_ensemble_report_accuracy_on_holdout_set(self):
        for key in ("ngram", "alphabet", "neural", "ensemble"):
            self.assertIn(key, self.model.accuracy)
            self.assertGreaterEqual(self.model.accuracy[key], 0.5)

    def test_cross_validation_accuracy_is_present_for_every_method(self):
        for key in ("ngram", "alphabet", "neural", "ensemble"):
            self.assertIn(key, self.model.cv_accuracy)
            self.assertGreaterEqual(self.model.cv_accuracy[key], 0.0)

    def test_ensemble_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(self.model.ensemble_weights.values()), 1.0, places=6)

    def test_classify_text_returns_all_four_results(self):
        outcome = self.model.classify_text("Il gatto nero dorme sul divano vicino alla finestra")
        for key in ("ngram", "alphabet", "neural", "ensemble"):
            self.assertIn(key, outcome)
            self.assertIn(outcome[key]["best"], self.model.languages)

    def test_classify_clear_italian_text(self):
        outcome = self.model.classify_text(
            "Buongiorno, oggi il sole splende e la citta e piena di gente allegra per le strade."
        )
        self.assertEqual(outcome["ensemble"]["best"], "it")

    def test_classify_clear_russian_text(self):
        outcome = self.model.classify_text(
            "Сегодня прекрасная погода, и весь город наполнен радостными людьми на улицах."
        )
        self.assertEqual(outcome["ensemble"]["best"], "ru")

    def test_empty_and_very_short_text_do_not_crash(self):
        outcome = self.model.classify_text("a")
        self.assertIn("ensemble", outcome)
        outcome_ws = self.model.classify_text("   ")
        self.assertIn("ensemble", outcome_ws)

    def test_save_and_load_roundtrip_preserves_verdicts(self):
        path = "/tmp/_lang_model_test.json"
        self.model.save(path)
        loaded = LanguageModel.load(path)
        text = "Il cane corre veloce nel parco"
        outcome1 = self.model.classify_text(text)
        outcome2 = loaded.classify_text(text)
        self.assertEqual(outcome1["ensemble"]["best"], outcome2["ensemble"]["best"])
        os.remove(path)

    def test_explain_hits_match_the_predicted_language(self):
        text = "Il gatto nero dorme sul divano vicino alla finestra"
        outcome = self.model.classify_text(text)
        explanation = self.model.explain(text, outcome)
        self.assertEqual(explanation["best"], outcome["ensemble"]["best"])
        self.assertIn("ngram_hits", explanation)
        self.assertIn("alphabet_hits", explanation)
        self.assertIn("neural_hits", explanation)

    def test_learn_from_feedback_returns_boolean_and_does_not_crash(self):
        changed = self.model.learn_from_feedback("Il cane corre veloce nel parco", "it")
        self.assertIn(changed, (True, False))

    def test_learn_from_feedback_rejects_unknown_language(self):
        changed = self.model.learn_from_feedback("Testo qualsiasi", "fr")
        self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
