import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import neural_method


class TestNeuralMethod(unittest.TestCase):
    def test_mlp_forward_returns_valid_probability_distribution(self):
        model = neural_method.MLP(n_features=5, n_classes=2, n_hidden=4, seed=1)
        x = [0.1, 0.0, 0.3, 0.2, 0.0]
        _, probs = model.forward(x)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_training_reduces_loss_on_separable_toy_data(self):
        X = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
        y = [0, 0, 0, 1, 1, 1]
        model = neural_method.MLP(n_features=2, n_classes=2, n_hidden=3, lr=0.3, seed=0)
        loss_before, _ = model.loss_and_acc(X, y)
        model.fit(X, y, epochs=150, batch_size=3)
        loss_after, acc_after = model.loss_and_acc(X, y)
        self.assertLess(loss_after, loss_before)
        self.assertGreaterEqual(acc_after, 0.8)

    def test_train_end_to_end_on_tiny_synthetic_corpus(self):
        corpus = {
            "ru": ["привет как твои дела " * 20, "хорошо у меня всё отлично " * 20],
            "it": ["ciao come stai bene " * 20, "tutto molto bene grazie " * 20],
        }
        model, vocab, langs = neural_method.train(corpus, quick=True)
        self.assertEqual(set(langs), {"ru", "it"})
        result = neural_method.classify("ciao bene grazie come stai", model, vocab, langs)
        self.assertIn(result["best"], langs)
        self.assertEqual(result["kind"], "probability")
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0, places=6)

    def test_serialization_roundtrip_preserves_predictions(self):
        model = neural_method.MLP(n_features=3, n_classes=2, n_hidden=2, seed=2)
        d = model.to_dict()
        restored = neural_method.MLP.from_dict(d)
        x = [0.2, 0.1, 0.0]
        self.assertEqual(model.forward(x)[1], restored.forward(x)[1])

    def test_online_update_moves_prediction_toward_target_class(self):
        corpus = {
            "ru": ["привет как твои дела " * 15],
            "it": ["ciao come stai bene " * 15],
        }
        model, vocab, langs = neural_method.train(corpus, quick=True)
        text = "ciao bene"
        x = neural_method.text_to_features(text, vocab)
        before = model.predict_proba(x)
        y_idx = langs.index("ru")
        model.online_update(x, y_idx, lr=0.2)
        after = model.predict_proba(x)
        self.assertGreater(after[y_idx], before[y_idx])


if __name__ == "__main__":
    unittest.main()
