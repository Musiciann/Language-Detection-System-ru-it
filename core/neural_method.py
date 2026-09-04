import math
import random
from collections import Counter
from core.preprocessing import preprocess

BIGRAM_VOCAB_SIZE = 200
LEARNING_RATE = 0.5
EPOCHS = 300

random.seed(42)


class Perceptron:

    def __init__(self, n_features: int, lr: float = LEARNING_RATE, epochs: int = EPOCHS):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, x: list) -> float:
        z = self.b + sum(wi * xi for wi, xi in zip(self.w, x))
        return self._sigmoid(z)

    def fit(self, X: list, y: list):
        n = len(X)
        for _ in range(self.epochs):
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                x, target = X[i], y[i]
                pred = self.predict_proba(x)
                error = pred - target
                for j in range(len(self.w)):
                    self.w[j] -= self.lr * error * x[j]
                self.b -= self.lr * error

    def to_dict(self) -> dict:
        return {"weights": self.w, "bias": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "Perceptron":
        obj = cls(n_features=len(d["weights"]), epochs=0)
        obj.w = d["weights"]
        obj.b = d["bias"]
        return obj


def build_bigram_vocab(texts: list, top_k: int = BIGRAM_VOCAB_SIZE) -> list:
    counter = Counter()
    for t in texts:
        t = preprocess(t)
        counter.update(t[i:i + 2] for i in range(len(t) - 1))
    return [g for g, _ in counter.most_common(top_k)]


def text_to_features(text: str, vocab: list) -> list:
    t = preprocess(text)
    total = max(len(t) - 1, 1)
    counter = Counter(t[i:i + 2] for i in range(len(t) - 1))
    return [counter.get(g, 0) / total for g in vocab]


def train(corpus_by_lang: dict) -> tuple:
    langs = list(corpus_by_lang.keys())
    if len(langs) != 2:
        raise ValueError("Нейросетевой метод поддерживает ровно 2 языка")
    lang_a, lang_b = langs

    all_texts = corpus_by_lang[lang_a] + corpus_by_lang[lang_b]
    vocab = build_bigram_vocab(all_texts)

    X, y = [], []
    for t in corpus_by_lang[lang_a]:
        X.append(text_to_features(t, vocab)); y.append(0.0)
    for t in corpus_by_lang[lang_b]:
        X.append(text_to_features(t, vocab)); y.append(1.0)

    model = Perceptron(n_features=len(vocab))
    model.fit(X, y)
    return model, vocab, (lang_a, lang_b)


def classify(text: str, model: Perceptron, vocab: list, langs: tuple) -> dict:
    x = text_to_features(text, vocab)
    p = model.predict_proba(x)
    lang_a, lang_b = langs
    best = lang_b if p >= 0.5 else lang_a
    return {"best": best, "scores": {lang_a: 1 - p, lang_b: p}, "kind": "probability"}


def top_features(model: Perceptron, vocab: list, langs: tuple, n: int = 20) -> dict:
    lang_a, lang_b = langs
    pairs = sorted(zip(vocab, model.w), key=lambda p: p[1])
    toward_a = [(g, w) for g, w in pairs if w < 0][:n]
    toward_b = list(reversed([(g, w) for g, w in pairs if w > 0][-n:]))
    return {lang_a: toward_a, lang_b: toward_b}
