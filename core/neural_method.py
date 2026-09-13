import math
import random

from collections import Counter
from core.preprocessing import preprocess
from core.corpus import split_train_val, chunk_documents

BIGRAM_VOCAB_SIZE = 180
HIDDEN_SIZE = 12
LEARNING_RATE = 0.15
L2_REG = 1e-4
MOMENTUM = 0.9
MAX_EPOCHS = 220
QUICK_EPOCHS = 80
PATIENCE = 20
CHUNK_CHARS = 380
MIN_CHUNK_CHARS = 120
VAL_RATIO = 0.25
SEED = 42

random.seed(SEED)


def _softmax(logits: list) -> list:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class MLP:

    def __init__(self, n_features: int, n_classes: int, n_hidden: int = HIDDEN_SIZE,
                 lr: float = LEARNING_RATE, l2: float = L2_REG,
                 momentum: float = MOMENTUM, seed: int = SEED):
        rnd = random.Random(seed)
        scale1 = math.sqrt(2.0 / max(n_features, 1))
        scale2 = math.sqrt(2.0 / max(n_hidden, 1))
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2
        self.momentum = momentum
        self.W1 = [[rnd.gauss(0, scale1) for _ in range(n_features)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rnd.gauss(0, scale2) for _ in range(n_hidden)] for _ in range(n_classes)]
        self.b2 = [0.0] * n_classes
        self._vW1 = [[0.0] * n_features for _ in range(n_hidden)]
        self._vb1 = [0.0] * n_hidden
        self._vW2 = [[0.0] * n_hidden for _ in range(n_classes)]
        self._vb2 = [0.0] * n_classes

    def forward(self, x: list) -> tuple:
        h_raw = [self.b1[i] + sum(self.W1[i][j] * x[j] for j in range(self.n_features))
                 for i in range(self.n_hidden)]
        h = [math.tanh(v) for v in h_raw]
        logits = [self.b2[k] + sum(self.W2[k][i] * h[i] for i in range(self.n_hidden))
                  for k in range(self.n_classes)]
        return h, _softmax(logits)

    def predict_proba(self, x: list) -> list:
        _, probs = self.forward(x)
        return probs

    def _example_grads(self, x: list, y_idx: int) -> tuple:
        h, probs = self.forward(x)
        dlogits = list(probs)
        dlogits[y_idx] -= 1.0

        dh = [sum(dlogits[k] * self.W2[k][i] for k in range(self.n_classes))
              for i in range(self.n_hidden)]
        dh_raw = [dh[i] * (1 - h[i] ** 2) for i in range(self.n_hidden)]
        return h, dlogits, dh_raw

    def _train_batch(self, X_batch: list, y_batch: list):
        m = len(X_batch)
        if m == 0:
            return
        gW1 = [[0.0] * self.n_features for _ in range(self.n_hidden)]
        gb1 = [0.0] * self.n_hidden
        gW2 = [[0.0] * self.n_hidden for _ in range(self.n_classes)]
        gb2 = [0.0] * self.n_classes

        for x, y in zip(X_batch, y_batch):
            h, dlogits, dh_raw = self._example_grads(x, y)
            for k in range(self.n_classes):
                row = gW2[k]
                for i in range(self.n_hidden):
                    row[i] += dlogits[k] * h[i]
                gb2[k] += dlogits[k]
            for i in range(self.n_hidden):
                row = gW1[i]
                dhi = dh_raw[i]
                for j in range(self.n_features):
                    row[j] += dhi * x[j]
                gb1[i] += dhi

        inv_m = 1.0 / m
        for k in range(self.n_classes):
            for i in range(self.n_hidden):
                grad = gW2[k][i] * inv_m + self.l2 * self.W2[k][i]
                self._vW2[k][i] = self.momentum * self._vW2[k][i] - self.lr * grad
                self.W2[k][i] += self._vW2[k][i]
            grad_b = gb2[k] * inv_m
            self._vb2[k] = self.momentum * self._vb2[k] - self.lr * grad_b
            self.b2[k] += self._vb2[k]

        for i in range(self.n_hidden):
            for j in range(self.n_features):
                grad = gW1[i][j] * inv_m + self.l2 * self.W1[i][j]
                self._vW1[i][j] = self.momentum * self._vW1[i][j] - self.lr * grad
                self.W1[i][j] += self._vW1[i][j]
            grad_b = gb1[i] * inv_m
            self._vb1[i] = self.momentum * self._vb1[i] - self.lr * grad_b
            self.b1[i] += self._vb1[i]

    def online_update(self, x: list, y_idx: int, lr: float = 0.05):
        h, dlogits, dh_raw = self._example_grads(x, y_idx)
        for k in range(self.n_classes):
            for i in range(self.n_hidden):
                self.W2[k][i] -= lr * (dlogits[k] * h[i] + self.l2 * self.W2[k][i])
            self.b2[k] -= lr * dlogits[k]
        for i in range(self.n_hidden):
            for j in range(self.n_features):
                self.W1[i][j] -= lr * (dh_raw[i] * x[j] + self.l2 * self.W1[i][j])
            self.b1[i] -= lr * dh_raw[i]

    def loss_and_acc(self, X: list, y_idx_list: list) -> tuple:
        loss = 0.0
        correct = 0
        for x, y in zip(X, y_idx_list):
            _, probs = self.forward(x)
            loss -= math.log(max(probs[y], 1e-12))
            if probs.index(max(probs)) == y:
                correct += 1
        n = len(X) or 1
        return loss / n, correct / n

    def fit(self, X: list, y_idx_list: list, X_val: list = None, y_val: list = None,
            epochs: int = MAX_EPOCHS, patience: int = PATIENCE, batch_size: int = 8) -> "MLP":
        n = len(X)
        if n == 0:
            return self
        best_val = math.inf
        best_state = self._snapshot()
        bad_epochs = 0
        order = list(range(n))
        for _ in range(epochs):
            random.shuffle(order)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self._train_batch([X[i] for i in idx], [y_idx_list[i] for i in idx])
            if X_val:
                val_loss, _ = self.loss_and_acc(X_val, y_val)
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_state = self._snapshot()
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
        if X_val:
            self._restore(best_state)
        return self

    def _snapshot(self) -> dict:
        return {
            "W1": [row[:] for row in self.W1], "b1": self.b1[:],
            "W2": [row[:] for row in self.W2], "b2": self.b2[:],
        }

    def _restore(self, state: dict):
        self.W1 = [row[:] for row in state["W1"]]
        self.b1 = state["b1"][:]
        self.W2 = [row[:] for row in state["W2"]]
        self.b2 = state["b2"][:]

    def to_dict(self) -> dict:
        return {
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "n_features": self.n_features, "n_hidden": self.n_hidden,
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        obj = cls(n_features=d["n_features"], n_classes=d["n_classes"],
                   n_hidden=d["n_hidden"], lr=0.0)
        obj.W1 = d["W1"]
        obj.b1 = d["b1"]
        obj.W2 = d["W2"]
        obj.b2 = d["b2"]
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


def train(corpus_by_lang: dict, quick: bool = False) -> tuple:
    langs = list(corpus_by_lang.keys())
    if len(langs) < 2:
        raise ValueError("Нейросетевому методу нужно минимум 2 языка")

    if quick:
        train_docs, val_docs = corpus_by_lang, {}
    else:
        train_docs, val_docs = split_train_val(corpus_by_lang, val_ratio=VAL_RATIO, seed=SEED)

    all_train_texts = [t for texts in train_docs.values() for t in texts]
    vocab = build_bigram_vocab(all_train_texts)

    X_train, y_train = [], []
    for idx, lang in enumerate(langs):
        for chunk in chunk_documents(train_docs.get(lang, []), CHUNK_CHARS, MIN_CHUNK_CHARS):
            X_train.append(text_to_features(chunk, vocab))
            y_train.append(idx)

    X_val, y_val = [], []
    for idx, lang in enumerate(langs):
        for doc in val_docs.get(lang, []):
            X_val.append(text_to_features(doc, vocab))
            y_val.append(idx)

    model = MLP(n_features=len(vocab), n_classes=len(langs))
    if quick:
        model.fit(X_train, y_train, epochs=QUICK_EPOCHS)
    else:
        model.fit(X_train, y_train, X_val or None, y_val or None,
                  epochs=MAX_EPOCHS, patience=PATIENCE)
    return model, vocab, tuple(langs)


def classify(text: str, model: MLP, vocab: list, langs: tuple) -> dict:
    x = text_to_features(text, vocab)
    probs = model.predict_proba(x)
    scores = {lang: p for lang, p in zip(langs, probs)}
    best = max(scores, key=scores.get)
    return {"best": best, "scores": scores, "kind": "probability"}


def top_features(model: MLP, vocab: list, langs: tuple, n: int = 20) -> dict:
    influence = []
    for j in range(model.n_features):
        contrib = [sum(model.W1[i][j] * model.W2[k][i] for i in range(model.n_hidden))
                   for k in range(model.n_classes)]
        influence.append(contrib)

    result = {}
    for k, lang in enumerate(langs):
        order = sorted(range(model.n_features), key=lambda j: -influence[j][k])[:n]
        result[lang] = [(vocab[j], influence[j][k]) for j in order]
    return result
