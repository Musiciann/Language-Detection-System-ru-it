import time
import json
import random
import datetime

from core import ngram_method, alphabet_method, neural_method, ensemble
from core.corpus import load_corpus, load_test_documents
from core.preprocessing import preprocess
from config import LANG_NAMES

METHOD_KEYS = ("ngram", "alphabet", "neural")
CV_FOLDS = 4


def _true_lang_from_filename(fname: str) -> str:
    for code in LANG_NAMES:
        if fname.startswith(f"doc_{code}"):
            return code
    return "?"


class LanguageModel:
    def __init__(self):
        self.languages = list(LANG_NAMES.keys())
        self.ngram_profiles = {}
        self.alphabet_profiles = {}
        self.neural_model = None
        self.neural_vocab = []
        self.neural_langs = None
        self.ensemble_weights = {}
        self.cv_accuracy = {}
        self.meta = {}
        self.test_results = []
        self.accuracy = {}
        self.times_ms = {}

    @classmethod
    def train(cls, corpus_dir, test_dir) -> "LanguageModel":
        m = cls()
        corpus_files = load_corpus(corpus_dir)          # {lang: [text, ...]}
        corpus_text = {lang: "\n".join(texts) for lang, texts in corpus_files.items()}

        m.ngram_profiles = {lang: ngram_method.build_profile(t) for lang, t in corpus_text.items()}
        m.alphabet_profiles = {lang: alphabet_method.build_profile(t) for lang, t in corpus_text.items()}
        m.neural_model, m.neural_vocab, m.neural_langs = neural_method.train(corpus_files)

        m.cv_accuracy = m._cross_validate(corpus_files, k=CV_FOLDS)
        m.ensemble_weights = ensemble.weights_from_accuracy(
            {k: m.cv_accuracy[k] for k in METHOD_KEYS if k in m.cv_accuracy}
        )

        m.meta = {
            "languages": LANG_NAMES,
            "corpus_docs": {lang: len(texts) for lang, texts in corpus_files.items()},
            "corpus_chars": {lang: len(t) for lang, t in corpus_text.items()},
            "top_n_grams": ngram_method.TOP_N_GRAMS,
            "max_n": ngram_method.MAX_N,
            "bigram_vocab_size": len(m.neural_vocab),
            "hidden_size": neural_method.HIDDEN_SIZE,
            "cv_folds": CV_FOLDS,
            "cv_accuracy": m.cv_accuracy,
            "ensemble_weights": m.ensemble_weights,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        m._evaluate_on_test_set(test_dir)
        return m

    def _cross_validate(self, corpus_files: dict, k: int = CV_FOLDS) -> dict:
        max_k = min(len(texts) for texts in corpus_files.values()) if corpus_files else 1
        k = max(2, min(k, max_k))

        rnd = random.Random(123)
        folds = {lang: list(texts) for lang, texts in corpus_files.items()}
        for lang in folds:
            rnd.shuffle(folds[lang])

        correct = {key: 0 for key in (*METHOD_KEYS, "ensemble")}
        total = 0
        equal_weights = {key: 1.0 / len(METHOD_KEYS) for key in METHOD_KEYS}

        for fold_idx in range(k):
            train_part, test_part = {}, []
            for lang, texts in folds.items():
                n = len(texts)
                lo = fold_idx * n // k
                hi = (fold_idx + 1) * n // k
                held_out = texts[lo:hi] or texts[:1]
                remaining = [t for t in texts if t not in held_out]
                train_part[lang] = remaining or texts
                for t in held_out:
                    test_part.append((lang, t))

            corpus_text = {lang: "\n".join(texts) for lang, texts in train_part.items()}
            ngram_profiles = {l: ngram_method.build_profile(t) for l, t in corpus_text.items()}
            alphabet_profiles = {l: alphabet_method.build_profile(t) for l, t in corpus_text.items()}
            neural_model, neural_vocab, neural_langs = neural_method.train(train_part, quick=True)

            for true_lang, text in test_part:
                r_ngram = ngram_method.classify(text, ngram_profiles)
                r_alpha = alphabet_method.classify(text, alphabet_profiles)
                r_neural = neural_method.classify(text, neural_model, neural_vocab, neural_langs)
                r_ens = ensemble.combine(
                    {"ngram": r_ngram, "alphabet": r_alpha, "neural": r_neural}, equal_weights
                )
                correct["ngram"] += int(r_ngram["best"] == true_lang)
                correct["alphabet"] += int(r_alpha["best"] == true_lang)
                correct["neural"] += int(r_neural["best"] == true_lang)
                correct["ensemble"] += int(r_ens["best"] == true_lang)
                total += 1

        total = total or 1
        return {key: correct[key] / total for key in correct}

    def _evaluate_on_test_set(self, test_dir):
        docs = load_test_documents(test_dir)
        keys = (*METHOD_KEYS, "ensemble")
        correct = {key: 0 for key in keys}
        times = {key: 0.0 for key in keys}
        results = []

        for doc in docs:
            true_lang = _true_lang_from_filename(doc["file"])
            outcome = self.classify_text(doc["text"])
            for key in keys:
                correct[key] += int(outcome[key]["best"] == true_lang)
                times[key] += outcome[key]["elapsed_ms"]
            results.append({
                "file": doc["file"],
                "text": doc["text"],
                "true": true_lang,
                **{key: outcome[key] for key in keys},
            })

        n = len(docs) or 1
        self.test_results = results
        self.accuracy = {k: correct[k] / n for k in correct}
        self.times_ms = times

    def classify_text(self, text: str) -> dict:
        out = {}

        t0 = time.perf_counter()
        r = ngram_method.classify(text, self.ngram_profiles)
        r["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        out["ngram"] = r

        t0 = time.perf_counter()
        r = alphabet_method.classify(text, self.alphabet_profiles)
        r["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        out["alphabet"] = r

        t0 = time.perf_counter()
        r = neural_method.classify(text, self.neural_model, self.neural_vocab, self.neural_langs)
        r["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        out["neural"] = r

        t0 = time.perf_counter()
        weights = self.ensemble_weights or {k: 1.0 / len(METHOD_KEYS) for k in METHOD_KEYS}
        r = ensemble.combine({k: out[k] for k in METHOD_KEYS}, weights)
        r["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        out["ensemble"] = r

        return out

    def explain(self, text: str, outcome: dict = None) -> dict:
        outcome = outcome or self.classify_text(text)
        best = outcome["ensemble"]["best"]
        others = [l for l in self.languages if l != best]

        doc_ngrams = ngram_method.build_profile(text, top_k=60)
        best_top_ngrams = set(self.ngram_profiles.get(best, [])[:60])
        ngram_hits = [g for g in doc_ngrams if g in best_top_ngrams][:8]

        doc_alpha = alphabet_method.build_profile(text)
        alpha_hits = sorted(
            doc_alpha.items(),
            key=lambda kv: -(kv[1] - max(
                (self.alphabet_profiles.get(l, {}).get(kv[0], 0.0) for l in others), default=0.0
            )),
        )
        alpha_hits = [ch for ch, diff in alpha_hits if diff > 0][:8]

        neural_infl = neural_method.top_features(self.neural_model, self.neural_vocab, self.neural_langs, n=200)
        t = preprocess(text)
        text_bigrams = {t[i:i + 2] for i in range(len(t) - 1)}
        neural_hits = [g for g, _ in neural_infl.get(best, []) if g in text_bigrams][:8]

        return {
            "best": best,
            "ngram_hits": ngram_hits,
            "alphabet_hits": alpha_hits,
            "neural_hits": neural_hits,
        }

    def learn_from_feedback(self, text: str, true_lang: str) -> bool:
        if not text.strip() or true_lang not in self.neural_langs:
            return False
        x = neural_method.text_to_features(text, self.neural_vocab)
        y_idx = self.neural_langs.index(true_lang)
        probs = self.neural_model.predict_proba(x)
        predicted_idx = probs.index(max(probs))
        if predicted_idx == y_idx and max(probs) > 0.75:
            return False
        self.neural_model.online_update(x, y_idx)
        return True

    def top_ngrams(self, lang: str, n: int = 30) -> list:
        return self.ngram_profiles.get(lang, [])[:n]

    def alphabet_top(self, lang: str, n: int = 20) -> list:
        profile = self.alphabet_profiles.get(lang, {})
        return sorted(profile.items(), key=lambda kv: -kv[1])[:n]

    def neural_top_features(self, n: int = 20) -> dict:
        return neural_method.top_features(self.neural_model, self.neural_vocab, self.neural_langs, n)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "ngram_profiles": self.ngram_profiles,
            "alphabet_profiles": self.alphabet_profiles,
            "neural": {
                **self.neural_model.to_dict(),
                "vocab": self.neural_vocab,
                "langs": list(self.neural_langs),
            },
            "ensemble_weights": self.ensemble_weights,
            "cv_accuracy": self.cv_accuracy,
            "test_results": self.test_results,
            "accuracy": self.accuracy,
            "times_ms": self.times_ms,
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LanguageModel":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = cls()
        m.meta = d["meta"]
        m.ngram_profiles = d["ngram_profiles"]
        m.alphabet_profiles = d["alphabet_profiles"]
        m.neural_langs = tuple(d["neural"]["langs"])
        m.neural_vocab = d["neural"]["vocab"]
        m.neural_model = neural_method.MLP.from_dict(d["neural"])
        m.ensemble_weights = d.get("ensemble_weights", {})
        m.cv_accuracy = d.get("cv_accuracy", {})
        m.test_results = d["test_results"]
        m.accuracy = d["accuracy"]
        m.times_ms = d["times_ms"]
        return m
