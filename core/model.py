import time
import json
import datetime

from core import ngram_method, alphabet_method, neural_method
from core.corpus import load_corpus, load_test_documents
from config import LANG_NAMES


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

        m.meta = {
            "languages": LANG_NAMES,
            "corpus_docs": {lang: len(texts) for lang, texts in corpus_files.items()},
            "corpus_chars": {lang: len(t) for lang, t in corpus_text.items()},
            "top_n_grams": ngram_method.TOP_N_GRAMS,
            "max_n": ngram_method.MAX_N,
            "bigram_vocab_size": len(m.neural_vocab),
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        m._evaluate_on_test_set(test_dir)
        return m

    def _evaluate_on_test_set(self, test_dir):
        docs = load_test_documents(test_dir)
        correct = {"ngram": 0, "alphabet": 0, "neural": 0}
        times = {"ngram": 0.0, "alphabet": 0.0, "neural": 0.0}
        results = []

        for doc in docs:
            true_lang = _true_lang_from_filename(doc["file"])
            outcome = self.classify_text(doc["text"])
            for key in ("ngram", "alphabet", "neural"):
                correct[key] += int(outcome[key]["best"] == true_lang)
                times[key] += outcome[key]["elapsed_ms"]
            results.append({
                "file": doc["file"],
                "text": doc["text"],
                "true": true_lang,
                "ngram": outcome["ngram"],
                "alphabet": outcome["alphabet"],
                "neural": outcome["neural"],
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

        return out

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
                "lang_a": self.neural_langs[0],
                "lang_b": self.neural_langs[1],
            },
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
        m.neural_langs = (d["neural"]["lang_a"], d["neural"]["lang_b"])
        m.neural_vocab = d["neural"]["vocab"]
        m.neural_model = neural_method.Perceptron.from_dict(d["neural"])
        m.test_results = d["test_results"]
        m.accuracy = d["accuracy"]
        m.times_ms = d["times_ms"]
        return m
