from collections import Counter
from core.preprocessing import preprocess, tokenize_words

TOP_N_GRAMS = 300
MAX_N = 5


def extract_ngrams(word: str, max_n: int = MAX_N) -> list:
    padded = "_" * (max_n - 1) + word + "_" * (max_n - 1)
    grams = []
    for n in range(1, max_n + 1):
        for i in range(len(padded) - n + 1):
            g = padded[i:i + n]
            if g.strip("_") == "":
                continue
            grams.append(g)
    return grams


def build_profile(text: str, top_k: int = TOP_N_GRAMS) -> list:
    words = tokenize_words(preprocess(text))
    counter = Counter()
    for w in words:
        counter.update(extract_ngrams(w))
    return [gram for gram, _ in counter.most_common(top_k)]


def distance(profile_doc: list, profile_lang: list) -> int:
    max_penalty = len(profile_lang)
    rank = {gram: i for i, gram in enumerate(profile_lang)}
    total = 0
    for i, gram in enumerate(profile_doc):
        total += abs(i - rank[gram]) if gram in rank else max_penalty
    return total


def classify(text: str, lang_profiles: dict) -> dict:
    doc_profile = build_profile(text)
    scores = {lang: distance(doc_profile, prof) for lang, prof in lang_profiles.items()}
    best = min(scores, key=scores.get)
    return {"best": best, "scores": scores, "kind": "distance"}
