import math
from collections import Counter
from core.preprocessing import preprocess, extract_letters


def build_profile(text: str) -> dict:
    letters = extract_letters(preprocess(text))
    total = len(letters) or 1
    counts = Counter(letters)
    return {ch: cnt / total for ch, cnt in counts.items()}


def distance(profile_doc: dict, profile_lang: dict) -> float:
    keys = set(profile_doc) | set(profile_lang)
    return math.sqrt(sum(
        (profile_doc.get(k, 0.0) - profile_lang.get(k, 0.0)) ** 2 for k in keys
    ))


def classify(text: str, lang_profiles: dict) -> dict:
    doc_profile = build_profile(text)
    scores = {lang: distance(doc_profile, prof) for lang, prof in lang_profiles.items()}
    best = min(scores, key=scores.get)
    return {"best": best, "scores": scores, "kind": "distance"}
