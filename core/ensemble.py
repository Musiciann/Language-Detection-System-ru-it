import math


def _distance_to_confidence(scores: dict) -> dict:
    vals = list(scores.values())
    lo = min(vals)
    scale = (max(vals) - lo) or 1.0
    normed = {k: -(v - lo) / scale for k, v in scores.items()}
    exps = {k: math.exp(v) for k, v in normed.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def method_confidence(result: dict) -> dict:
    if result["kind"] == "probability":
        return dict(result["scores"])
    return _distance_to_confidence(result["scores"])


def weights_from_accuracy(accuracy: dict, sharpness: float = 4.0, floor: float = 0.08) -> dict:
    if not accuracy:
        return {}
    exps = {m: math.exp(sharpness * a) for m, a in accuracy.items()}
    total = sum(exps.values()) or 1.0
    weights = {m: v / total for m, v in exps.items()}
    n = len(weights)
    return {m: (1 - floor) * w + floor / n for m, w in weights.items()}


def combine(method_results: dict, weights: dict = None) -> dict:
    n_methods = len(method_results) or 1
    weights = weights or {m: 1.0 / n_methods for m in method_results}

    confidences = {m: method_confidence(r) for m, r in method_results.items()}
    langs = set()
    for c in confidences.values():
        langs |= set(c.keys())

    combined = {lang: 0.0 for lang in langs}
    for method, conf in confidences.items():
        w = weights.get(method, 1.0 / n_methods)
        for lang, p in conf.items():
            combined[lang] += w * p

    total = sum(combined.values()) or 1.0
    combined = {lang: v / total for lang, v in combined.items()}
    best = max(combined, key=combined.get)

    verdicts = [r["best"] for r in method_results.values()]
    agreement = verdicts.count(best) / len(verdicts) if verdicts else 0.0

    sorted_probs = sorted(combined.values(), reverse=True)
    margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

    if agreement == 1.0 and margin > 0.35:
        confidence_level = "высокая"
    elif agreement >= 0.5 and margin > 0.12:
        confidence_level = "средняя"
    else:
        confidence_level = "низкая"

    return {
        "best": best,
        "scores": combined,
        "kind": "probability",
        "agreement": round(agreement, 3),
        "margin": round(margin, 4),
        "confidence_level": confidence_level,
        "weights": weights,
    }
