import os
import re
import json
import datetime

from config import LANG_NAMES, REPORTS_DIR, CORPUS_DIR, TEST_DIR, MODEL_PATH, MAX_UPLOAD_TEXT_CHARS
from core.model import LanguageModel

ALL_KEYS = ("ngram", "alphabet", "neural", "ensemble")
METHOD_LABELS = {
    "ngram": "N-грамм",
    "alphabet": "Алфавитный",
    "neural": "Нейросетевой (MLP)",
    "ensemble": "Ансамбль (взвеш. голосование)",
}
from web.templating import render, escape

_SAFE_NAME_RE = re.compile(r"[^A-Za-zА-Яа-яЁё0-9._-]+")


def _safe_filename(name: str, fallback: str) -> str:
    name = os.path.basename(name or "").strip() or fallback
    name = _SAFE_NAME_RE.sub("_", name)
    if not name.lower().endswith(".txt"):
        name += ".txt"
    return name[:120]


def _method_result_json(key: str, r: dict) -> dict:
    d = {"best": r["best"], "scores": r["scores"], "kind": r["kind"], "elapsed_ms": round(r["elapsed_ms"], 3)}
    if key == "ensemble":
        d["agreement"] = r.get("agreement")
        d["margin"] = r.get("margin")
        d["confidence_level"] = r.get("confidence_level")
        d["weights"] = r.get("weights")
    return d


def classify(model: LanguageModel, body: dict) -> dict:
    text = (body or {}).get("text", "")
    if not isinstance(text, str):
        raise ValueError("Поле 'text' должно быть строкой")
    text = text[:MAX_UPLOAD_TEXT_CHARS]
    if not text.strip():
        return {"empty": True}
    outcome = model.classify_text(text)
    result = {k: _method_result_json(k, v) for k, v in outcome.items()}
    result["explanation"] = model.explain(text, outcome)
    return result


def upload(model: LanguageModel, body: dict) -> dict:
    items = (body or {}).get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Ожидался непустой список 'items'")

    results = []
    for item in items:
        fname = str(item.get("filename", "документ.txt"))
        text = str(item.get("text", ""))[:MAX_UPLOAD_TEXT_CHARS]
        if not text.strip():
            results.append({"filename": fname, "error": "пустой файл"})
            continue
        outcome = model.classify_text(text)
        results.append({
            "filename": fname,
            "results": {k: _method_result_json(k, v) for k, v in outcome.items()},
        })
    return {"items": results}


def feedback(model: LanguageModel, body: dict) -> dict:
    text = (body or {}).get("text", "")
    lang = (body or {}).get("lang", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Поле 'text' пустое")
    if lang not in LANG_NAMES:
        raise ValueError("Поле 'lang' должно быть одним из: " + ", ".join(LANG_NAMES))
    text = text[:MAX_UPLOAD_TEXT_CHARS]
    changed = model.learn_from_feedback(text, lang)
    if changed:
        model.save(MODEL_PATH)
    return {"ok": True, "applied": changed}


def _method_cell(r: dict) -> str:
    lang = r["best"]
    scores = r["scores"]
    if r["kind"] == "probability":
        parts = " ".join(f"{LANG_NAMES.get(l,l)[:2].lower()}:{v:.3f}" for l, v in scores.items())
    else:
        parts = " ".join(f"{LANG_NAMES.get(l,l)[:2].lower()}:{v:.0f}" for l, v in scores.items())
    extra = f' · увер.: {r["confidence_level"]}' if r.get("confidence_level") else ""
    return (f'<span class="tag {lang}">{LANG_NAMES.get(lang, lang)}</span>'
            f'<div class="scores">{parts}{extra} · {r["elapsed_ms"]:.2f} мс</div>')


def generate_report(model: LanguageModel, body: dict) -> dict:
    items = (body or {}).get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Пакет пуст — нечего включать в отчёт")

    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H%M%S")
    report_dir = os.path.join(REPORTS_DIR, folder_name)
    attachments_dir = os.path.join(report_dir, "attachments")
    os.makedirs(attachments_dir, exist_ok=True)

    rows_html = []
    correct = {k: 0 for k in ALL_KEYS}
    labeled_count = 0
    totals_ms = {k: 0.0 for k in ALL_KEYS}
    used_names = set()

    for idx, item in enumerate(items):
        raw_name = str(item.get("filename") or f"текст_{idx+1}")
        text = str(item.get("text", ""))[:MAX_UPLOAD_TEXT_CHARS]
        expected = str(item.get("expected") or "").strip()
        if expected not in LANG_NAMES:
            expected = ""

        safe_name = _safe_filename(raw_name, f"текст_{idx+1}.txt")
        base, ext = os.path.splitext(safe_name)
        n = 1
        while safe_name in used_names:
            n += 1
            safe_name = f"{base}_{n}{ext}"
        used_names.add(safe_name)

        with open(os.path.join(attachments_dir, safe_name), "w", encoding="utf-8") as f:
            f.write(text)

        outcome = model.classify_text(text) if text.strip() else None

        if outcome:
            for key in totals_ms:
                totals_ms[key] += outcome[key]["elapsed_ms"]
            if expected:
                labeled_count += 1
                for key in correct:
                    correct[key] += int(outcome[key]["best"] == expected)

        expected_cell = f'<span class="tag {expected}">{LANG_NAMES[expected]}</span>' if expected else "—"
        if outcome:
            method_cells = "".join(f"<td>{_method_cell(outcome[k])}</td>" for k in ALL_KEYS)
        else:
            method_cells = '<td colspan="4">пустой документ — пропущен</td>'

        rows_html.append(f'''
        <tr>
          <td><a href="attachments/{safe_name}" target="_blank">{escape(raw_name)}</a></td>
          <td>{expected_cell}</td>
          {method_cells}
        </tr>''')

    n_items = len(items)
    if labeled_count:
        acc_rows = "".join(
            f"<tr><td>{METHOD_LABELS[key]}</td><td>{correct[key]}/{labeled_count} ({correct[key]/labeled_count*100:.0f}%)</td>"
            f"<td>{totals_ms[key]:.2f} мс</td><td>{totals_ms[key]/max(n_items,1):.2f} мс</td></tr>"
            for key in ALL_KEYS
        )
        summary_block = f'''
        <p>Документов с указанным ожидаемым языком: {labeled_count} из {n_items}.</p>
        <table>
          <thead><tr><th>Метод</th><th>Точность</th><th>Суммарное время</th><th>Среднее время/документ</th></tr></thead>
          <tbody>{acc_rows}</tbody>
        </table>'''
    else:
        time_rows = "".join(
            f"<tr><td>{METHOD_LABELS[key]}</td><td>{totals_ms[key]:.2f} мс</td><td>{totals_ms[key]/max(n_items,1):.2f} мс</td></tr>"
            for key in ALL_KEYS
        )
        summary_block = f'''
        <p>Ожидаемый язык не указан ни для одного документа — точность не считается, только время обработки.</p>
        <table>
          <thead><tr><th>Метод</th><th>Суммарное время</th><th>Среднее время/документ</th></tr></thead>
          <tbody>{time_rows}</tbody>
        </table>'''

    corpus_summary = ", ".join(
        f"{LANG_NAMES[l]} — {model.meta['corpus_docs'][l]} докум., {model.meta['corpus_chars'][l]} симв."
        for l in model.languages
    )
    cv_summary = ", ".join(
        f"{METHOD_LABELS[k]}: {model.cv_accuracy.get(k, 0)*100:.0f}%" for k in ALL_KEYS
    ) if model.cv_accuracy else "не вычислена"

    report_html = render(
        "report.html",
        generated_at=now.strftime("%d.%m.%Y %H:%M:%S"),
        item_count=str(n_items),
        trained_at=model.meta.get("trained_at", "?"),
        corpus_summary=corpus_summary,
        top_n_grams=str(model.meta.get("top_n_grams", "?")),
        max_n=str(model.meta.get("max_n", "?")),
        bigram_vocab_size=str(model.meta.get("bigram_vocab_size", "?")),
        cv_summary=cv_summary,
        items_rows="".join(rows_html),
        summary_block=summary_block,
    )
    report_path = os.path.join(report_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    with open(os.path.join(report_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"created={now.strftime('%d.%m.%Y %H:%M:%S')}\nitem_count={n_items}\n")

    return {"url": f"/reports/{folder_name}/report.html", "folder": folder_name}


def retrain() -> LanguageModel:
    model = LanguageModel.train(CORPUS_DIR, TEST_DIR)
    model.save(MODEL_PATH)
    return model
