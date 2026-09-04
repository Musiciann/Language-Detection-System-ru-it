import os
from config import LANG_NAMES, REPORTS_DIR
from web.templating import render, render_page, escape

def index_page(model) -> str:
    corpus_info = " &nbsp;·&nbsp; ".join(
        f"{LANG_NAMES[l]}: {model.meta['corpus_docs'][l]} документов, {model.meta['corpus_chars'][l]} символов"
        for l in model.languages
    )
    content = render("index.html", meta_row=corpus_info)
    return render_page(content, title="Определить язык")

def _bar_row(label: str, value_pct: float, extra: str = "") -> str:
    return (f'<div class="bar-row"><span class="bar-label">{escape(label)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{value_pct:.1f}%"></div></div>'
            f'<span class="bar-value">{extra}</span></div>')


def _corpus_cards(model) -> str:
    cards = []
    for lang in model.languages:
        cards.append(f'''
        <div class="stat-card">
          <div class="stat-method">{LANG_NAMES[lang]}</div>
          <div class="stat-acc">{model.meta["corpus_docs"][lang]}</div>
          <div class="stat-sub">документов в обучающем корпусе</div>
          <div class="stat-sub">{model.meta["corpus_chars"][lang]} символов</div>
        </div>''')
    return "".join(cards)


def _ngram_tables(model) -> str:
    cols = []
    for lang in model.languages:
        rows = "".join(
            f'<tr><td>{i+1}</td><td class="mono">{escape(g)}</td></tr>'
            for i, g in enumerate(model.top_ngrams(lang, 30))
        )
        cols.append(f'''
        <div>
          <h3>{LANG_NAMES[lang]}</h3>
          <table><thead><tr><th>#</th><th>N-грамма</th></tr></thead><tbody>{rows}</tbody></table>
        </div>''')
    return "".join(cols)


def _alphabet_tables(model) -> str:
    cols = []
    for lang in model.languages:
        top = model.alphabet_top(lang, 20)
        max_freq = max((f for _, f in top), default=1) or 1
        rows = "".join(_bar_row(ch, f/max_freq*100, f"{f*100:.2f}%") for ch, f in top)
        cols.append(f'<div><h3>{LANG_NAMES[lang]}</h3><div class="bars">{rows}</div></div>')
    return "".join(cols)


def _neural_tables(model) -> str:
    feats = model.neural_top_features(20)
    cols = []
    for lang, pairs in feats.items():
        max_w = max((abs(w) for _, w in pairs), default=1) or 1
        rows = "".join(
            _bar_row(g, abs(w)/max_w*100, f"{w:+.3f}") for g, w in pairs
        )
        cols.append(f'<div><h3>{LANG_NAMES[lang]}</h3><div class="bars">{rows}</div></div>')
    return "".join(cols)


def _accuracy_cards(model) -> str:
    labels = {"ngram": "N-грамм", "alphabet": "Алфавитный", "neural": "Нейросетевой"}
    cards = []
    for key, label in labels.items():
        acc = model.accuracy.get(key, 0) * 100
        t = model.times_ms.get(key, 0)
        cards.append(f'''
        <div class="stat-card">
          <div class="stat-method">{label}</div>
          <div class="stat-acc">{acc:.0f}%</div>
          <div class="stat-sub">точность на тестовой коллекции</div>
          <div class="stat-sub">{t:.2f} мс суммарно</div>
        </div>''')
    return "".join(cards)


def _test_rows(model) -> str:
    def chip(r, true_lang):
        cls = "match" if r["best"] == true_lang else "mismatch"
        return f'<span class="tag {r["best"]} {cls}">{LANG_NAMES[r["best"]]}</span>'

    rows = []
    for doc in model.test_results:
        rows.append(f'''
        <tr>
          <td>{escape(doc["file"])}</td>
          <td><span class="tag {doc["true"]}">{LANG_NAMES.get(doc["true"], "?")}</span></td>
          <td>{chip(doc["ngram"], doc["true"])}</td>
          <td>{chip(doc["alphabet"], doc["true"])}</td>
          <td>{chip(doc["neural"], doc["true"])}</td>
        </tr>''')
    return "".join(rows)


def model_page(model) -> str:
    content = render(
        "model_info.html",
        corpus_cards=_corpus_cards(model),
        trained_at=model.meta.get("trained_at", "?"),
        bigram_vocab_size=str(model.meta.get("bigram_vocab_size", "?")),
        top_n_grams=str(model.meta.get("top_n_grams", "?")),
        max_n=str(model.meta.get("max_n", "?")),
        ngram_tables=_ngram_tables(model),
        alphabet_tables=_alphabet_tables(model),
        neural_tables=_neural_tables(model),
        accuracy_cards=_accuracy_cards(model),
        test_count=str(len(model.test_results)),
        test_rows=_test_rows(model),
    )
    return render_page(content, title="Модель и обучение")

def task_page() -> str:
    content = render("task.html")
    return render_page(content, title="Условия работы")

def reports_list_page() -> str:
    rows = []
    if os.path.isdir(REPORTS_DIR):
        entries = sorted(
            (e for e in os.scandir(REPORTS_DIR) if e.is_dir()),
            key=lambda e: e.stat().st_mtime, reverse=True,
        )
        for e in entries:
            meta_path = os.path.join(e.path, "meta.txt")
            item_count = "?"
            created = "?"
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    d = dict(line.strip().split("=", 1) for line in f if "=" in line)
                    item_count = d.get("item_count", "?")
                    created = d.get("created", "?")
            rows.append(f'''
            <tr>
              <td><a href="/reports/{e.name}/report.html">{e.name}</a></td>
              <td>{escape(created)}</td>
              <td>{escape(item_count)}</td>
            </tr>''')

    empty_hint = "" if rows else '<p class="empty-hint">Пока нет ни одного отчёта — сформируйте его на главной странице.</p>'
    content = render("reports_list.html", reports_rows="".join(rows), empty_hint=empty_hint)
    return render_page(content, title="Отчёты")
