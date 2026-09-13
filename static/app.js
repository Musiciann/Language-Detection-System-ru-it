(function () {
  "use strict";

  const LANG_NAMES = { ru: "Русский", it: "Итальянский" };
  const METHOD_LABELS = {
    ngram: "Метод N-грамм",
    alphabet: "Алфавитный метод",
    neural: "Нейросетевой метод (MLP)",
    ensemble: "Итоговое решение (ансамбль)",
  };

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  async function postJSON(url, data) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || `Ошибка запроса (${res.status})`);
    return body;
  }

  document.querySelectorAll(".nav-links a").forEach((a) => {
    if (a.dataset.path === window.location.pathname) a.classList.add("active");
  });

  function scoresToPercents(scores, kind) {
    const langs = Object.keys(scores);
    if (kind === "probability") {
      const out = {};
      langs.forEach((l) => (out[l] = scores[l] * 100));
      return out;
    }
    const inv = {};
    let sum = 0;
    langs.forEach((l) => { inv[l] = 1 / (scores[l] + 1e-9); sum += inv[l]; });
    const out = {};
    langs.forEach((l) => (out[l] = (inv[l] / sum) * 100));
    return out;
  }

  function renderResultCard(methodKey, r) {
    const pct = scoresToPercents(r.scores, r.kind);
    const ruPct = (pct.ru || 0).toFixed(1);
    const itPct = (pct.it || 0).toFixed(1);
    const isEnsemble = methodKey === "ensemble";
    const card = document.createElement("div");
    card.className = "result-card" + (isEnsemble ? " result-card-ensemble" : "");
    const confBadge = isEnsemble
      ? `<span class="conf-badge conf-${r.confidence_level === "высокая" ? "high" : r.confidence_level === "средняя" ? "mid" : "low"}">уверенность: ${r.confidence_level}</span>`
      : "";
    const agreementLine = isEnsemble
      ? `<div class="result-time">согласие методов: ${Math.round((r.agreement || 0) * 100)}% · отрыв лидера: ${((r.margin || 0) * 100).toFixed(1)} п.п.</div>`
      : "";
    card.innerHTML = `
      <div class="result-head">
        <span class="result-method">${METHOD_LABELS[methodKey]}</span>
        <span class="result-verdict ${r.best}">${LANG_NAMES[r.best]}</span>
      </div>
      ${confBadge}
      <div class="bar-track">
        <div class="bar-fill ru" style="width:${ruPct}%"></div>
        <div class="bar-fill it" style="width:${itPct}%"></div>
      </div>
      <div class="bar-labels"><span>ru ${ruPct}%</span><span>it ${itPct}%</span></div>
      <div class="result-time">${r.kind === "probability" ? "вероятность класса" : "мера расстояния (инвертирована для наглядности)"} · ${r.elapsed_ms.toFixed(2)} мс</div>
      ${agreementLine}`;
    return card;
  }

  function renderExplanation(explanation) {
    if (!explanation) return "";
    const list = (label, items) => {
      if (!items || !items.length) return `<div class="explain-row"><span class="explain-label">${label}:</span> <span class="hint">совпадений не найдено</span></div>`;
      const chips = items.map((g) => `<code class="explain-chip">${escapeHtml(g)}</code>`).join(" ");
      return `<div class="explain-row"><span class="explain-label">${label}:</span> ${chips}</div>`;
    };
    return `
      <div class="explain-box">
        <div class="explain-title">Почему «${LANG_NAMES[explanation.best]}»?</div>
        ${list("N-граммы текста из топ-профиля языка", explanation.ngram_hits)}
        ${list("Буквы, характерные для этого языка", explanation.alphabet_hits)}
        ${list("Биграммы, важные для нейросети", explanation.neural_hits)}
      </div>`;
  }

  function renderFeedback(text) {
    const box = document.createElement("div");
    box.className = "feedback-box";
    box.innerHTML = `
      <span class="feedback-label">Вердикт неверен? Подскажите правильный язык — модель дообучится:</span>
      <button type="button" class="chip feedback-btn" data-lang="ru">это русский</button>
      <button type="button" class="chip feedback-btn" data-lang="it">это итальянский</button>
      <span class="feedback-status"></span>`;
    box.querySelectorAll(".feedback-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const status = box.querySelector(".feedback-status");
        status.textContent = "Отправляю…";
        try {
          const res = await postJSON("/api/feedback", { text, lang: btn.dataset.lang });
          status.textContent = res.applied
            ? "Спасибо — сеть дообучена на этом примере."
            : "Модель уже была уверена и права — дообучение не потребовалось.";
        } catch (e) {
          status.textContent = "Ошибка: " + e.message;
        }
      });
    });
    return box;
  }

  function renderResults(container, outcome, text) {
    container.innerHTML = "";
    if (outcome.ensemble) container.appendChild(renderResultCard("ensemble", outcome.ensemble));
    ["ngram", "alphabet", "neural"].forEach((key) => container.appendChild(renderResultCard(key, outcome[key])));
    if (outcome.explanation) {
      const wrap = document.createElement("div");
      wrap.innerHTML = renderExplanation(outcome.explanation);
      container.appendChild(wrap.firstElementChild);
    }
    if (typeof text === "string" && text.trim()) container.appendChild(renderFeedback(text));
  }

  const textarea = document.getElementById("input-text");
  if (textarea) {
    const resultsEl = document.getElementById("results");
    const liveDot = document.getElementById("live-dot");
    let debounceTimer = null;

    async function classifyLive(text) {
      if (!text.trim()) {
        resultsEl.innerHTML = '<p class="empty-hint">Начните печатать — результаты появятся автоматически.</p>';
        return null;
      }
      liveDot.classList.add("active");
      try {
        const outcome = await postJSON("/api/classify", { text });
        if (outcome.empty) return null;
        renderResults(resultsEl, outcome, text);
        return outcome;
      } catch (e) {
        resultsEl.innerHTML = `<p class="empty-hint">Ошибка: ${escapeHtml(e.message)}</p>`;
        return null;
      } finally {
        setTimeout(() => liveDot.classList.remove("active"), 250);
      }
    }

    textarea.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => classifyLive(textarea.value), 300);
    });

    document.querySelectorAll("[data-example]").forEach((btn) => {
      const EXAMPLES = {
        ru: "Городская библиотека объявила о начале летней программы чтения для детей и подростков. Каждую субботу здесь будут проходить встречи с писателями и мастер-классы по иллюстрации книг.",
        it: "La biblioteca comunale ha annunciato l'inizio del programma estivo di lettura per bambini e ragazzi. Ogni sabato si terranno incontri con scrittori e laboratori di illustrazione di libri.",
      };
      btn.addEventListener("click", () => {
        textarea.value = EXAMPLES[btn.dataset.example];
        classifyLive(textarea.value);
      });
    });

    document.getElementById("btn-clear").addEventListener("click", () => {
      textarea.value = "";
      resultsEl.innerHTML = '<p class="empty-hint">Начните печатать — результаты появятся автоматически.</p>';
    });

    const batchList = document.getElementById("batch-list");
    const batchEmpty = document.getElementById("batch-empty");
    let batchItems = [];
    let batchCounter = 0;

    function renderBatch() {
      batchList.innerHTML = "";
      if (!batchItems.length) {
        batchList.appendChild(batchEmpty);
        return;
      }
      batchItems.forEach((item) => {
        const row = document.createElement("div");
        row.className = "batch-item";
        const chip = (r) => (r ? `<span class="tag ${r.best}">${LANG_NAMES[r.best]}</span>` : "");
        row.innerHTML = `
          <span class="fname">${escapeHtml(item.filename)}</span>
          ${item.results ? ["ensemble", "ngram", "alphabet", "neural"].map((k) => chip(item.results[k])).join(" ") : "<span class=\"hint\">без предпросмотра</span>"}
          <select data-id="${item.id}">
            <option value="">язык неизвестен</option>
            <option value="ru" ${item.expected === "ru" ? "selected" : ""}>ожид.: русский</option>
            <option value="it" ${item.expected === "it" ? "selected" : ""}>ожид.: итальянский</option>
          </select>
          <button type="button" class="remove-btn" data-id="${item.id}" title="удалить из пакета">✕</button>`;
        row.querySelector("select").addEventListener("change", (e) => {
          const it = batchItems.find((b) => b.id === item.id);
          if (it) it.expected = e.target.value;
        });
        row.querySelector(".remove-btn").addEventListener("click", () => {
          batchItems = batchItems.filter((b) => b.id !== item.id);
          renderBatch();
        });
        batchList.appendChild(row);
      });
    }

    async function addToBatch(filename, text) {
      const id = ++batchCounter;
      const entry = { id, filename, text, expected: "", results: null };
      batchItems.push(entry);
      renderBatch();
      try {
        const outcome = await postJSON("/api/classify", { text });
        if (!outcome.empty) entry.results = outcome;
      } catch (e) {  }
      renderBatch();
    }

    document.getElementById("btn-add-batch").addEventListener("click", () => {
      const text = textarea.value;
      if (!text.trim()) return;
      batchCounter += 1;
      const name = `текст_${batchCounter}.txt`;
      addToBatch(name, text);
    });

    document.getElementById("btn-batch-clear").addEventListener("click", () => {
      batchItems = [];
      renderBatch();
    });

    document.getElementById("btn-generate-report").addEventListener("click", async () => {
      const box = document.getElementById("report-link-box");
      if (!batchItems.length) {
        box.innerHTML = '<p class="empty-hint">Пакет пуст — нечего включать в отчёт.</p>';
        return;
      }
      box.innerHTML = '<p class="empty-hint">Формирую отчёт…</p>';
      try {
        const payload = { items: batchItems.map((b) => ({ filename: b.filename, text: b.text, expected: b.expected })) };
        const res = await postJSON("/api/report", payload);
        box.innerHTML = `<a href="${res.url}" target="_blank">Открыть отчёт →</a> <span class="hint">сохранён также на странице «Отчёты»</span>`;
      } catch (e) {
        box.innerHTML = `<p class="empty-hint">Ошибка: ${escapeHtml(e.message)}</p>`;
      }
    });

    renderBatch();

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    document.getElementById("browse-btn").addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("click", (e) => { if (e.target.id !== "browse-btn") fileInput.click(); });

    ["dragenter", "dragover"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add("drag-over"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove("drag-over"); })
    );
    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) handleFiles(files);
    });
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length) handleFiles(e.target.files);
      fileInput.value = "";
    });

    function handleFiles(fileList) {
      [...fileList].forEach((file) => {
        if (file.size > 2 * 1024 * 1024) {
          addToBatch(file.name + " (слишком большой, >2 Мб — не загружен)", "");
          return;
        }
        const reader = new FileReader();
        reader.onload = (ev) => addToBatch(file.name, ev.target.result);
        reader.onerror = () => addToBatch(file.name + " (ошибка чтения)", "");
        reader.readAsText(file, "UTF-8");
      });
    }
  }

  const retrainBtn = document.getElementById("btn-retrain");
  if (retrainBtn) {
    retrainBtn.addEventListener("click", async () => {
      const status = document.getElementById("retrain-status");
      retrainBtn.disabled = true;
      status.textContent = "Переобучаю модель на data/corpus…";
      try {
        const res = await postJSON("/api/retrain", {});
        status.textContent = `Готово (${res.trained_at}). Обновляю страницу…`;
        setTimeout(() => window.location.reload(), 700);
      } catch (e) {
        status.textContent = "Ошибка: " + e.message;
        retrainBtn.disabled = false;
      }
    });
  }
})();
