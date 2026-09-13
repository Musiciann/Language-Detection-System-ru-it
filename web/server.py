import os
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import STATIC_DIR, REPORTS_DIR, MODEL_PATH, CORPUS_DIR, TEST_DIR
from core.model import LanguageModel
from web import pages, api

class _ModelHolder:
    """Обёртка, чтобы /api/retrain мог заменить модель, а все обработчики
    всегда видели актуальную версию (через holder.model, а не захваченную
    по значению переменную)."""
    def __init__(self, model):
        self.model = model


def load_or_train_model() -> LanguageModel:
    if os.path.isfile(MODEL_PATH):
        try:
            return LanguageModel.load(MODEL_PATH)
        except Exception as e:  # noqa: BLE001 - модель повреждена, переобучаем
            print(f"[!] Не удалось загрузить {MODEL_PATH} ({e}), переобучаю…")
    print("Модель не найдена — обучаю на data/corpus (один раз, при первом запуске)…")
    model = LanguageModel.train(CORPUS_DIR, TEST_DIR)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    return model


HOLDER = _ModelHolder(load_or_train_model())


def _is_within(base_dir: str, target_path: str) -> bool:
    base = os.path.realpath(base_dir)
    target = os.path.realpath(target_path)
    return target == base or target.startswith(base + os.sep)


class Handler(BaseHTTPRequestHandler):
    server_version = "LangDetectApp/1.0"

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str):
        if not os.path.isfile(path):
            self._send_html("<h1>404</h1><p>Файл не найден.</p>", 404)
            return
        ctype, _ = mimetypes.guess_type(path)
        ctype = ctype or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} · {fmt % args}")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                self._send_html(pages.index_page(HOLDER.model))
            elif path == "/model":
                self._send_html(pages.model_page(HOLDER.model))
            elif path == "/task":
                self._send_html(pages.task_page())
            elif path == "/reports":
                self._send_html(pages.reports_list_page())
            elif path.startswith("/reports/"):
                rel = path[len("/reports/"):]
                target = os.path.join(REPORTS_DIR, rel)
                if _is_within(REPORTS_DIR, target):
                    self._send_file(target)
                else:
                    self._send_html("<h1>403</h1>", 403)
            elif path.startswith("/static/"):
                rel = path[len("/static/"):]
                target = os.path.join(STATIC_DIR, rel)
                if _is_within(STATIC_DIR, target):
                    self._send_file(target)
                else:
                    self._send_html("<h1>403</h1>", 403)
            else:
                self._send_html("<h1>404</h1><p>Страница не найдена.</p>", 404)
        except Exception as e:  # noqa: BLE001
            self._send_html(f"<h1>500</h1><pre>{e}</pre>", 500)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/classify":
                body = self._read_json_body()
                self._send_json(api.classify(HOLDER.model, body))
            elif path == "/api/upload":
                body = self._read_json_body()
                self._send_json(api.upload(HOLDER.model, body))
            elif path == "/api/report":
                body = self._read_json_body()
                self._send_json(api.generate_report(HOLDER.model, body))
            elif path == "/api/retrain":
                HOLDER.model = api.retrain()
                self._send_json({"ok": True, "trained_at": HOLDER.model.meta["trained_at"]})
            elif path == "/api/feedback":
                body = self._read_json_body()
                self._send_json(api.feedback(HOLDER.model, body))
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"internal error: {e}"}, 500)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return ThreadingHTTPServer((host, port), Handler)
