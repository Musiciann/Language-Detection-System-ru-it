import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

LANG_NAMES = {"ru": "Русский", "it": "Итальянский"}

HOST = "127.0.0.1"
PORT = 8765

MAX_UPLOAD_TEXT_CHARS = 500_000
