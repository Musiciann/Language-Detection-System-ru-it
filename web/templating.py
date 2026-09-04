import re
from config import TEMPLATES_DIR
import os

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

_cache = {}


def _read_template(name: str) -> str:
    if name not in _cache:
        path = os.path.join(TEMPLATES_DIR, name)
        with open(path, encoding="utf-8") as f:
            _cache[name] = f.read()
    return _cache[name]


def render(name: str, **ctx) -> str:
    text = _read_template(name)
    return _PLACEHOLDER_RE.sub(lambda m: str(ctx.get(m.group(1), m.group(0))), text)


def render_page(content: str, title: str, active: str = "") -> str:
    return render("layout.html", title=title, content=content, active=active)


def escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))
