import re

_WORD_RE = re.compile(r"[a-zà-öø-ÿа-яё]+")
_LETTER_RE = re.compile(r"[a-zà-öø-ÿа-яё]")


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> list:
    return _WORD_RE.findall(text)


def extract_letters(text: str) -> list:
    return _LETTER_RE.findall(text)
