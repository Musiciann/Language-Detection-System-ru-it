import os
import random


def load_corpus(corpus_dir: str) -> dict:
    corpus = {}
    for lang in sorted(os.listdir(corpus_dir)):
        lang_path = os.path.join(corpus_dir, lang)
        if not os.path.isdir(lang_path):
            continue
        texts = []
        for fname in sorted(os.listdir(lang_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(lang_path, fname), encoding="utf-8") as f:
                    texts.append(f.read())
        corpus[lang] = texts
    return corpus


def load_test_documents(test_dir: str) -> list:
    docs = []
    for fname in sorted(os.listdir(test_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(test_dir, fname), encoding="utf-8") as f:
                docs.append({"file": fname, "text": f.read()})
    return docs


def split_train_val(corpus_by_lang: dict, val_ratio: float = 0.25, seed: int = 42) -> tuple:
    rnd = random.Random(seed)
    train, val = {}, {}
    for lang, docs in corpus_by_lang.items():
        docs = list(docs)
        rnd.shuffle(docs)
        n_val = max(1, round(len(docs) * val_ratio)) if len(docs) > 1 else 0
        val[lang] = docs[:n_val]
        train[lang] = docs[n_val:] or docs  # язык не должен остаться совсем без обучающих примеров
    return train, val


def chunk_documents(docs: list, chunk_chars: int = 380, min_chars: int = 120) -> list:
    chunks = []
    for doc in docs:
        text = (doc or "").strip()
        if not text:
            continue
        if len(text) <= chunk_chars:
            chunks.append(text)
            continue
        for i in range(0, len(text), chunk_chars):
            piece = text[i:i + chunk_chars]
            if len(piece) >= min_chars:
                chunks.append(piece)
    return chunks
