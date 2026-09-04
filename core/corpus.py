import os


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
