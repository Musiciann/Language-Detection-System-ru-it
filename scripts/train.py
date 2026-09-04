import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CORPUS_DIR, TEST_DIR, MODEL_PATH, MODELS_DIR, LANG_NAMES
from core.model import LanguageModel


def main():
    print("Обучение модели на корпусе:", CORPUS_DIR)
    model = LanguageModel.train(CORPUS_DIR, TEST_DIR)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    print(f"\nКорпус: " + ", ".join(
        f"{LANG_NAMES[l]} — {d} докум., {c} симв."
        for l, (d, c) in zip(model.meta["corpus_docs"],
                              zip(model.meta["corpus_docs"].values(), model.meta["corpus_chars"].values()))
    ))
    print(f"Признаков нейросети (биграмм): {model.meta['bigram_vocab_size']}")
    print(f"\nТочность на тестовой коллекции ({len(model.test_results)} документов):")
    for method, acc in model.accuracy.items():
        print(f"  {method:<10}: {acc*100:5.1f}%   ({model.times_ms[method]:.2f} мс суммарно)")

    print(f"\nМодель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
