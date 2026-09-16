import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.corpus import split_train_val, chunk_documents


class TestCorpusHelpers(unittest.TestCase):
    def test_split_train_val_keeps_every_document_exactly_once(self):
        corpus = {"ru": [f"doc{i}" for i in range(8)]}
        train, val = split_train_val(corpus, val_ratio=0.25, seed=1)
        self.assertEqual(sorted(train["ru"] + val["ru"]), sorted(corpus["ru"]))
        self.assertGreaterEqual(len(val["ru"]), 1)
        self.assertGreaterEqual(len(train["ru"]), 1)

    def test_split_train_val_is_deterministic_with_seed(self):
        corpus = {"it": [f"doc{i}" for i in range(10)]}
        t1, v1 = split_train_val(corpus, val_ratio=0.3, seed=7)
        t2, v2 = split_train_val(corpus, val_ratio=0.3, seed=7)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)

    def test_chunk_documents_respects_min_length(self):
        chunks = chunk_documents(["a" * 1000], chunk_chars=300, min_chars=100)
        self.assertTrue(all(len(c) >= 100 for c in chunks))
        self.assertGreaterEqual(len(chunks), 3)

    def test_chunk_documents_keeps_short_document_whole(self):
        chunks = chunk_documents(["short text"], chunk_chars=300, min_chars=100)
        self.assertEqual(chunks, ["short text"])

    def test_chunk_documents_skips_empty_strings(self):
        chunks = chunk_documents(["", "  ", "hello"], chunk_chars=300, min_chars=1)
        self.assertEqual(chunks, ["hello"])


if __name__ == "__main__":
    unittest.main()
