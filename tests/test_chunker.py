from __future__ import annotations

import unittest

from app.ingestion.chunker import build_sentence_aware_chunks, split_long_sentence, split_sentences


class ChunkerTests(unittest.TestCase):
    def test_split_sentences_keeps_sentence_boundaries(self) -> None:
        text = "第一句说明问题。第二句继续补充！第三句给出结论？"
        self.assertEqual(
            split_sentences(text),
            ["第一句说明问题。", "第二句继续补充！", "第三句给出结论？"],
        )

    def test_sentence_aware_chunks_prefer_complete_sentences(self) -> None:
        text = "第一句比较短。第二句也比较短。第三句同样不长。"
        chunks = build_sentence_aware_chunks(text, chunk_size=16, overlap=6)
        self.assertTrue(all(chunk.endswith(("。", "！", "？")) for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 2)

    def test_long_sentence_falls_back_to_secondary_breaks(self) -> None:
        text = "这是一个很长很长的句子，里面包含多个逗号、顿号、以及空格用于兜底切分"
        chunks = split_long_sentence(text, chunk_size=12)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 2)


if __name__ == "__main__":
    unittest.main()
