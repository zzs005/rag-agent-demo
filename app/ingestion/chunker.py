from __future__ import annotations

import re
from typing import List, Optional

from .schemas import Chunk, Document


DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 250
SENTENCE_ENDINGS = "\u3002\uff01\uff1f\uff1b.!?;"
SECONDARY_BREAKS = "\uff0c\u3001,\uff1a:)]}\u3011> \t"
CLOSING_QUOTES = "\"'\u201d\u2019\uff09)]}\u3011>"


def guess_section_title(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines[:8]:
        if 4 <= len(line) <= 80 and re.match(r"^\d+[.\u3001]?\s*\S+", line):
            return line

    for line in lines[:8]:
        if not (2 <= len(line) <= 60):
            continue
        if re.fullmatch(r"\d+[.\u3001]?", line):
            continue
        return line

    return None


def split_paragraphs(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized)]
    return [p for p in paragraphs if p]


def split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"[ \t]+", " ", text.strip())
    if not normalized:
        return []

    sentences: List[str] = []
    buffer: List[str] = []
    text_len = len(normalized)

    for index, char in enumerate(normalized):
        buffer.append(char)

        if char not in SENTENCE_ENDINGS:
            continue

        next_char = normalized[index + 1] if index + 1 < text_len else ""
        if next_char and next_char in CLOSING_QUOTES:
            continue

        sentence = "".join(buffer).strip()
        if sentence:
            sentences.append(sentence)
        buffer = []

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)

    return _merge_title_like_sentences(sentences)


def split_long_sentence(text: str, chunk_size: int) -> List[str]:
    stripped = text.strip()
    if len(stripped) <= chunk_size:
        return [stripped]

    pieces: List[str] = []
    start = 0

    while start < len(stripped):
        remaining = len(stripped) - start
        if remaining <= chunk_size:
            pieces.append(stripped[start:].strip())
            break

        end = start + chunk_size
        window = stripped[start:end]
        cut = _find_last_breakpoint(window)
        if cut <= 0:
            cut = chunk_size

        piece = stripped[start : start + cut].strip()
        if piece:
            pieces.append(piece)
        start += cut

    return pieces


def build_sentence_aware_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    sentences = split_sentences(text)
    if not sentences:
        return split_long_sentence(text, chunk_size=chunk_size)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append("\n".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(split_long_sentence(sentence, chunk_size=chunk_size))
            continue

        projected_len = current_len + len(sentence) + (1 if current else 0)
        if projected_len <= chunk_size:
            current.append(sentence)
            current_len = projected_len
            continue

        if current:
            chunks.append("\n".join(current).strip())
            current = _build_overlap_sentences(current, overlap)
            current_len = _joined_length(current)

        current.append(sentence)
        current_len = _joined_length(current)

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    results: List[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        paragraphs = split_paragraphs(page_text)

        if page.page_number <= 3:
            print(
                f"[CHUNK] page={page.page_number} page_chars={len(page_text)} paragraphs={len(paragraphs)}"
            )

        buffer_parts: List[str] = []
        buffer_len = 0

        def push_chunk(text: str) -> None:
            nonlocal chunk_index
            content = text.strip()
            if not content:
                return

            results.append(
                Chunk(
                    chunk_id=f"{document.document_id}_chunk_{chunk_index:04d}",
                    document_id=document.document_id,
                    document_name=document.document_name,
                    source=document.document_name,
                    chunk_index=chunk_index,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section_title=guess_section_title(content),
                    text=content,
                    char_count=len(content),
                )
            )
            chunk_index += 1

        def flush_buffer() -> None:
            nonlocal buffer_parts, buffer_len
            if not buffer_parts:
                return
            push_chunk("\n\n".join(buffer_parts))
            buffer_parts = []
            buffer_len = 0

        for para in paragraphs:
            if len(para) > chunk_size:
                flush_buffer()
                for piece in build_sentence_aware_chunks(para, chunk_size=chunk_size, overlap=overlap):
                    push_chunk(piece)
                continue

            projected_len = buffer_len + len(para) + (2 if buffer_parts else 0)
            if projected_len <= chunk_size:
                buffer_parts.append(para)
                buffer_len = projected_len
            else:
                flush_buffer()
                buffer_parts.append(para)
                buffer_len = len(para)

        flush_buffer()

    return results


def _find_last_breakpoint(text: str) -> int:
    for index in range(len(text) - 1, 0, -1):
        if text[index] in SECONDARY_BREAKS or text[index] in SENTENCE_ENDINGS:
            return index + 1
    return 0


def _build_overlap_sentences(sentences: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or not sentences:
        return []

    selected: List[str] = []
    current_len = 0
    for sentence in reversed(sentences):
        projected_len = current_len + len(sentence) + (1 if selected else 0)
        if selected and projected_len > overlap:
            break
        selected.insert(0, sentence)
        current_len = projected_len

    return selected


def _joined_length(sentences: List[str]) -> int:
    if not sentences:
        return 0
    return sum(len(sentence) for sentence in sentences) + max(0, len(sentences) - 1)


def _merge_title_like_sentences(sentences: List[str]) -> List[str]:
    if len(sentences) < 2:
        return sentences

    merged: List[str] = []
    index = 0
    while index < len(sentences):
        current = sentences[index]
        if (
            index + 1 < len(sentences)
            and len(current) <= 30
            and re.match(r"^(\d+[.\u3001]?\s*)?\S+$", current.replace("\n", ""))
            and current[-1] not in SENTENCE_ENDINGS
        ):
            merged.append(f"{current}\n{sentences[index + 1]}")
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged
