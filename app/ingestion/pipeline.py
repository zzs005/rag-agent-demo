from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .cleaner import clean_text
from .chunker import chunk_document
from .loaders import load_document
from .schemas import Document, PageContent, Chunk


def clean_document_pages(document: Document) -> Document:
    cleaned_pages: List[PageContent] = []

    for page in document.pages:
        raw_len = len(page.text)
        cleaned = clean_text(page.text)
        cleaned_len = len(cleaned)

        if page.page_number <= 3:
            print(
                f"[CLEAN] page={page.page_number} raw_len={raw_len} cleaned_len={cleaned_len}"
            )

        cleaned_pages.append(
            PageContent(
                page_number=page.page_number,
                text=cleaned,
            )
        )

    return Document(
        document_id=document.document_id,
        document_name=document.document_name,
        file_path=document.file_path,
        file_type=document.file_type,
        pages=cleaned_pages,
    )


def save_chunks_jsonl(chunks: List[Chunk], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SAVE] chunks_count={len(chunks)} output={path}")

    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


def ingest_document(file_path: str, output_dir: str) -> str:
    document = load_document(file_path)
    print(f"[PIPELINE] loaded pages={len(document.pages)}")

    cleaned_document = clean_document_pages(document)
    chunks = chunk_document(cleaned_document)

    print(f"[PIPELINE] generated chunks={len(chunks)}")

    output_file = Path(output_dir) / f"{Path(file_path).stem}_chunks.jsonl"
    save_chunks_jsonl(chunks, str(output_file))
    return str(output_file)