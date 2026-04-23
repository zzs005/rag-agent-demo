from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import uuid4

from pypdf import PdfReader

from .schemas import Document, PageContent

def make_document_id(file_path: Path) -> str:
    #文件名 + 随机后缀也行
    return f"{file_path.stem}_{uuid4().hex[:8]}"

def load_pdf(file_path: str) -> Document:
    path = Path(file_path)
    reader = PdfReader(str(path))

    pages: List[PageContent] = []

    print(f"[PDF] loading: {path.name}, total_pages={len(reader.pages)}")

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[PDF] page {i} extract failed: {e}")
            text = ""

        if i <= 3:
            preview = text[:120].replace("\n", " ")
            print(f"[PDF] page {i} chars={len(text)} preview={preview!r}")

        pages.append(
            PageContent(
                page_number=i,
                text=text,
            )
        )

    return Document(
        document_id=make_document_id(path),
        document_name=path.name,
        file_path=str(path),
        file_type="pdf",
        pages=pages,
    )

def load_document(file_path: str) -> Document:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")