from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def load_chunks_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    return chunks


def load_all_chunks(processed_dir: str) -> List[Dict[str, Any]]:
    path = Path(processed_dir)
    all_chunks: List[Dict[str, Any]] = []

    for jsonl_file in sorted(path.glob("*_chunks.jsonl")):
        chunks = load_chunks_from_jsonl(str(jsonl_file))
        all_chunks.extend(chunks)

    return all_chunks