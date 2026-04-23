from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能找到 app 包
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.ingestion.pipeline import ingest_document


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {RAW_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        print(f"[START] {pdf_path.name}")
        try:
            output_path = ingest_document(
                file_path=str(pdf_path),
                output_dir=str(PROCESSED_DIR),
            )
            print(f"[DONE ] {pdf_path.name} -> {output_path}\n")
        except Exception as e:
            print(f"[ERROR] {pdf_path.name}: {e}\n")


if __name__ == "__main__":
    main()