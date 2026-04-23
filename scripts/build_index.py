from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.services.index_service import rebuild_index


def main() -> None:
    result = rebuild_index()
    print("Index rebuild finished.")
    print(f"chunks_count : {result['chunks_count']}")
    print(f"vectors_path : {result['vectors_path']}")
    print(f"metadata_path: {result['metadata_path']}")


if __name__ == "__main__":
    main()