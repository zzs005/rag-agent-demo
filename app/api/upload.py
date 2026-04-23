from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.api.schemas import UploadResponse
from app.ingestion.pipeline import ingest_document
from app.services.index_service import rebuild_index


router = APIRouter(prefix="/upload", tags=["upload"])

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for now.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DIR / filename

    content = await file.read()
    file_path.write_bytes(content)

    try:
        output_path = ingest_document(
            file_path=str(file_path),
            output_dir=str(PROCESSED_DIR),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    try:
        rebuild_index()
        reindexed = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index rebuild failed after upload: {e}")

    return UploadResponse(
        filename=filename,
        saved_path=str(file_path),
        chunks_output_path=str(output_path),
        reindexed=reindexed,
        status="success",
    )