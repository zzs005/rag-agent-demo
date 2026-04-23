from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import ReindexResponse
from app.services.index_service import rebuild_index


router = APIRouter(prefix="/reindex", tags=["reindex"])


@router.post("", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    try:
        result = rebuild_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

    return ReindexResponse(
        status=result["status"],
        chunks_count=result["chunks_count"],
        vectors_path=result["vectors_path"],
        metadata_path=result["metadata_path"],
    )