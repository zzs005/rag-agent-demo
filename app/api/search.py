from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_search_service
from app.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.retrieval.search_service import SearchService


router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, search_service: SearchService = Depends(get_search_service)) -> SearchResponse:
    results = search_service.search(query=request.query, top_k=request.top_k)

    items = []
    for item in results:
        items.append(
            SearchResultItem(
                chunk_id=item.chunk_id,
                source=item.source,
                page_start=item.page_start,
                page_end=item.page_end,
                section_title=item.section_title,
                score=item.score,
                text_preview=item.text[:300].replace("\n", " "),
            )
        )

    return SearchResponse(
        query=request.query,
        results=items,
    )