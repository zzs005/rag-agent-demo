from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    chunk_id: str
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    score: float
    text_preview: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: Optional[str] = None


class CitationResponse(BaseModel):
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_id: str
    section_title: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    answer_type: str
    confidence: str
    citations: List[CitationResponse]
    used_chunks: List[str]
    refused: bool
    refusal_reason: Optional[str] = None
    conversation_id: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    agent_trace: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    chunks_output_path: str
    reindexed: bool
    status: str


class ReindexResponse(BaseModel):
    status: str
    chunks_count: int
    vectors_path: str
    metadata_path: str
