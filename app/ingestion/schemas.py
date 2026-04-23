from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class PageContent (BaseModel):
    page_number: int 
    text: str 

class Document(BaseModel):
    document_id : str 
    document_name : str 
    file_path : str
    file_type:str 
    pages: List[PageContent] = Field(default_factory=list)

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    source: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    char_count: int



