from __future__ import annotations

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.search import router as search_router
from app.api.upload import router as upload_router
from app.api.reindex import router as reindex_router


app = FastAPI(
    title="KB Agent API",
    version="0.2.0",
    description="Knowledge Base Agent backend with upload, search, chat, and reindex endpoints.",
)


@app.get("/")
def root():
    return {"message": "KB Agent API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(upload_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(reindex_router)