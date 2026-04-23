from __future__ import annotations

import os

from dotenv import load_dotenv

from app.retrieval.embedder import LocalEmbedder
from app.retrieval.vector_store import NumpyVectorStore
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.hybrid_search import merge_hybrid_results
from app.retrieval.reranker import LocalReranker
from app.retrieval.schemas import RetrievedChunk


load_dotenv()


class SearchService:
    def __init__(self, vectors_path: str, metadata_path: str):
        self.embedder = LocalEmbedder()

        self.vector_store = NumpyVectorStore()
        self.vector_store.load(vectors_path=vectors_path, metadata_path=metadata_path)

        self.keyword_store = KeywordStore()
        self.keyword_store.load(metadata_path=metadata_path)

        self.reranker = LocalReranker()

        self.retrieve_candidates = int(os.getenv("RETRIEVE_CANDIDATES", "12"))

    def search_vector(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k)

    def search_keyword(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.keyword_store.search(query=query, top_k=top_k)

    def search_hybrid_candidates(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        candidate_k = top_k or self.retrieve_candidates

        vector_results = self.search_vector(query=query, top_k=candidate_k)
        keyword_results = self.search_keyword(query=query, top_k=candidate_k)

        hybrid_results = merge_hybrid_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=candidate_k,
        )
        return hybrid_results

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidates = self.search_hybrid_candidates(query=query, top_k=self.retrieve_candidates)
        reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return reranked