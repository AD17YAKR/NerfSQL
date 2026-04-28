import json
from pathlib import Path

import faiss
import numpy as np
from pinecone import Pinecone
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.core.config import settings

_DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "schema_chunks.json"

_RERANKER_MODEL_ALIASES = {
    "cross-encoder/ms-marco-MiniLM-L6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-TinyBERT-L2-v2": "Xenova/ms-marco-TinyBERT-L-2-v2",
}

class SchemaRetriever:
    def __init__(
        self,
        chunks_path: Path | str = _DEFAULT_CHUNKS_PATH,
        top_k: int = 5,
        fetch_k: int | None = None,
        rerank_enabled: bool | None = None,
        reranker_model: str | None = None,
    ):
        self.top_k = top_k
        self.fetch_k = max(fetch_k or (top_k * 4), top_k)
        self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.pinecone_api_key = settings.pinecone_api_key
        self.pinecone_index_name = settings.pinecone_index_name
        self.pinecone_namespace = settings.pinecone_namespace
        self.pinecone_index = None

        self.rerank_enabled = settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        self.reranker_model_name = settings.reranker_model if reranker_model is None else reranker_model
        self.reranker = None
        if self.rerank_enabled:
            try:
                mapped_model = _RERANKER_MODEL_ALIASES.get(self.reranker_model_name, self.reranker_model_name)
                self.reranker = TextCrossEncoder(model_name=mapped_model)
            except Exception:
                self.reranker = None

        if self.pinecone_api_key:
            try:
                pc = Pinecone(api_key=self.pinecone_api_key)
                self.pinecone_index = pc.Index(self.pinecone_index_name)
            except Exception:
                self.pinecone_index = None

        with open(chunks_path) as f:
            self.chunks: list[str] = json.load(f)
        if self.chunks:
            embeddings = np.array(list(self.model.embed(self.chunks)))
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype(np.float32))
        else:
            self.index = None

    def _rerank(self, query: str, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        if not self.rerank_enabled or self.reranker is None:
            return candidates[: self.top_k]

        scores = np.array(list(self.reranker.rerank(query, candidates)))
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order[: self.top_k]]

    def _retrieve_candidates(self, query: str) -> list[str]:
        vec = np.array(list(self.model.embed([query])), dtype=np.float32)[0]

        if self.pinecone_index is not None:
            try:
                response = self.pinecone_index.query(
                    vector=vec.tolist(),
                    top_k=self.fetch_k,
                    namespace=self.pinecone_namespace,
                    include_metadata=True,
                )
                matches = response.get("matches") if isinstance(response, dict) else response.matches
                pinecone_chunks = []
                for match in matches:
                    metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
                    chunk = metadata.get("chunk")
                    if chunk:
                        pinecone_chunks.append(chunk)
                if pinecone_chunks:
                    return pinecone_chunks
            except Exception:
                pass

        if not self.index:
            return []
        _, ids = self.index.search(vec.reshape(1, -1), self.fetch_k)
        return [self.chunks[i] for i in ids[0] if i < len(self.chunks)]

    def retrieve(self, query: str) -> str:
        candidates = self._retrieve_candidates(query)
        reranked = self._rerank(query, candidates)
        return "\n".join(reranked)
