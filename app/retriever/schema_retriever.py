import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SchemaRetriever:
    def __init__(self, chunks_path: str = "data/schema_chunks.json", top_k: int = 5):
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        with open(chunks_path) as f:
            self.chunks: list[str] = json.load(f)
        if self.chunks:
            embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype(np.float32))
        else:
            self.index = None

    def retrieve(self, query: str) -> str:
        if not self.index:
            return ""
        vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        _, ids = self.index.search(vec, self.top_k)
        return "\n".join(self.chunks[i] for i in ids[0] if i < len(self.chunks))
