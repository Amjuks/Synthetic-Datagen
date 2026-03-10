from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)
