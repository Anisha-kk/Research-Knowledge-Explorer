from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder="./hf_cache"
        )

    def embed(self, texts):
        """
        Fast batch embedding for RAG ingestion
        """

        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=128,               # increased batch size
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,    # directly compatible with FAISS IP
            device="cpu"                 
        )

        return embeddings