import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGQueryEngine:
    def __init__(self, data_dir="crawl_data"):
        self.data_dir = data_dir
        self.index_file = os.path.join(data_dir, "faiss_index.bin")
        self.metadata_file = os.path.join(data_dir, "metadata.json")

        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Vector index not found at {self.index_file}")

        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_file}")

        # Load FAISS index
        self.index = faiss.read_index(self.index_file)

        # Load metadata
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embedding model (same one used during crawl)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = 5):
        # Embed query
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")

        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append({
                "id": entry["id"],
                "url": entry["url"],
                "title": entry.get("title", ""),
                "content": entry["content"],
                "distance": float(dist)
            })

        return results
