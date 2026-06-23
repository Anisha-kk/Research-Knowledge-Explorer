import faiss
import numpy as np
import json
import os


class VectorStore:
    def __init__(self, index_path, doc_path, dim=384):
        self.index_path = index_path
        self.doc_path = doc_path
        self.dim = dim

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Use cosine similarity via Inner Product
        self.index = (
            faiss.read_index(index_path)
            if os.path.exists(index_path)
            else faiss.IndexFlatIP(dim)
        )

        self.documents = (
            json.load(open(doc_path, "r", encoding="utf-8"))
            if os.path.exists(doc_path)
            else []
        )

    def clear(self):
        """
        Reset upload index completely.
        Useful when a new paper is uploaded.
        """

        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []

        faiss.write_index(self.index, self.index_path)

        with open(self.doc_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    #creates a semantic vector space of documents
    def add(self, vectors, docs):
        vectors = np.array(vectors).astype("float32")

        # normalize for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        
        self.documents.extend(docs)

        faiss.write_index(self.index, self.index_path)

        with open(self.doc_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

    #Similarity based search 
    def search(self, query_vector, k=5):
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, k)#nearest-neighbor search in embedding space
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(self.documents):
                continue
            results.append({
                "doc_id": self.documents[idx]["pdf_path"],
                "doc": self.documents[idx],
                "score": float(score)
            })
        return results
    
    def get_docs_for_file(self, file_path):
        return [d
            for d in self.documents
            if d.get("pdf_path") == file_path
        ]