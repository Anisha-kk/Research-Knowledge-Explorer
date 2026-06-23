class Retriever:
    def __init__(self,global_store,upload_store,embedder):
        self.global_store = global_store
        self.upload_store = upload_store
        self.embedder = embedder    

    def retrieve(self, query, mode="global", k=5):
        qvec = self.embedder.embed([query])[0].astype("float32")
        # -------------------------
        # GLOBAL MODE
        # -------------------------
        if mode == "global":
            dense_results = self.global_store.search(qvec.reshape(1, -1),k)
            return dense_results or []
        # -------------------------
        # UPLOAD MODE
        # -------------------------
        if mode == "upload":
            dense_results = self.upload_store.search(qvec.reshape(1, -1),k)
            return dense_results or []
        # -------------------------
        # HYBRID MODE
        # -------------------------
        g_results = self.global_store.search(qvec.reshape(1, -1), k)
        u_results = self.upload_store.search(qvec.reshape(1, -1), k)

        g_results = g_results or []
        u_results = u_results or []

        combined = g_results + u_results

        # optional: sort by relevance score (recommended)
        combined = sorted(combined, key=lambda x: x["score"], reverse=True)

        return combined