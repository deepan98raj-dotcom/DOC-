from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class RetrievalService:
    """Retrieve relevant document chunks using semantic search."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedding_service.embed_text(
            query
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        retrieved_chunks = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for i in range(len(documents)):
            retrieved_chunks.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )

        return retrieved_chunks
    