from app.services.bm25_service import BM25Service
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class HybridRetrievalService:
    """Combine dense retrieval and BM25 using RRF."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.bm25_service = BM25Service()

        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 from ChromaDB documents."""

        data = self.vector_store.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        chunk_ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not documents:
            raise RuntimeError(
                "No documents found in ChromaDB."
            )

        self.bm25_service.build_index(
            chunk_ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
        )

    def _refresh_if_needed(self) -> None:
        """Refresh BM25 when ChromaDB has changed."""

        chroma_count = self.vector_store.count()
        bm25_count = len(
            self.bm25_service.documents
        )

        if chroma_count != bm25_count:
            self._build_bm25_index()

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """Check whether a document is indexed."""

        return self.vector_store.has_document(
            document_id
        )

    def list_documents(self) -> list[dict]:
        """List indexed documents."""

        return self.vector_store.list_documents()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        document_id: str | None = None,
    ) -> list[dict]:
        """Run document-aware hybrid retrieval."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        self._refresh_if_needed()

        dense_results = self._dense_search(
            query=query,
            top_k=candidate_k,
            document_id=document_id,
        )

        bm25_results = self.bm25_service.search(
            query=query,
            top_k=candidate_k,
            document_id=document_id,
        )

        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        return fused_results[:top_k]

    def _dense_search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[dict]:
        """Run dense search, optionally within one document."""

        query_embedding = self.embedding_service.embed_text(
            query
        )

        where = None

        if document_id:
            where = {
                "document_id": document_id,
            }

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        dense_results = []

        for i in range(len(documents)):
            dense_results.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )

        return dense_results

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        rrf_k: int = 60,
    ) -> list[dict]:
        """Fuse rankings using Reciprocal Rank Fusion."""

        fused: dict[str, dict] = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            fused[chunk_id]["dense_rank"] = rank

            fused[chunk_id]["hybrid_score"] += (
                1 / (rrf_k + rank)
            )

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            fused[chunk_id]["bm25_rank"] = rank

            fused[chunk_id]["hybrid_score"] += (
                1 / (rrf_k + rank)
            )

        return sorted(
            fused.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )
