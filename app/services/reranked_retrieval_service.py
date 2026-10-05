from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)
from app.services.reranker_service import RerankerService


class RerankedRetrievalService:
    """Hybrid retrieval followed by Cross-Encoder reranking."""

    def __init__(self):
        self.hybrid_retrieval = HybridRetrievalService()
        self.reranker = RerankerService()

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """Check whether a document is indexed."""

        return self.hybrid_retrieval.document_exists(
            document_id
        )

    def list_documents(self) -> list[dict]:
        """Return indexed documents."""

        return self.hybrid_retrieval.list_documents()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        document_id: str | None = None,
    ) -> list[dict]:
        """Retrieve and rerank document-aware results."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than "
                "or equal to top_k."
            )

        hybrid_candidates = self.hybrid_retrieval.retrieve(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            document_id=document_id,
        )

        return self.reranker.rerank(
            query=query,
            candidates=hybrid_candidates,
            top_k=top_k,
        )
