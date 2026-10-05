import re

from rank_bm25 import BM25Okapi


class BM25Service:
    """Perform sparse keyword retrieval using BM25."""

    def __init__(self):
        self.chunk_ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def build_index(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Build the BM25 index."""

        if not (
            len(chunk_ids)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "chunk_ids, documents, and metadatas "
                "must have the same length."
            )

        self.chunk_ids = chunk_ids
        self.documents = documents
        self.metadatas = metadatas

        tokenized_documents = [
            self._tokenize(document)
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """Search using BM25, optionally within one document."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        candidate_indices = range(len(scores))

        if document_id:
            candidate_indices = [
                index
                for index in candidate_indices
                if self.metadatas[index].get(
                    "document_id"
                ) == document_id
            ]

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "chunk_id": self.chunk_ids[index],
                    "text": self.documents[index],
                    "metadata": self.metadatas[index],
                    "score": float(scores[index]),
                }
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric words."""

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
