from sentence_transformers import CrossEncoder


class RerankerService:
    """Rerank retrieved documents using a Cross-Encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank candidate documents for the given query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            result["reranker_score"] = float(score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return reranked_results[:top_k]
