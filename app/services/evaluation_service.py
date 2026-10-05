from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuery:
    """Ground-truth question for retrieval evaluation."""

    query: str
    relevant_pages: set[int]


class RetrievalEvaluator:
    """Calculate standard information-retrieval metrics."""

    @staticmethod
    def _result_pages(
        results: list[dict],
        k: int,
    ) -> list[int]:
        """Extract page numbers from the top-k results."""

        pages = []

        for result in results[:k]:
            metadata = result.get("metadata", {})
            page_number = metadata.get("page_number")

            if page_number is not None:
                pages.append(int(page_number))

        return pages

    def hit_at_k(
        self,
        results: list[dict],
        relevant_pages: set[int],
        k: int,
    ) -> float:
        """Return 1 if any relevant page appears in top-k."""

        result_pages = self._result_pages(results, k)

        return float(
            any(
                page in relevant_pages
                for page in result_pages
            )
        )

    def reciprocal_rank(
        self,
        results: list[dict],
        relevant_pages: set[int],
    ) -> float:
        """Return reciprocal rank of the first relevant result."""

        for rank, result in enumerate(
            results,
            start=1,
        ):
            metadata = result.get("metadata", {})
            page_number = metadata.get("page_number")

            if page_number is not None:
                if int(page_number) in relevant_pages:
                    return 1.0 / rank

        return 0.0

    def precision_at_k(
        self,
        results: list[dict],
        relevant_pages: set[int],
        k: int,
    ) -> float:
        """Calculate Precision@K."""

        result_pages = self._result_pages(results, k)

        if not result_pages:
            return 0.0

        relevant_count = sum(
            page in relevant_pages
            for page in result_pages
        )

        return relevant_count / len(result_pages)

    def recall_at_k(
        self,
        results: list[dict],
        relevant_pages: set[int],
        k: int,
    ) -> float:
        """Calculate Recall@K."""

        if not relevant_pages:
            return 0.0

        result_pages = self._result_pages(results, k)

        retrieved_relevant = sum(
            page in relevant_pages
            for page in result_pages
        )

        return min(
            retrieved_relevant / len(relevant_pages),
            1.0,
        )

    def evaluate(
        self,
        results_by_query: list[tuple[EvaluationQuery, list[dict]]],
        k_values: tuple[int, ...] = (1, 3, 5),
    ) -> dict:
        """Evaluate a complete retrieval system."""

        if not results_by_query:
            return {}

        metrics = {}

        for k in k_values:
            metrics[f"hit@{k}"] = sum(
                self.hit_at_k(
                    results,
                    evaluation_query.relevant_pages,
                    k,
                )
                for evaluation_query, results in results_by_query
            ) / len(results_by_query)

            metrics[f"precision@{k}"] = sum(
                self.precision_at_k(
                    results,
                    evaluation_query.relevant_pages,
                    k,
                )
                for evaluation_query, results in results_by_query
            ) / len(results_by_query)

            metrics[f"recall@{k}"] = sum(
                self.recall_at_k(
                    results,
                    evaluation_query.relevant_pages,
                    k,
                )
                for evaluation_query, results in results_by_query
            ) / len(results_by_query)

        metrics["mrr"] = sum(
            self.reciprocal_rank(
                results,
                evaluation_query.relevant_pages,
            )
            for evaluation_query, results in results_by_query
        ) / len(results_by_query)

        return metrics
