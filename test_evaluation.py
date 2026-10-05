from app.services.evaluation_service import (
    EvaluationQuery,
    RetrievalEvaluator,
)
from app.services.retrieval_service import RetrievalService
from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)
from app.services.reranked_retrieval_service import (
    RerankedRetrievalService,
)


EVALUATION_DATASET = [
    EvaluationQuery(
        query="What mathematical concepts are used in machine learning?",
        relevant_pages={3},
    ),
    EvaluationQuery(
        query="Is matrix multiplication commutative?",
        relevant_pages={11},
    ),
    EvaluationQuery(
        query="What is the formula for polynomial linear regression?",
        relevant_pages={83},
    ),
    EvaluationQuery(
        query="How can regularization address overfitting?",
        relevant_pages={86},
    ),
    EvaluationQuery(
        query="What is curve fitting regression in machine learning?",
        relevant_pages={14},
    ),
]


print("=" * 80)
print("DOCMIND AI - RETRIEVAL EVALUATION")
print("=" * 80)

print()
print("Loading Dense Retrieval...")
dense_retriever = RetrievalService()

print("Loading Hybrid Retrieval...")
hybrid_retriever = HybridRetrievalService()

print("Loading Hybrid + Reranker...")
reranked_retriever = RerankedRetrievalService()


def run_dense(query: str) -> list[dict]:
    return dense_retriever.retrieve(
        query=query,
        top_k=5,
    )


def run_hybrid(query: str) -> list[dict]:
    return hybrid_retriever.retrieve(
        query=query,
        top_k=5,
        candidate_k=20,
    )


def run_reranked(query: str) -> list[dict]:
    return reranked_retriever.retrieve(
        query=query,
        top_k=5,
        candidate_k=20,
    )


systems = {
    "Dense": run_dense,
    "Hybrid": run_hybrid,
    "Hybrid + Reranker": run_reranked,
}


evaluator = RetrievalEvaluator()


for system_name, retrieval_function in systems.items():

    print()
    print("=" * 80)
    print(system_name)
    print("=" * 80)

    results_by_query = []

    for evaluation_query in EVALUATION_DATASET:

        results = retrieval_function(
            evaluation_query.query
        )

        results_by_query.append(
            (
                evaluation_query,
                results,
            )
        )

        pages = [
            result["metadata"].get("page_number")
            for result in results
        ]

        print()
        print("Query:")
        print(evaluation_query.query)
        print("Expected pages:")
        print(evaluation_query.relevant_pages)
        print("Retrieved pages:")
        print(pages)

    metrics = evaluator.evaluate(
        results_by_query,
        k_values=(1, 3, 5),
    )

    print()
    print("METRICS")
    print("-" * 80)

    for name, value in metrics.items():
        print(f"{name:<15} {value:.4f}")
