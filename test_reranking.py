from app.services.reranked_retrieval_service import (
    RerankedRetrievalService,
)


retriever = RerankedRetrievalService()

query = "What is linear algebra used for in machine learning?"

results = retriever.retrieve(
    query=query,
    top_k=5,
    candidate_k=20,
)

print("Query:", query)
print("Final Results:", len(results))

for index, result in enumerate(
    results,
    start=1,
):
    metadata = result["metadata"]

    print()
    print(f"Result {index}")
    print("-" * 60)
    print("Chunk ID:", result["chunk_id"])
    print("Page:", metadata.get("page_number"))
    print("Dense Rank:", result.get("dense_rank"))
    print("BM25 Rank:", result.get("bm25_rank"))
    print(
        "Hybrid Score:",
        round(result.get("hybrid_score", 0.0), 6),
    )
    print(
        "Reranker Score:",
        round(result["reranker_score"], 6),
    )
    print("Source:", metadata.get("source"))
    print("Text:")
    print(result["text"][:500])
