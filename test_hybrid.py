from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)


retriever = HybridRetrievalService()

query = "What is linear algebra used for in machine learning?"

results = retriever.retrieve(
    query=query,
    top_k=5,
    candidate_k=20,
)

print("Query:", query)
print("Results:", len(results))

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
    print("Dense Rank:", result["dense_rank"])
    print("BM25 Rank:", result["bm25_rank"])
    print(
        "Hybrid Score:",
        round(result["hybrid_score"], 6),
    )
    print("Source:", metadata.get("source"))
    print("Text:")
    print(result["text"][:500])
