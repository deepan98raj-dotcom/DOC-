from app.services.reranked_retrieval_service import RerankedRetrievalService

query = "What mathematical concepts are essential for machine learning?"

retriever = RerankedRetrievalService()

results = retriever.retrieve(
    query=query,
    top_k=5,
    candidate_k=20,
    document_id="ml-math-001",
)

print("=" * 80)
print("DOCMIND AI - HYBRID + RERANKER TEST")
print("=" * 80)
print()
print("Query:", query)
print("Result count:", len(results))

for i, result in enumerate(results, start=1):
    metadata = result.get("metadata", {})

    print()
    print(f"--- Result {i} ---")
    print("Chunk ID:", result.get("chunk_id"))
    print("Page:", metadata.get("page_number"))
    print("Source:", metadata.get("source_type"))
    print("Dense rank:", result.get("dense_rank"))
    print("BM25 rank:", result.get("bm25_rank"))
    print("Hybrid score:", result.get("hybrid_score"))
    print("Reranker score:", result.get("reranker_score"))