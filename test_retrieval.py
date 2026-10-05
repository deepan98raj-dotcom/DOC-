from app.services.retrieval_service import RetrievalService


retriever = RetrievalService()

results = retriever.retrieve(
    "What is linear algebra used for in machine learning?",
    top_k=5,
)

print("Results:", len(results))

for index, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print()
    print(f"Result {index}")
    print("-" * 60)
    print("Chunk ID:", result["chunk_id"])
    print("Page:", metadata.get("page_number"))
    print("Distance:", round(result["distance"], 4))
    print("Source:", metadata.get("source"))
    print("Text:")
    print(result["text"][:500])
    