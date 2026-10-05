from app.services.rag_service import RAGService


DOCUMENT_ID = "ml-math-001"

query = (
    "What mathematical concepts are essential for machine learning, "
    "and why are they important?"
)

print("=" * 80)
print("DOCMIND AI - END-TO-END RAG TEST")
print("=" * 80)
print()
print("Document:", DOCUMENT_ID)
print("Query:", query)

try:
    rag = RAGService(
        top_k=5,
        candidate_k=20,
    )

    print()
    print("LLM provider:", rag.llm.provider)
    print("LLM model:", rag.llm.model_name)

    result = rag.answer(
        query=query,
        document_id=DOCUMENT_ID,
    )

    print()
    print("-" * 80)
    print("ANSWER")
    print("-" * 80)
    print(result["answer"])

    print()
    print("-" * 80)
    print("SOURCES")
    print("-" * 80)

    for source in result["sources"]:
        print()
        print("Source ID:", source["source_id"])
        print("Page:", source["page_number"])
        print("Source type:", source["source_type"])
        print("Content type:", source["content_type"])
        print("Chunk ID:", source["chunk_id"])
        print("Reranker score:", source["reranker_score"])

except Exception as exc:
    print()
    print("=" * 80)
    print("RAG TEST FAILED")
    print("=" * 80)
    print(type(exc).__name__ + ":", exc)