from collections import Counter, defaultdict

from app.services.vector_store import VectorStore


store = VectorStore()

data = store.collection.get(
    include=["metadatas"]
)

document_counts = defaultdict(Counter)

for metadata in data.get(
    "metadatas",
    [],
):
    document_id = metadata.get(
        "document_id",
        "unknown",
    )

    content_type = metadata.get(
        "content_type",
        "unknown",
    )

    document_counts[
        document_id
    ][content_type] += 1


print("=" * 80)
print("DOCMIND AI - CHROMADB DOCUMENT INVENTORY")
print("=" * 80)

for document_id, counts in sorted(
    document_counts.items()
):
    print()
    print("Document ID:", document_id)

    for content_type, count in sorted(
        counts.items()
    ):
        print(
            f"  {content_type:<15} {count}"
        )

print()
print("Total ChromaDB records:", store.count())
