import chromadb


PERSIST_DIRECTORY = "data/chroma"
COLLECTION_NAME = "docmind_documents"


client = chromadb.PersistentClient(
    path=PERSIST_DIRECTORY
)

try:
    client.delete_collection(
        COLLECTION_NAME
    )
    print(
        f"Deleted collection: {COLLECTION_NAME}"
    )
except Exception as exc:
    print(
        f"Collection delete notice: {exc}"
    )

client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "DocMind AI document embeddings"
    },
)

collection = client.get_collection(
    COLLECTION_NAME
)

print(
    "New ChromaDB collection created."
)

print(
    "Record count:",
    collection.count(),
)
