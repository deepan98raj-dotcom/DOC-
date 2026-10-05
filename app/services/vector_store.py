from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    """Store and retrieve document embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "docmind_documents",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "DocMind AI document embeddings"
            },
        )

    def add_documents(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store document chunks and embeddings."""

        if not chunk_ids:
            return

        if not (
            len(chunk_ids)
            == len(texts)
            == len(embeddings)
            == len(metadatas)
        ):
            raise ValueError(
                "chunk_ids, texts, embeddings, and "
                "metadatas must have the same length."
            )

        self.collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict:
        """Search for similar chunks, optionally within one document."""

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def has_document(
        self,
        document_id: str,
    ) -> bool:
        """Check whether a document exists in the index."""

        result = self.collection.get(
            where={
                "document_id": document_id,
            },
            limit=1,
            include=[
                "metadatas",
            ],
        )

        return bool(result.get("ids"))

    def list_documents(self) -> list[dict]:
        """Return unique indexed documents."""

        result = self.collection.get(
            include=[
                "metadatas",
            ]
        )

        documents: dict[str, dict] = {}

        for metadata in result.get("metadatas", []):
            if not metadata:
                continue

            document_id = metadata.get("document_id")

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get(
                        "filename",
                        "Unknown",
                    ),
                    "chunk_count": 0,
                }

            documents[document_id]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda item: item["filename"].lower(),
        )

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self.collection.count()
