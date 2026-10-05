from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class NormalizedChunk:
    """Standardized representation of a searchable document chunk."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    word_count: int
    source: str
    created_at: str
    metadata: dict[str, Any]


class NormalizationService:
    """Normalize extracted document chunks into a consistent schema."""

    def normalize_chunk(
        self,
        chunk_id: str,
        document_id: str,
        filename: str,
        page_number: int,
        chunk_index: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> NormalizedChunk:
        text = text.strip()

        if not text:
            raise ValueError("Chunk text cannot be empty.")

        word_count = len(text.split())

        created_at = datetime.now(timezone.utc).isoformat()

        return NormalizedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            chunk_index=chunk_index,
            text=text,
            word_count=word_count,
            source=filename,
            created_at=created_at,
            metadata=metadata or {},
        )
        