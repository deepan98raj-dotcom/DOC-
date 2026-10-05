from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents one searchable document chunk."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str


class ChunkingService:
    """Split extracted document text into searchable chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int,
    ) -> list[DocumentChunk]:
        """Split page text into overlapping chunks."""

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{document_id}"
                            f"-p{page_number}"
                            f"-c{chunk_index}"
                        ),
                        document_id=document_id,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    )
                )

                chunk_index += 1

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        return chunks
