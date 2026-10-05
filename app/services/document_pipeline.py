from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.normalization_service import NormalizationService


class DocumentPipeline:
    """Run document extraction, chunking, and normalization."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        self.document_service = DocumentService()
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.normalization_service = NormalizationService()

    def process(
        self,
        file_path: str,
        document_id: str,
        filename: str,
    ) -> list:
        """Process a PDF into normalized searchable chunks."""

        extracted = self.document_service.extract_text(
            file_path
        )

        normalized_chunks = []

        for page in extracted["pages"]:
            chunks = self.chunking_service.chunk_text(
                text=page["text"],
                document_id=document_id,
                page_number=page["page_number"],
            )

            for chunk in chunks:
                normalized = self.normalization_service.normalize_chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    metadata={
                        "page_word_count": page["word_count"],
                        "page_image_count": page["image_count"],
                        "document_page_count": extracted["page_count"],
                    },
                )

                normalized_chunks.append(normalized)

        return normalized_chunks