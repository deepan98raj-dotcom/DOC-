from app.services.document_pipeline import DocumentPipeline
from app.services.embedding_service import EmbeddingService
from app.services.multimodal_indexing_service import (
    MultimodalIndexingService,
)
from app.services.vector_store import VectorStore


class IndexingService:
    """Create embeddings and index document content."""

    def __init__(self):
        self.pipeline = DocumentPipeline(
            chunk_size=800,
            chunk_overlap=120,
        )

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        self.multimodal_indexing = (
            MultimodalIndexingService()
        )

    def index_document(
        self,
        file_path: str,
        document_id: str,
        filename: str,
    ) -> int:
        """Process and index text chunks."""

        chunks = self.pipeline.process(
            file_path=file_path,
            document_id=document_id,
            filename=filename,
        )

        if not chunks:
            return 0

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_service.embed_documents(
                texts
            )
        )

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "word_count": chunk.word_count,
                "source": chunk.source,
                "content_type": "text",
                "source_type": "text",
            }
            for chunk in chunks
        ]

        self.vector_store.add_documents(
            chunk_ids=chunk_ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def index_multimodal_assets(
        self,
        manifest: dict,
        filename: str,
    ) -> int:
        """Index tables, images and visual pages."""

        return self.multimodal_indexing.index_assets(
            manifest=manifest,
            filename=filename,
        )
