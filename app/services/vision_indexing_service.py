from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class VisionIndexingService:
    """Index Vision-generated descriptions into ChromaDB."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def index_analysis(
        self,
        analysis: dict,
        filename: str,
        content_type: str = "visual_page",
    ) -> str:
        """Index one Vision analysis."""

        document_id = analysis["document_id"]
        page_number = analysis["page_number"]

        asset_id = analysis.get(
            "asset_id",
            f"{document_id}-visual-page-{page_number:04d}",
        )

        record_id = f"{asset_id}-vision"

        visual_type = analysis.get(
            "visual_type",
            "unknown",
        )

        summary = analysis.get(
            "summary",
            "",
        )

        detailed_description = analysis.get(
            "detailed_description",
            "",
        )

        text_content = analysis.get(
            "text_content",
            "",
        )

        key_elements = analysis.get(
            "key_elements",
            [],
        )

        if isinstance(key_elements, list):
            key_element_text = ", ".join(
                str(item)
                for item in key_elements
            )
        else:
            key_element_text = str(
                key_elements
            )

        searchable_text = (
            f"Visual content from {filename}. "
            f"Page {page_number}. "
            f"Visual type: {visual_type}.\n\n"
            f"Summary: {summary}\n\n"
            f"Description: {detailed_description}\n\n"
            f"Visible text: {text_content}\n\n"
            f"Key elements: {key_element_text}"
        )

        embedding = self.embedding_service.embed_text(
            searchable_text
        )

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "page_number": page_number,
            "content_type": content_type,
            "source_type": "vision",
            "asset_id": asset_id,
            "asset_path": analysis.get(
                "image_path"
            ),
            "visual_type": visual_type,
            "vision_model": analysis.get(
                "model"
            ),
            "vision_analyzed": True,
        }

        self.vector_store.add_documents(
            chunk_ids=[record_id],
            texts=[searchable_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        return record_id
