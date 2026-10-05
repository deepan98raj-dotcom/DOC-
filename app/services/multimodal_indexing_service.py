from typing import Any

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class MultimodalIndexingService:
    """Index tables, images, and visual pages into ChromaDB."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def index_assets(
        self,
        manifest: dict[str, Any],
        filename: str,
    ) -> int:
        """Index multimodal assets from a PDF manifest."""

        document_id = manifest.get("document_id")

        if not document_id:
            raise ValueError(
                "Manifest must contain document_id."
            )

        records = []

        # --------------------------------------------------
        # Tables
        # --------------------------------------------------

        for table in manifest.get("tables", []):
            if table.get("error"):
                continue

            table_id = table["table_id"]

            markdown = (
                table.get("markdown") or ""
            ).strip()

            if not markdown:
                rows = table.get(
                    "rows",
                    [],
                )

                row_text = []

                for row in rows:
                    row_text.append(
                        " | ".join(
                            str(cell)
                            if cell is not None
                            else ""
                            for cell in row
                        )
                    )

                markdown = "\n".join(
                    row_text
                )

            if not markdown:
                continue

            page_number = table.get(
                "page_number"
            )

            searchable_text = (
                f"Table from {filename}. "
                f"Page {page_number}. "
                f"\n\n{markdown}"
            )

            records.append(
                {
                    "id": table_id,
                    "text": searchable_text,
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_number,
                        "content_type": "table",
                        "source_type": "table",
                        "asset_id": table_id,
                        "asset_path": (
                            f"data/processed/"
                            f"{document_id}/tables/"
                            f"{table_id}.json"
                        ),
                    },
                }
            )

        # --------------------------------------------------
        # Images
        # --------------------------------------------------

        for image in manifest.get("images", []):
            if image.get("error"):
                continue

            image_id = image["image_id"]
            page_number = image.get(
                "page_number"
            )

            width = image.get(
                "width",
                "unknown",
            )

            height = image.get(
                "height",
                "unknown",
            )

            extension = image.get(
                "extension",
                "unknown",
            )

            # This is intentionally metadata-based until
            # Vision-LLM image understanding is added.
            searchable_text = (
                f"Visual image asset from {filename}. "
                f"Page {page_number}. "
                f"Embedded image with dimensions "
                f"{width} by {height}. "
                f"Image format: {extension}. "
                f"This page contains visual content."
            )

            records.append(
                {
                    "id": image_id,
                    "text": searchable_text,
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_number,
                        "content_type": "image",
                        "source_type": "image",
                        "asset_id": image_id,
                        "asset_path": image.get(
                            "file_path"
                        ),
                        "width": width,
                        "height": height,
                        "extension": extension,
                    },
                }
            )

        # --------------------------------------------------
        # Visual pages / vector drawings
        # --------------------------------------------------

        for page in manifest.get("pages", []):
            if not page.get(
                "has_visual_content"
            ):
                continue

            if page.get(
                "image_count",
                0,
            ) > 0:
                continue

            page_number = page[
                "page_number"
            ]

            drawing_count = page.get(
                "drawing_count",
                0,
            )

            table_count = page.get(
                "table_count",
                0,
            )

            page_id = (
                f"{document_id}"
                f"-visual-page-{page_number:04d}"
            )

            searchable_text = (
                f"Visual page from {filename}. "
                f"Page {page_number}. "
                f"Contains {drawing_count} "
                f"vector drawing objects and "
                f"{table_count} detected tables. "
                f"This page contains visual content."
            )

            records.append(
                {
                    "id": page_id,
                    "text": searchable_text,
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_number,
                        "content_type": "visual_page",
                        "source_type": "visual_page",
                        "asset_id": page_id,
                        "asset_path": page.get(
                            "rendered_page_path"
                        ),
                        "drawing_count": drawing_count,
                        "table_count": table_count,
                    },
                }
            )

        if not records:
            return 0

        texts = [
            record["text"]
            for record in records
        ]

        ids = [
            record["id"]
            for record in records
        ]

        metadatas = [
            record["metadata"]
            for record in records
        ]

        embeddings = (
            self.embedding_service.embed_documents(
                texts
            )
        )

        self.vector_store.add_documents(
            chunk_ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(records)
