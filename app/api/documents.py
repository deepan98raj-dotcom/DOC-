from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.document import DocumentUploadResponse
from app.services.document_service import DocumentService
from app.services.indexing_service import IndexingService
from app.services.multimodal_pdf_service import (
    MultimodalPDFService,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

document_service = DocumentService()
indexing_service = IndexingService()
multimodal_service = MultimodalPDFService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload, extract, index and process a PDF."""

    try:
        document_service.validate_pdf(
            file.filename
        )

        document_id = str(uuid4())

        safe_filename = Path(
            file.filename
        ).name

        stored_filename = (
            f"{document_id}_{safe_filename}"
        )

        file_path = (
            UPLOAD_DIR / stored_filename
        )

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_path.write_bytes(content)

        # -----------------------------------------------
        # Text extraction
        # -----------------------------------------------

        extracted = (
            document_service.extract_text(
                str(file_path)
            )
        )

        # -----------------------------------------------
        # Text indexing
        # -----------------------------------------------

        indexed_chunk_count = (
            indexing_service.index_document(
                file_path=str(file_path),
                document_id=document_id,
                filename=safe_filename,
            )
        )

        # -----------------------------------------------
        # Multimodal extraction
        # -----------------------------------------------

        multimodal = (
            multimodal_service.extract_assets(
                file_path=str(file_path),
                document_id=document_id,
            )
        )

        # -----------------------------------------------
        # Multimodal indexing
        # -----------------------------------------------

        indexed_multimodal_count = (
            indexing_service.index_multimodal_assets(
                manifest=multimodal,
                filename=safe_filename,
            )
        )

        return {
            "document_id": document_id,
            "filename": safe_filename,
            "page_count": extracted["page_count"],
            "text_length": extracted["text_length"],
            "word_count": extracted["word_count"],
            "image_count": extracted["image_count"],
            "indexed_chunk_count": indexed_chunk_count,
            "indexed_multimodal_count": (
                indexed_multimodal_count
            ),
            "visual_page_count": (
                multimodal["visual_page_count"]
            ),
            "extracted_image_count": (
                multimodal["image_count"]
            ),
            "table_count": (
                multimodal["table_count"]
            ),
            "drawing_page_count": (
                multimodal["drawing_page_count"]
            ),
            "multimodal_manifest_path": (
                f"data/processed/"
                f"{document_id}/"
                "multimodal_manifest.json"
            ),
            "metadata": extracted["metadata"],
            "pages": extracted["pages"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed: "
                f"{exc}"
            ),
        ) from exc
