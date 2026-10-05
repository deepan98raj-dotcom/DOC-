from pathlib import Path
import shutil

from app.services.document_service import DocumentService
from app.services.indexing_service import IndexingService
from app.services.multimodal_pdf_service import (
    MultimodalPDFService,
)
from app.services.vector_store import VectorStore


DOCUMENT_ID = "ml-math-001"

UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path(
    "data/processed"
) / DOCUMENT_ID


pdf_files = sorted(
    UPLOAD_DIR.glob("*.pdf"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not pdf_files:
    raise RuntimeError(
        "No PDF files found in data/uploads."
    )

pdf_path = pdf_files[0]

filename = pdf_path.name

print("=" * 80)
print("DOCMIND AI - CANONICAL DOCUMENT REBUILD")
print("=" * 80)

print()
print("Selected PDF:")
print(pdf_path)

# Remove only the previous canonical processed assets.
if PROCESSED_DIR.exists():
    shutil.rmtree(
        PROCESSED_DIR
    )

document_service = DocumentService()

print()
print("Extracting PDF text...")

extracted = (
    document_service.extract_text(
        str(pdf_path)
    )
)

print(
    "Pages:",
    extracted["page_count"],
)

print(
    "Words:",
    extracted["word_count"],
)

indexing_service = IndexingService()

print()
print("Indexing text chunks...")

text_count = (
    indexing_service.index_document(
        file_path=str(pdf_path),
        document_id=DOCUMENT_ID,
        filename=filename,
    )
)

print(
    "Text chunks indexed:",
    text_count,
)

print()
print("Extracting multimodal assets...")

multimodal_service = MultimodalPDFService()

manifest = multimodal_service.extract_assets(
    file_path=str(pdf_path),
    document_id=DOCUMENT_ID,
)

print(
    "Embedded images:",
    manifest["image_count"],
)

print(
    "Validated tables:",
    manifest["table_count"],
)

print(
    "Visual pages:",
    manifest["visual_page_count"],
)

print()
print("Indexing multimodal assets...")

multimodal_count = (
    indexing_service.index_multimodal_assets(
        manifest=manifest,
        filename=filename,
    )
)

print(
    "Multimodal records indexed:",
    multimodal_count,
)

store = VectorStore()

print()
print(
    "TOTAL CHROMADB RECORDS:",
    store.count(),
)

print()
print(
    "Canonical document rebuild complete."
)
