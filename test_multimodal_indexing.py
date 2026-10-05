from pathlib import Path

from app.services.multimodal_pdf_service import (
    MultimodalPDFService,
)
from app.services.multimodal_indexing_service import (
    MultimodalIndexingService,
)
from app.services.vector_store import VectorStore


pdf_files = list(
    Path("data/uploads").glob("*.pdf")
)

if not pdf_files:
    raise RuntimeError(
        "No PDF files found."
    )

pdf_path = pdf_files[0]

document_id = "ml-math-001"

filename = (
    "Mathematics for machine learning.pdf"
)

print("=" * 80)
print("DOCMIND AI - MULTIMODAL INDEXING TEST")
print("=" * 80)

extractor = MultimodalPDFService()

manifest = extractor.extract_assets(
    file_path=str(pdf_path),
    document_id=document_id,
)

print()
print("PDF:", pdf_path)
print("Tables:", manifest["table_count"])
print("Images:", manifest["image_count"])
print(
    "Visual pages:",
    manifest["visual_page_count"],
)

indexer = MultimodalIndexingService()

indexed_count = indexer.index_assets(
    manifest=manifest,
    filename=filename,
)

print()
print(
    "Multimodal records indexed:",
    indexed_count,
)

store = VectorStore()

print(
    "Total ChromaDB records:",
    store.count(),
)

print()
print("Multimodal indexing complete.")
