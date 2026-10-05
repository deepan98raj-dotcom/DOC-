from pathlib import Path

from app.services.multimodal_pdf_service import (
    MultimodalPDFService,
)


pdf_files = list(
    Path("data/uploads").glob("*.pdf")
)

if not pdf_files:
    raise RuntimeError(
        "No PDF files found in data/uploads."
    )

pdf_path = pdf_files[0]

document_id = "multimodal-quality-002"

service = MultimodalPDFService()

result = service.extract_assets(
    file_path=str(pdf_path),
    document_id=document_id,
)

print("=" * 80)
print("DOCMIND AI - MULTIMODAL QUALITY TEST")
print("=" * 80)

print()
print("Pages:", result["page_count"])
print("Validated tables:", result["table_count"])
print("Embedded images:", result["image_count"])
print(
    "Visual pages:",
    result["visual_page_count"],
)

print()
print("Validated tables")
print("-" * 80)

for table in result["tables"][:10]:
    print(
        f'{table["table_id"]} | '
        f'Page: {table["page_number"]} | '
        f'Rows: {table["row_count"]} | '
        f'Columns: {table["column_count"]}'
    )

    if table.get("markdown"):
        print(
            table["markdown"][:300]
        )

print()
print("Visual pages")
print("-" * 80)

for page in result["pages"]:
    if page["has_visual_content"]:
        print(
            f'Page {page["page_number"]} | '
            f'Images: {page["image_count"]} | '
            f'Drawings: {page["drawing_count"]} | '
            f'Tables: {page["table_count"]}'
        )

print()
print(
    "Manifest:",
    (
        f"data/processed/{document_id}/"
        "multimodal_manifest.json"
    ),
)
