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

document_id = "multimodal-test-001"

service = MultimodalPDFService()

result = service.extract_assets(
    file_path=str(pdf_path),
    document_id=document_id,
)

print("=" * 80)
print("DOCMIND AI - MULTIMODAL PDF EXTRACTION")
print("=" * 80)

print()
print("PDF:")
print(pdf_path)

print()
print("Pages:", result["page_count"])
print(
    "Visual pages:",
    result["visual_page_count"],
)
print(
    "Embedded images:",
    result["image_count"],
)
print(
    "Tables:",
    result["table_count"],
)
print(
    "Pages containing drawings:",
    result["drawing_page_count"],
)

print()
print("Visual pages:")
print("-" * 80)

for page in result["pages"]:
    if page["has_visual_content"]:
        print(
            f'Page {page["page_number"]} | '
            f'Images: {page["image_count"]} | '
            f'Drawings: {page["drawing_count"]} | '
            f'Tables: {page["table_count"]} | '
            f'Render: '
            f'{page.get("rendered_page_path", "none")}'
        )

print()
print("Extracted images:")
print("-" * 80)

for image in result["images"][:10]:
    print(
        f'{image["image_id"]} | '
        f'Page: {image["page_number"]} | '
        f'Size: {image.get("width")}x'
        f'{image.get("height")} | '
        f'File: {image.get("file_path")}'
    )

print()
print("Extracted tables:")
print("-" * 80)

for table in result["tables"][:10]:
    print(
        f'{table["table_id"]} | '
        f'Page: {table["page_number"]}'
    )

print()
print(
    "Manifest:",
    (
        "data/processed/"
        f"{document_id}/multimodal_manifest.json"
    ),
)
