from pathlib import Path
import json

from app.services.vision_service import VisionService
from app.services.vision_indexing_service import (
    VisionIndexingService,
)


DOCUMENT_ID = "ml-math-001"

MANIFEST_PATH = (
    Path("data/processed")
    / DOCUMENT_ID
    / "multimodal_manifest.json"
)

MAX_PAGES = 5


if not MANIFEST_PATH.exists():
    raise RuntimeError(
        f"Manifest not found: {MANIFEST_PATH}"
    )


manifest = json.loads(
    MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)

filename = Path(
    manifest["pdf_path"]
).name


# Select the strongest visual candidates.
candidates = []

for page in manifest.get("pages", []):

    if not page.get(
        "has_visual_content"
    ):
        continue

    rendered_path = page.get(
        "rendered_page_path"
    )

    if not rendered_path:
        continue

    image_path = Path(
        rendered_path
    )

    if not image_path.exists():
        continue

    score = (
        page.get("image_count", 0) * 1000
        + page.get("table_count", 0) * 100
        + page.get("drawing_count", 0)
    )

    candidates.append(
        (
            score,
            page,
        )
    )


candidates.sort(
    key=lambda item: item[0],
    reverse=True,
)

selected_pages = [
    page
    for _, page in candidates[:MAX_PAGES]
]


if not selected_pages:
    raise RuntimeError(
        "No rendered visual pages were found."
    )


print("=" * 80)
print("DOCMIND AI - VISION ANALYSIS")
print("=" * 80)

print()
print("Document:", DOCUMENT_ID)

print(
    "Vision model:",
    "qwen2.5vl:3b",
)

print(
    "Selected pages:",
    [
        page["page_number"]
        for page in selected_pages
    ],
)


vision_service = VisionService()

vision_indexer = VisionIndexingService()

analyses = []


for page in selected_pages:

    page_number = page[
        "page_number"
    ]

    image_path = page[
        "rendered_page_path"
    ]

    print()
    print("-" * 80)
    print(
        f"Analyzing page {page_number}"
    )

    print(
        "Image:",
        image_path,
    )

    try:

        analysis = (
            vision_service.analyze_image(
                image_path=image_path,
                document_id=DOCUMENT_ID,
                page_number=page_number,
            )
        )

        analysis["asset_id"] = (
            f"{DOCUMENT_ID}"
            f"-visual-page-{page_number:04d}"
        )

        analyses.append(
            analysis
        )

        print()
        print(
            "Visual type:",
            analysis.get(
                "visual_type",
                "unknown",
            ),
        )

        print(
            "Summary:",
            analysis.get(
                "summary",
                "",
            ),
        )

        print(
            "Description:",
            analysis.get(
                "detailed_description",
                "",
            ),
        )

        print(
            "Visible text:",
            analysis.get(
                "text_content",
                "",
            ),
        )

        print(
            "Key elements:",
            analysis.get(
                "key_elements",
                [],
            ),
        )

        record_id = (
            vision_indexer.index_analysis(
                analysis=analysis,
                filename=filename,
                content_type="visual_page",
            )
        )

        print(
            "Indexed record:",
            record_id,
        )

    except Exception as exc:

        print()
        print(
            f"Vision analysis failed on page "
            f"{page_number}: {exc}"
        )


manifest[
    "vision_analyses"
] = analyses


MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("VISION ANALYSIS COMPLETE")
print("=" * 80)

print(
    "Successful analyses:",
    len(analyses),
)

print(
    "Manifest:",
    MANIFEST_PATH,
)
