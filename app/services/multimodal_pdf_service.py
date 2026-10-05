from pathlib import Path
from typing import Any
import json

import pymupdf


class MultimodalPDFService:
    """Extract reliable multimodal assets from PDFs."""

    def __init__(
        self,
        output_root: str = "data/processed",
        render_visual_pages: bool = True,
        min_table_rows: int = 2,
        min_table_columns: int = 2,
        min_table_non_empty_cells: int = 4,
        min_drawings_for_visual: int = 20,
    ):
        self.output_root = Path(output_root)
        self.render_visual_pages = render_visual_pages

        self.min_table_rows = min_table_rows
        self.min_table_columns = min_table_columns
        self.min_table_non_empty_cells = (
            min_table_non_empty_cells
        )
        self.min_drawings_for_visual = (
            min_drawings_for_visual
        )

    def extract_assets(
        self,
        file_path: str,
        document_id: str,
    ) -> dict[str, Any]:
        """Extract images, validated tables and visual pages."""

        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {file_path}"
            )

        document_output = (
            self.output_root / document_id
        )

        images_output = document_output / "images"
        pages_output = document_output / "pages"
        tables_output = document_output / "tables"

        images_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        pages_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        tables_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        pages = []
        extracted_images = []
        extracted_tables = []

        seen_xrefs: set[int] = set()

        with pymupdf.open(pdf_path) as document:

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                page_images = page.get_images(
                    full=True
                )

                drawings = page.get_drawings()

                # Use strict table detection.
                try:
                    table_finder = page.find_tables(
                        strategy="lines_strict"
                    )

                    detected_tables = getattr(
                        table_finder,
                        "tables",
                        [],
                    )

                except Exception as exc:
                    print(
                        f"Warning: table detection failed "
                        f"on page {page_number}: {exc}"
                    )
                    detected_tables = []

                # ------------------------------------------
                # Validate tables
                # ------------------------------------------

                valid_tables = []

                for table in detected_tables:

                    try:
                        rows = table.extract()

                        if not rows:
                            continue

                        row_count = len(rows)

                        column_count = max(
                            (
                                len(row)
                                for row in rows
                            ),
                            default=0,
                        )

                        non_empty_cells = sum(
                            1
                            for row in rows
                            for cell in row
                            if cell is not None
                            and str(cell).strip()
                        )

                        if (
                            row_count
                            < self.min_table_rows
                        ):
                            continue

                        if (
                            column_count
                            < self.min_table_columns
                        ):
                            continue

                        if (
                            non_empty_cells
                            < self.min_table_non_empty_cells
                        ):
                            continue

                        valid_tables.append(
                            table
                        )

                    except Exception:
                        continue

                # ------------------------------------------
                # Detect visual content
                # ------------------------------------------

                has_visual_content = bool(
                    page_images
                    or valid_tables
                    or (
                        len(drawings)
                        >= self.min_drawings_for_visual
                    )
                )

                page_info = {
                    "page_number": page_number,
                    "image_count": len(
                        page_images
                    ),
                    "drawing_count": len(
                        drawings
                    ),
                    "table_count": len(
                        valid_tables
                    ),
                    "has_visual_content": (
                        has_visual_content
                    ),
                }

                pages.append(page_info)

                # ------------------------------------------
                # Embedded images
                # ------------------------------------------

                for image_index, image_info in enumerate(
                    page_images,
                    start=1,
                ):
                    xref = image_info[0]

                    if xref in seen_xrefs:
                        continue

                    seen_xrefs.add(xref)

                    try:
                        image_data = (
                            document.extract_image(
                                xref
                            )
                        )

                        if not image_data:
                            continue

                        extension = image_data[
                            "ext"
                        ]

                        image_number = (
                            len(extracted_images) + 1
                        )

                        image_path = (
                            images_output
                            / f"image_{image_number:04d}.{extension}"
                        )

                        image_path.write_bytes(
                            image_data["image"]
                        )

                        extracted_images.append(
                            {
                                "image_id": (
                                    f"{document_id}"
                                    f"-img-{image_number:04d}"
                                ),
                                "document_id": document_id,
                                "page_number": page_number,
                                "image_index": image_index,
                                "xref": xref,
                                "width": image_data.get(
                                    "width"
                                ),
                                "height": image_data.get(
                                    "height"
                                ),
                                "extension": extension,
                                "file_path": str(
                                    image_path
                                ),
                            }
                        )

                    except Exception as exc:
                        print(
                            f"Warning: image extraction failed "
                            f"on page {page_number}: {exc}"
                        )

                # ------------------------------------------
                # Tables
                # ------------------------------------------

                for table_index, table in enumerate(
                    valid_tables,
                    start=1,
                ):
                    try:
                        rows = table.extract()

                        table_number = (
                            len(extracted_tables) + 1
                        )

                        markdown = ""

                        to_markdown = getattr(
                            table,
                            "to_markdown",
                            None,
                        )

                        if callable(to_markdown):
                            markdown = (
                                to_markdown()
                            )

                        table_json_path = (
                            tables_output
                            / f"table_{table_number:04d}.json"
                        )

                        table_md_path = (
                            tables_output
                            / f"table_{table_number:04d}.md"
                        )

                        table_record = {
                            "table_id": (
                                f"{document_id}"
                                f"-table-{table_number:04d}"
                            ),
                            "document_id": document_id,
                            "page_number": page_number,
                            "table_index": table_index,
                            "bbox": (
                                list(table.bbox)
                                if table.bbox
                                else None
                            ),
                            "rows": rows,
                            "markdown": markdown,
                            "json_path": str(
                                table_json_path
                            ),
                            "markdown_path": (
                                str(table_md_path)
                                if markdown
                                else None
                            ),
                            "row_count": len(
                                rows
                            ),
                            "column_count": max(
                                (
                                    len(row)
                                    for row in rows
                                ),
                                default=0,
                            ),
                        }

                        extracted_tables.append(
                            table_record
                        )

                        table_json_path.write_text(
                            json.dumps(
                                table_record,
                                indent=2,
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )

                        if markdown:
                            table_md_path.write_text(
                                markdown,
                                encoding="utf-8",
                            )

                    except Exception as exc:
                        print(
                            f"Warning: table extraction failed "
                            f"on page {page_number}: {exc}"
                        )

                # ------------------------------------------
                # Render visually interesting pages
                # ------------------------------------------

                if (
                    self.render_visual_pages
                    and has_visual_content
                ):
                    try:
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(
                                1.5,
                                1.5,
                            ),
                            alpha=False,
                        )

                        page_path = (
                            pages_output
                            / f"page_{page_number:04d}.png"
                        )

                        pixmap.save(
                            str(page_path)
                        )

                        page_info[
                            "rendered_page_path"
                        ] = str(page_path)

                    except Exception as exc:
                        page_info[
                            "render_error"
                        ] = str(exc)

        manifest = {
            "document_id": document_id,
            "pdf_path": str(pdf_path),
            "page_count": len(pages),
            "visual_page_count": sum(
                page["has_visual_content"]
                for page in pages
            ),
            "image_count": len(
                extracted_images
            ),
            "table_count": len(
                extracted_tables
            ),
            "drawing_page_count": sum(
                page["drawing_count"] > 0
                for page in pages
            ),
            "pages": pages,
            "images": extracted_images,
            "tables": extracted_tables,
        }

        manifest_path = (
            document_output
            / "multimodal_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return manifest
