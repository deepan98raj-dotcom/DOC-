from pathlib import Path

import pymupdf


class DocumentService:
    """Service responsible for PDF validation and document inspection."""

    ALLOWED_EXTENSION = ".pdf"

    def validate_pdf(self, filename: str) -> None:
        """Validate that the uploaded file is a PDF."""

        if not filename:
            raise ValueError("Filename is required.")

        if Path(filename).suffix.lower() != self.ALLOWED_EXTENSION:
            raise ValueError("Only PDF files are supported.")

    def extract_text(self, file_path: str) -> dict:
        """Extract structured information from a PDF."""

        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        pages = []
        total_text_length = 0
        total_word_count = 0
        total_images = 0

        with pymupdf.open(pdf_path) as document:

            metadata = document.metadata

            page_count = len(document)

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                text = page.get_text("text").strip()

                words = text.split()
                word_count = len(words)

                images = page.get_images(full=True)
                image_count = len(images)

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "text_length": len(text),
                        "word_count": word_count,
                        "image_count": image_count,
                    }
                )

                total_text_length += len(text)
                total_word_count += word_count
                total_images += image_count

        return {
            "page_count": page_count,
            "text_length": total_text_length,
            "word_count": total_word_count,
            "image_count": total_images,
            "metadata": metadata,
            "pages": pages,
        }
