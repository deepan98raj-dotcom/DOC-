from typing import Any

from pydantic import BaseModel


class PageContent(BaseModel):
    page_number: int
    text: str
    text_length: int
    word_count: int
    image_count: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    text_length: int
    word_count: int
    image_count: int

    indexed_chunk_count: int
    indexed_multimodal_count: int

    visual_page_count: int
    extracted_image_count: int
    table_count: int
    drawing_page_count: int

    multimodal_manifest_path: str

    metadata: dict[str, Any]
    pages: list[PageContent]
