import pytest

from app.services.chunking_service import ChunkingService


def test_chunk_text_creates_chunks():
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = (
        "This is a test document with enough content "
        "to demonstrate overlapping chunks in the "
        "DocMind AI RAG pipeline."
    )

    chunks = service.chunk_text(
        text=text,
        document_id="doc123",
        page_number=1,
    )

    assert len(chunks) > 0
    assert chunks[0].document_id == "doc123"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_empty_text_returns_empty_list():
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = service.chunk_text(
        text="",
        document_id="doc123",
        page_number=1,
    )

    assert chunks == []


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError):
        ChunkingService(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_ids_are_unique():
    service = ChunkingService(
        chunk_size=50,
        chunk_overlap=10,
    )

    text = (
        "This is a longer document used to verify "
        "that every generated chunk has a unique "
        "identifier."
    )

    chunks = service.chunk_text(
        text=text,
        document_id="doc123",
        page_number=2,
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    