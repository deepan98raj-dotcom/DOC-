from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for document question answering."""

    query: str = Field(
        ...,
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    candidate_k: int = Field(
        default=20,
        ge=5,
        le=100,
    )


class SourceResponse(BaseModel):
    """Citation/source information."""

    source_id: str
    filename: str
    page_number: int | str
    chunk_id: str
    content_type: str
    source_type: str
    asset_id: str | None
    asset_path: str | None
    reranker_score: float | None


class QueryResponse(BaseModel):
    """RAG answer response."""

    query: str
    document_id: str | None
    answer: str
    sources: list[SourceResponse]


class DocumentSummary(BaseModel):
    """Indexed document summary."""

    document_id: str
    filename: str
    chunk_count: int
