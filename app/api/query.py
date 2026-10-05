from fastapi import APIRouter, HTTPException

from app.schemas.query import (
    DocumentSummary,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_service import RAGService


router = APIRouter(
    tags=["RAG"],
)

rag_service = RAGService(
    top_k=5,
    candidate_k=20,
)


@router.get(
    "/documents",
    response_model=list[DocumentSummary],
)
async def list_indexed_documents():
    """List documents currently available for querying."""

    return rag_service.list_documents()


@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query_all_documents(
    request: QueryRequest,
):
    """Query across all indexed documents."""

    try:
        result = rag_service.answer(
            query=request.query,
        )

        return {
            "query": request.query,
            "document_id": None,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        ) from exc


@router.post(
    "/documents/{document_id}/query",
    response_model=QueryResponse,
)
async def query_document(
    document_id: str,
    request: QueryRequest,
):
    """Query only the selected document."""

    if not rag_service.document_exists(
        document_id
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document '{document_id}' "
                "is not indexed."
            ),
        )

    try:
        result = rag_service.answer(
            query=request.query,
            document_id=document_id,
        )

        return {
            "query": request.query,
            "document_id": document_id,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        ) from exc
