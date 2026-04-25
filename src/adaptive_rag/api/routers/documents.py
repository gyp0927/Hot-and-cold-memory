"""Document ingestion endpoints."""

import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from adaptive_rag.api.schemas.document import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentResponse,
)
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.extractors.pdf import extract_pdf
from adaptive_rag.ingestion.extractors.docx import extract_docx
from adaptive_rag.ingestion.extractors.image import extract_image
from adaptive_rag.ingestion.extractors.text import extract_text_from_bytes
from adaptive_rag.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

# Global pipeline instance
_pipeline: IngestionPipeline | None = None


def set_pipeline(pipeline: IngestionPipeline) -> None:
    """Set the global ingestion pipeline."""
    global _pipeline
    _pipeline = pipeline


def _extract_text(filename: str, content: bytes) -> str:
    """Extract text from file based on extension.

    Args:
        filename: Original filename.
        content: File content as bytes.

    Returns:
        Extracted text.

    Raises:
        HTTPException: If file type is unsupported or extraction fails.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".txt", ".md", ".markdown", ".rst", ".json", ".csv"):
        return extract_text_from_bytes(content)

    elif ext == ".pdf":
        return extract_pdf(content)

    elif ext in (".docx", ".doc"):
        return extract_docx(content)

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"):
        return extract_image(content)

    else:
        # Try UTF-8 text as fallback
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf, .docx, .png, .jpg, .jpeg"
            )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    tags: str = Form(""),
) -> DocumentUploadResponse:
    """Upload a document for ingestion.

    Supports: .txt, .md, .pdf, .docx, .png, .jpg, .jpeg
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        content = await file.read()
        text = _extract_text(file.filename or "upload", content)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content extracted from file")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        result = await _pipeline.ingest_text(
            text=text,
            source_uri=file.filename or "upload",
            title=title or file.filename,
            tags=tag_list,
        )

        return DocumentUploadResponse(
            document_id=result.document_id,
            status=result.status,
            chunks_created=result.chunks_created,
            message="Document ingested successfully" if result.status == "success" else result.error or "Failed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("upload_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/text", response_model=DocumentUploadResponse)
async def upload_text(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """Upload text content directly."""
    raise HTTPException(status_code=501, detail="Direct text upload not yet implemented")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    """Get document status and metadata."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
