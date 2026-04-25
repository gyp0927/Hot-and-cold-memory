"""PDF document extractor."""

from io import BytesIO

from adaptive_rag.core.exceptions import IngestionError


def extract_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file.

    Args:
        file_bytes: PDF file content as bytes.

    Returns:
        Extracted text.

    Raises:
        IngestionError: If extraction fails.
    """
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        raise IngestionError(f"Failed to extract PDF: {e}") from e
