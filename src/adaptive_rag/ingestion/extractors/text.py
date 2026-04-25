"""Plain text document extractor."""

from pathlib import Path

from adaptive_rag.core.exceptions import IngestionError


def extract_text(file_path: str) -> str:
    """Extract text from a plain text file.

    Args:
        file_path: Path to the text file.

    Returns:
        Extracted text content.

    Raises:
        IngestionError: If file cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise IngestionError(f"File not found: {file_path}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            return path.read_text(encoding="gbk")
        except Exception as e:
            raise IngestionError(f"Failed to read {file_path}: {e}") from e
    except Exception as e:
        raise IngestionError(f"Failed to read {file_path}: {e}") from e


def extract_text_from_bytes(content: bytes) -> str:
    """Extract text from bytes.

    Args:
        content: File content as bytes.

    Returns:
        Decoded text.

    Raises:
        IngestionError: If content cannot be decoded.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return content.decode("gbk")
        except Exception as e:
            raise IngestionError(f"Failed to decode text content: {e}") from e
