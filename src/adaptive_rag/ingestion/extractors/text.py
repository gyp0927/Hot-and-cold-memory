"""Plain text document extractor."""

from pathlib import Path

from adaptive_rag.core.exceptions import IngestionError

# 允许少量控制字符（如 BOM、零宽字符），但超过阈值说明是二进制文件
_MAX_BINARY_RATIO: float = 0.05


def _is_valid_text(text: str) -> bool:
    """Heuristic: reject strings that look like raw binary data.

    Counts characters that are neither printable nor common whitespace.
    If the ratio exceeds _MAX_BINARY_RATIO, the content is treated as
    binary garbage rather than legitimate text.
    """
    if not text:
        return False
    bad = sum(
        1
        for c in text
        if not (c.isprintable() or c in " \t\n\r")
    )
    return (bad / len(text)) <= _MAX_BINARY_RATIO


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
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="gbk")
        except Exception as e:
            raise IngestionError(f"Failed to read {file_path}: {e}") from e
    except Exception as e:
        raise IngestionError(f"Failed to read {file_path}: {e}") from e

    if not _is_valid_text(text):
        raise IngestionError(
            f"File {file_path} does not appear to be valid text "
            "(too many non-printable characters). "
            "If this is a PDF or DOCX, please use the correct extension."
        )
    return text


def extract_text_from_bytes(content: bytes) -> str:
    """Extract text from bytes.

    Args:
        content: File content as bytes.

    Returns:
        Decoded text.

    Raises:
        IngestionError: If content cannot be decoded or looks like binary.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gbk")
        except Exception as e:
            raise IngestionError(f"Failed to decode text content: {e}") from e

    if not _is_valid_text(text):
        raise IngestionError(
            "Content does not appear to be valid text "
            "(too many non-printable characters). "
            "If this is a PDF or DOCX, please upload with the correct extension."
        )
    return text
