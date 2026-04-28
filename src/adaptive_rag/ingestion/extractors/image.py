"""Image OCR extractor (requires Tesseract)."""

from io import BytesIO

from adaptive_rag.core.exceptions import IngestionError


def extract_image(file_bytes: bytes) -> str:
    """Extract text from an image using OCR.

    Args:
        file_bytes: Image file content as bytes.

    Returns:
        Extracted text.

    Raises:
        IngestionError: If OCR is not available or extraction fails.
    """
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        return text.strip()
    except ImportError as e:
        raise IngestionError(
            "OCR not available. To process images, install Tesseract OCR: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        ) from e
    except Exception as e:
        raise IngestionError(f"Failed to extract image text: {e}") from e
