"""Image processing utilities for ad prediction."""

import base64
import io
from pathlib import Path
from typing import Union

from PIL import Image


def load_image_as_base64(image_source: Union[str, Path, bytes]) -> str:
    """
    Load an image from various sources and convert to base64.

    Args:
        image_source: Can be:
            - File path (str or Path)
            - Raw bytes
            - Already base64-encoded string (returns as-is)

    Returns:
        str: Base64-encoded image data (without data URI prefix)
    """
    # If it's already a base64 string, return it
    if isinstance(image_source, str):
        # Check if it looks like base64 (alphanumeric + /+=)
        if len(image_source) > 100 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in image_source[-20:]):
            return image_source
        # Otherwise treat as file path
        image_source = Path(image_source)

    # Load from file path
    if isinstance(image_source, Path):
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    else:
        image_bytes = image_source

    # Convert to base64
    return base64.b64encode(image_bytes).decode("utf-8")


def resize_image_if_needed(
    image_source: Union[str, Path, bytes],
    max_width: int = 2048,
    max_height: int = 2048,
) -> bytes:
    """
    Resize image if it exceeds maximum dimensions.

    OpenAI has limits on image size. This function resizes while maintaining aspect ratio.

    Args:
        image_source: Image file path, bytes, or base64 string
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels

    Returns:
        bytes: Resized image as JPEG bytes
    """
    # Load image
    if isinstance(image_source, str) and not Path(image_source).exists():
        # Assume base64
        image_bytes = base64.b64decode(image_source)
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_source, (str, Path)):
        image = Image.open(image_source)
    else:
        image = Image.open(io.BytesIO(image_source))

    # Check if resize needed
    if image.width <= max_width and image.height <= max_height:
        # No resize needed, return original bytes
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()

    # Calculate new dimensions maintaining aspect ratio
    ratio = min(max_width / image.width, max_height / image.height)
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)

    # Resize
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to bytes
    output = io.BytesIO()
    resized.save(output, format="JPEG", quality=85)
    return output.getvalue()


def validate_image(image_source: Union[str, Path, bytes]) -> bool:
    """
    Validate that the image can be opened and is a valid format.

    Args:
        image_source: Image file path, bytes, or base64 string

    Returns:
        bool: True if valid image
    """
    try:
        if isinstance(image_source, str) and not Path(image_source).exists():
            # Assume base64
            image_bytes = base64.b64decode(image_source)
            Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_source, (str, Path)):
            Image.open(image_source)
        else:
            Image.open(io.BytesIO(image_source))
        return True
    except Exception:
        return False


def prepare_image_for_api(image_source: Union[str, Path, bytes]) -> str:
    """
    Prepare an image for OpenAI Vision API.

    This combines validation, resizing, and base64 encoding.

    Args:
        image_source: Image file path, bytes, or base64 string

    Returns:
        str: Base64-encoded, resized image ready for API

    Raises:
        ValueError: If image is invalid
    """
    if not validate_image(image_source):
        raise ValueError("Invalid image format or corrupted image")

    # Resize if needed
    resized_bytes = resize_image_if_needed(image_source)

    # Convert to base64
    return base64.b64encode(resized_bytes).decode("utf-8")
