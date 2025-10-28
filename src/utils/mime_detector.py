"""MIME type detection for creative media (images and videos)."""

import base64
from typing import Literal, Tuple


# Supported MIME types for Gemini API
SUPPORTED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

SUPPORTED_VIDEO_MIMES = {
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/quicktime",  # Alternative for .mov
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/x-msvideo",  # Alternative for .avi
    "video/x-ms-wmv",   # WMV
    "video/3gpp",
}

ALL_SUPPORTED_MIMES = SUPPORTED_IMAGE_MIMES | SUPPORTED_VIDEO_MIMES


# Magic bytes for file format detection
MAGIC_BYTES = {
    # Images
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # Needs additional check for WEBP

    # Videos
    b"\x00\x00\x00\x14ftypisom": "video/mp4",
    b"\x00\x00\x00\x18ftypmp42": "video/mp4",
    b"\x00\x00\x00\x1cftypisom": "video/mp4",
    b"\x00\x00\x00\x20ftypisom": "video/mp4",
    b"ftyp": "video/mp4",  # Generic MP4 (check at offset 4)
    b"\x00\x00\x00\x14ftypqt": "video/quicktime",
    b"\x1aE\xdf\xa3": "video/webm",  # WebM/Matroska
    b"RIFF": "video/avi",  # Needs additional check for AVI
    b"\x00\x00\x01\xba": "video/mpeg",
    b"\x00\x00\x01\xb3": "video/mpeg",
    b"FLV": "video/x-flv",
    b"\x30\x26\xb2\x75": "video/x-ms-wmv",
    b"\x00\x00\x00\x14ftyp3gp": "video/3gpp",
}


MediaCategory = Literal["image", "video"]


def detect_mime_type_from_base64(base64_data: str) -> Tuple[str, MediaCategory]:
    """
    Detect MIME type and media category from base64-encoded data.

    Args:
        base64_data: Base64-encoded media data (image or video)

    Returns:
        Tuple of (mime_type, media_category)

    Raises:
        ValueError: If format cannot be detected or is unsupported

    Example:
        >>> mime, category = detect_mime_type_from_base64(image_b64)
        >>> print(mime, category)
        ('image/jpeg', 'image')
    """
    try:
        # Decode base64 to bytes
        data = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {str(e)}")

    if len(data) < 12:
        raise ValueError("File data too short to determine format")

    # Check magic bytes
    mime_type = _detect_from_magic_bytes(data)

    if not mime_type:
        raise ValueError(
            "Could not detect file format. Supported formats: "
            f"Images (JPEG, PNG, GIF, WebP), Videos (MP4, WebM, MOV, AVI, MPEG, FLV, WMV, 3GPP)"
        )

    # Verify it's supported
    if mime_type not in ALL_SUPPORTED_MIMES:
        raise ValueError(
            f"Unsupported MIME type: {mime_type}. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED_MIMES))}"
        )

    # Determine category
    if mime_type in SUPPORTED_IMAGE_MIMES:
        category: MediaCategory = "image"
    else:
        category = "video"

    return mime_type, category


def _detect_from_magic_bytes(data: bytes) -> str | None:
    """
    Detect MIME type from file magic bytes.

    Args:
        data: Raw file bytes

    Returns:
        MIME type string or None if not detected
    """
    # Check for exact matches first
    for magic, mime in MAGIC_BYTES.items():
        if data.startswith(magic):
            # Special handling for RIFF (could be WebP or AVI)
            if magic == b"RIFF" and len(data) > 12:
                # Check WEBP signature
                if data[8:12] == b"WEBP":
                    return "image/webp"
                # Check AVI signature
                elif data[8:12] == b"AVI ":
                    return "video/avi"
            else:
                return mime

    # Check for MP4 variants (ftyp at offset 4)
    if len(data) > 12 and data[4:8] == b"ftyp":
        ftyp_brand = data[8:12]

        # Common MP4 brands
        if ftyp_brand in [b"isom", b"iso2", b"mp41", b"mp42", b"avc1", b"M4V ", b"M4A "]:
            return "video/mp4"
        # QuickTime
        elif ftyp_brand in [b"qt  ", b"qtif"]:
            return "video/quicktime"
        # 3GPP
        elif ftyp_brand in [b"3gp4", b"3gp5", b"3gp6", b"3g2a"]:
            return "video/3gpp"
        else:
            # Unknown ftyp brand, default to mp4
            return "video/mp4"

    return None


def validate_mime_type(mime_type: str) -> Tuple[str, MediaCategory]:
    """
    Validate and normalize a MIME type string.

    Args:
        mime_type: MIME type string to validate

    Returns:
        Tuple of (normalized_mime_type, media_category)

    Raises:
        ValueError: If MIME type is not supported

    Example:
        >>> mime, category = validate_mime_type("video/mp4")
        >>> print(mime, category)
        ('video/mp4', 'video')
    """
    # Normalize to lowercase
    mime_normalized = mime_type.lower().strip()

    # Handle common aliases
    mime_aliases = {
        "image/jpg": "image/jpeg",
        "video/quicktime": "video/mov",
        "video/x-msvideo": "video/avi",
    }
    mime_normalized = mime_aliases.get(mime_normalized, mime_normalized)

    # Check if supported
    if mime_normalized not in ALL_SUPPORTED_MIMES:
        raise ValueError(
            f"Unsupported MIME type: {mime_type}. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED_MIMES))}"
        )

    # Determine category
    if mime_normalized in SUPPORTED_IMAGE_MIMES:
        category: MediaCategory = "image"
    else:
        category = "video"

    return mime_normalized, category


def get_media_type_from_mime(mime_type: str) -> MediaCategory:
    """
    Get media category (image or video) from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        'image' or 'video'

    Example:
        >>> get_media_type_from_mime("video/mp4")
        'video'
    """
    if mime_type in SUPPORTED_IMAGE_MIMES:
        return "image"
    elif mime_type in SUPPORTED_VIDEO_MIMES:
        return "video"
    else:
        raise ValueError(f"Unknown MIME type: {mime_type}")
