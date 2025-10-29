"""API Key authentication middleware for pipeguru-ssr-api."""

import logging
import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# API Key header configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def get_api_key() -> str:
    """Get the expected API key from environment variables.

    Returns:
        str: The API key from SSR_API_KEY environment variable

    Raises:
        RuntimeError: If SSR_API_KEY is not configured
    """
    api_key = os.getenv("SSR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SSR_API_KEY environment variable is not set. "
            "Please configure it in your .env file."
        )
    return api_key


async def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the provided API key against the expected value.

    This dependency can be used on any FastAPI route to require authentication.

    Args:
        api_key: The API key from the request header

    Returns:
        str: The validated API key

    Raises:
        HTTPException: 401 if the API key is missing or invalid

    Example:
        ```python
        @app.get("/protected", dependencies=[Depends(validate_api_key)])
        async def protected_endpoint():
            return {"message": "Access granted"}
        ```
    """
    try:
        expected_key = get_api_key()
    except RuntimeError as e:
        logger.error(f"API key validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation is not properly configured",
        )

    if api_key != expected_key:
        logger.warning(f"Invalid API key attempt from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key
