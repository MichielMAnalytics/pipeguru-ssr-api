"""Health check endpoint."""

from fastapi import APIRouter

from src.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current status and version of the API.
    """
    return HealthResponse(status="healthy", version="0.1.0")
