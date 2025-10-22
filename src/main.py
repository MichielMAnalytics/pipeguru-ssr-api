"""Main FastAPI application."""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import analyze, health, rate

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PipeGuru SSR API",
    description="Analyze ad creatives using personas and Semantic-Similarity Rating (SSR)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/v1", tags=["health"])
app.include_router(rate.router, prefix="/v1", tags=["rating"])
app.include_router(analyze.router, prefix="/v1", tags=["analysis"])


@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    logger.info("🚀 Starting pipeguru-ssr-api v1.0.0")

    # Check Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("⚠️  GEMINI_API_KEY not set - predictions will fail!")
    else:
        logger.info("✓ Gemini API key configured")

    logger.info("✓ Creative analysis endpoint enabled")
    logger.info("✓ Multiple reference sets enabled")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PipeGuru SSR API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/v1/analyze-creative",
            "health": "/v1/health",
            "rate": "/v1/rate"
        },
        "docs": "/docs",
    }
