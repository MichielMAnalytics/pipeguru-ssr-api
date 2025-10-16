"""Main FastAPI application."""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import health, personas, predict, rate

app = FastAPI(
    title="PipeGuru SSR API",
    description="FastAPI wrapper for Semantic-Similarity Rating methodology",
    version="0.1.0",
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
app.include_router(personas.router, prefix="/v1", tags=["personas"])
app.include_router(predict.router, prefix="/v1", tags=["prediction"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PipeGuru SSR API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/v1/health",
    }
