"""
API v1 endpoints.

Main endpoints:
- /analyze-creative: Analyze ad creatives with personas (supports brand familiarity)
- /personas/generate: Generate synthetic personas for testing

Additional endpoints:
- /rate: Core SSR rating endpoint
- /health: Health check
"""

# Export routers for main app
from . import analyze, health, rate

__all__ = ["analyze", "health", "rate"]
