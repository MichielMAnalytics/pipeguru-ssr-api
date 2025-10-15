# PipeGuru SSR API

FastAPI wrapper for the [semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating) package.

## What is this?

This is a RESTful API that wraps the Semantic-Similarity Rating (SSR) methodology, providing HTTP endpoints for converting LLM text responses into probability distributions over Likert scales.

Part of the [PipeGuru v3](https://github.com/MichielMAnalytics/pipeguru-ssr-api) architecture - an ad performance prediction platform for e-commerce brands.

**Note**: This package depends on [semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating) which is installed directly from GitHub (not yet on PyPI).

## Quick Start

### Using Docker

```bash
docker-compose up
```

API available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### Local Development

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn src.main:app --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Installation from Source

```bash
# Clone repository
git clone https://github.com/MichielMAnalytics/pipeguru-ssr-api.git
cd pipeguru-ssr-api

# Install
pip install -r requirements.txt
```

## API Endpoints

### `POST /v1/rate`

Convert text responses to probability distributions.

**Request:**
```json
{
  "responses": [
    "I really like this product",
    "Not sure about this"
  ],
  "reference_sentences": [
    "I definitely would not purchase",
    "I probably would not purchase",
    "I might or might not purchase",
    "I probably would purchase",
    "I definitely would purchase"
  ],
  "temperature": 1.0,
  "epsilon": 0.01
}
```

**Response:**
```json
{
  "success": true,
  "ratings": [
    {
      "response": "I really like this product",
      "pmf": [0.05, 0.10, 0.15, 0.30, 0.40],
      "expected_value": 3.90,
      "most_likely_rating": 5,
      "confidence": 0.40
    }
  ],
  "metadata": {
    "num_responses": 2,
    "num_reference_sentences": 5
  }
}
```

### `GET /v1/health`

Health check endpoint.

## Documentation

- [API Reference](docs/api-reference.md)
- [Self-Hosting Guide](docs/self-hosting.md)
- [Quick Start Tutorial](docs/quickstart.md)

## Architecture

This API is Layer 2 in the PipeGuru v3 stack:

```
┌─────────────────────────────────┐
│  SaaS Dashboard (Proprietary)   │  ← Full platform
├─────────────────────────────────┤
│  pipeguru-ssr-api (This Repo)   │  ← FastAPI wrapper (open source)
├─────────────────────────────────┤
│  semantic-similarity-rating     │  ← Core SSR methodology (MIT)
└─────────────────────────────────┘
```

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! Please open an issue or PR.

## Credits

Built on the [semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating) package by PyMC Labs.

Based on research: "Measuring Synthetic Consumer Purchase Intent Using Semantic-Similarity Ratings" (2025)
