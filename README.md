# PipeGuru SSR API

FastAPI wrapper for the [semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating) package.

## What is this?

This is a RESTful API that wraps the Semantic-Similarity Rating (SSR) methodology, providing HTTP endpoints for converting LLM text responses into probability distributions over Likert scales.

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

### Core SSR Endpoints

#### `POST /v1/rate`

Convert text responses to probability distributions using SSR.

**Example:**
```bash
curl -X POST http://localhost:8000/v1/rate \
  -H "Content-Type: application/json" \
  -d '{
    "responses": ["I really like this product"],
    "reference_sentences": [
      "I definitely would not purchase",
      "I probably would not purchase",
      "I might or might not purchase",
      "I probably would purchase",
      "I definitely would purchase"
    ]
  }'
```

**Returns:** PMF distribution, expected value, most likely rating, confidence

#### `GET /v1/health`

Health check endpoint.

### Ad Prediction Endpoints (NEW!)

#### `POST /v1/predict-ad`

Predict ad performance using synthetic personas and GPT-4 Vision.

**What it does:**
1. Generates N synthetic customer personas
2. Shows ad to each persona using GPT-4 Vision
3. Converts responses to probability distributions via SSR
4. Aggregates into final prediction

**Example:**
```bash
# Encode image
IMAGE_B64=$(base64 -i ad.jpg)

curl -X POST http://localhost:8000/v1/predict-ad \
  -H "Content-Type: application/json" \
  -d "{
    \"ad_image_base64\": \"$IMAGE_B64\",
    \"num_personas\": 10,
    \"segment\": \"millennial_women\"
  }"
```

**Returns:** Predicted conversion rate, confidence, PMF, individual persona results, cost estimate

**Cost:** ~$0.01-0.02 per persona (~$0.20-0.40 for 20 personas)

#### `POST /v1/generate-personas`

Generate synthetic customer personas (for testing/debugging).

**Example:**
```bash
curl -X POST http://localhost:8000/v1/generate-personas \
  -H "Content-Type: application/json" \
  -d '{"num_personas": 3, "segment": "millennial_women"}'
```

#### `GET /v1/personas/segments`

List available persona segments.

**Segments:** `general_consumer`, `millennial_women`, `gen_z`

## Documentation

- [API Reference](docs/api-reference.md)
- [Self-Hosting Guide](docs/self-hosting.md)
- [Quick Start Tutorial](docs/quickstart.md)

## Architecture

```
┌─────────────────────────────────┐
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
