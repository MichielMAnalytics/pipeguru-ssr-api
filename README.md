# PipeGuru SSR API

FastAPI service for analyzing ad creatives using synthetic personas and Semantic Similarity Rating (SSR).

## What is this?

This API analyzes ad creatives (images) by evaluating them through the perspective of specific personas, returning both:
- **Qualitative feedback**: "I would buy this because..." (LLM reasoning)
- **Quantitative scores**: 1-5 purchase likelihood ratings with confidence

Built on the [semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating) methodology for converting LLM text responses into probability distributions.

## Quick Start

```bash
# Clone repository
git clone https://github.com/MichielMAnalytics/pipeguru-ssr-api.git
cd pipeguru-ssr-api

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Generate reference embeddings (one-time setup)
python scripts/generate_reference_embeddings.py

# Run server
uvicorn src.main:app --reload
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

## API Endpoints

### Main Endpoint

#### `POST /v1/analyze-creative`

Analyze an ad creative with specific personas. Returns qualitative feedback and quantitative scores per persona, plus aggregate metrics.

**Request:**
```python
{
  "creative_base64": "...",  # Base64-encoded image
  "personas": [              # Your persona descriptions
    "You are Sarah, a 35-year-old marketing manager earning $85k/year...",
    "You are Mike, a 28-year-old software engineer earning $110k/year..."
  ]
}
```

**Response:**
```python
{
  "persona_results": [
    {
      "persona_id": 1,
      "persona_description": "You are Sarah...",
      "qualitative_feedback": "I would buy this because it speaks to my need for...",
      "quantitative_score": 4,
      "expected_value": 3.8,
      "pmf": [0.05, 0.10, 0.15, 0.40, 0.30],
      "confidence": 0.85
    }
  ],
  "aggregate": {
    "average_score": 3.8,
    "predicted_conversion_rate": 0.62,
    "pmf_aggregate": [0.08, 0.12, 0.18, 0.35, 0.27],
    "confidence": 0.82
  },
  "metadata": {
    "num_personas": 10,
    "cost_usd": 0.15,
    "processing_time_seconds": 23.5
  }
}
```

**Cost:** ~$0.0015 per persona (~$0.015 for 10 personas)
**Time:** ~10-30 seconds depending on number of personas

### Helper Endpoints

#### `GET /v1/health`

Health check endpoint - returns API status and enabled features.

#### `POST /v1/rate`

Core SSR rating endpoint for advanced use. Convert text responses to probability distributions.

## Example Usage

### Python

```python
import requests
import base64

# Load creative image
with open("ad.jpg", "rb") as f:
    creative_b64 = base64.b64encode(f.read()).decode()

# Define personas (your target audience)
personas = [
    """You are Sarah, a 35-year-old marketing manager earning $85k/year in Seattle.
    You value quality and sustainability, shop online weekly, and have moderate price sensitivity.""",

    """You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco.
    You prioritize convenience, are tech-savvy, and have low price sensitivity.""",

    """You are Elena, a 42-year-old small business owner earning $65k/year in Austin.
    You're budget-conscious, shop monthly, and carefully research before purchasing."""
]

# Analyze creative
response = requests.post(
    "http://localhost:8000/v1/analyze-creative",
    json={
        "creative_base64": creative_b64,
        "personas": personas
    }
)

result = response.json()

# Per-persona results
print("Individual Persona Feedback:\n")
for persona_result in result["persona_results"]:
    print(f"Persona {persona_result['persona_id']}:")
    print(f"  Qualitative: {persona_result['qualitative_feedback'][:100]}...")
    print(f"  Score: {persona_result['quantitative_score']}/5")
    print(f"  Confidence: {persona_result['confidence']:.1%}\n")

# Aggregate metrics
print("\nAggregate Results:")
print(f"  Average Score: {result['aggregate']['average_score']}/5")
print(f"  Predicted Conversion Rate: {result['aggregate']['predicted_conversion_rate']:.1%}")
print(f"  Confidence: {result['aggregate']['confidence']:.1%}")
print(f"\nCost: ${result['metadata']['cost_usd']}")
print(f"Processing Time: {result['metadata']['processing_time_seconds']}s")
```

### cURL

```bash
# Encode image
IMAGE_B64=$(base64 -i ad.jpg)

# Analyze
curl -X POST http://localhost:8000/v1/analyze-creative \
  -H "Content-Type: application/json" \
  -d "{
    \"creative_base64\": \"$IMAGE_B64\",
    \"personas\": [
      \"You are Sarah, a 35-year-old marketing manager earning \$85k/year in Seattle. You value quality and sustainability.\",
      \"You are Mike, a 28-year-old software engineer earning \$110k/year in San Francisco. You prioritize convenience.\"
    ]
  }"
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key |
| `GEMINI_MODEL` | No | gemini-2.5-flash | Gemini model to use |
| `PORT` | No | 8000 | Server port |
| `LOG_LEVEL` | No | INFO | Logging level |

## Documentation

- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Features

### Phase 1 Enhancements (Based on Research Paper)

Implementation validated against the paper: **"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"** (arXiv:2510.08338v2)

#### 🎯 Multiple Reference Sets
- **6 reference statement sets** with semantic variations (default: enabled)
- **PMF averaging** across all sets for stability
- **KS similarity improvement**: ~0.88 vs ~0.72 (single set)
- Configurable via `use_multiple_reference_sets` parameter

#### 👥 Enhanced Personas
Demographics-focused persona generation based on paper findings:
- **Age and income emphasis** (strongest signals for purchase intent)
- **Explicit income brackets** with discretionary spending levels
- **Structured DEMOGRAPHICS section** in all personas
- **3 segments**: general_consumer, millennial_women, gen_z

#### 🚧 Validation & Testing (In Progress)
We're actively working on comprehensive validation metrics and test coverage to ensure prediction accuracy aligns with research benchmarks.

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
