# PipeGuru SSR API

FastAPI service for analyzing ad creatives using synthetic personas and Semantic Similarity Rating (SSR).

## What is this?

This API analyzes ad creatives (images and videos) by evaluating them through the perspective of specific personas, returning both:
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

Analyze an ad creative (image or video) with specific personas. Returns qualitative feedback and quantitative scores per persona, plus aggregate metrics.

**Supported Formats:**
- **Images**: JPEG, PNG, GIF, WebP
- **Videos**: MP4, WebM, MOV, AVI, MPEG, FLV, WMV, 3GPP

**Request:**
```python
{
  "creative_base64": "...",  # Base64-encoded image or video
  "personas": [              # Your persona descriptions
    "You are Sarah, a 35-year-old marketing manager earning $85k/year...",
    "You are Mike, a 28-year-old software engineer earning $110k/year..."
  ],
  "mime_type": "video/mp4",  # Optional - auto-detects if not provided

  # Optional: Brand familiarity parameters
  "brand_context": "...",    # Comprehensive brand information
  "brand_familiarity_distribution": "emerging_brand",  # Preset or custom dict
  "brand_familiarity_seed": None  # None = deterministic, int = random with seed
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
      "rating_certainty": 0.85,
      "brand_familiarity_level": 3  # Only present if brand familiarity enabled
    }
  ],
  "aggregate": {
    "average_score": 3.8,
    "predicted_purchase_intent": 0.62,
    "pmf_aggregate": [0.08, 0.12, 0.18, 0.35, 0.27],
    "persona_agreement": 0.82,
    "qualitative_summary": "Most personas found the product appealing due to its value proposition and design. Key strengths include clear messaging and attractive pricing. Some concerns were raised about shipping costs and delivery timeframes."
  },
  "metadata": {
    "num_personas": 10,
    "llm_calls": 11,
    "llm_model": "gemini-2.5-flash",
    "media_type": "image",
    "mime_type": "image/jpeg",
    # Brand familiarity metadata (only present if enabled)
    "brand_familiarity": {
      "enabled": True,
      "distribution_used": "emerging_brand",
      "seed": None,
      "level_distribution": {
        "level_1": {"count": 4, "percentage": 40, "label": "Never heard"},
        "level_2": {"count": 3, "percentage": 30, "label": "Vaguely aware"},
        "level_3": {"count": 2, "percentage": 20, "label": "Familiar"},
        "level_4": {"count": 1, "percentage": 8, "label": "Very familiar"},
        "level_5": {"count": 0, "percentage": 2, "label": "Brand advocate"}
      }
    }
  }
}
```

**Note:** For videos, keep file size under 10MB for optimal performance.

### Brand Familiarity Feature

The brand familiarity feature allows you to test how ad performance varies based on consumers' prior knowledge of your brand. This is critical for understanding whether your creative works for first-time viewers vs. brand loyalists.

#### Overview

When analyzing ads, you can optionally specify:
1. **Brand context** - Comprehensive information about your brand (identity, values, products, etc.)
2. **Brand familiarity distribution** - How familiar your target personas are with the brand
3. **Assignment mode** - Deterministic (exact percentages) or random (with seed for reproducibility)

The system then:
- Assigns familiarity levels (1-5) to each persona based on the distribution
- Generates appropriate context instructions per level to prevent data leakage
- Evaluates the ad with each persona's knowledge level
- Returns familiarity levels and distribution statistics in the response

#### Familiarity Levels

| Level | Label | Description |
|-------|-------|-------------|
| 1 | Never heard | Zero knowledge, complete first-time exposure |
| 2 | Vaguely aware | Seen once or twice, superficial recognition only |
| 3 | Familiar | Knows what they do, general understanding of positioning |
| 4 | Very familiar | Has engaged with brand, purchased before, clear opinions |
| 5 | Brand advocate | Deeply loyal, extensive knowledge, emotional connection |

#### Preset Distributions

| Preset | Description | Distribution |
|--------|-------------|--------------|
| `uniform` | Equal distribution across all levels | 20% / 20% / 20% / 20% / 20% |
| `new_brand` | Brand just launched | 70% / 20% / 8% / 2% / 0% |
| `emerging_brand` | Growing brand awareness | 40% / 30% / 20% / 8% / 2% |
| `established_brand` | Well-known in market | 10% / 20% / 40% / 20% / 10% |
| `popular_brand` | High brand recognition | 5% / 15% / 30% / 35% / 15% |
| `cult_brand` | Niche with passionate fans | 50% / 20% / 10% / 10% / 10% |

#### Request Parameters

```python
{
  # Required for brand familiarity
  "brand_context": str,  # Max 5000 chars - comprehensive brand information

  # Optional - controls distribution of familiarity levels
  "brand_familiarity_distribution": str | dict,  # Preset name or custom dict

  # Optional - controls assignment mode
  "brand_familiarity_seed": int | None  # None = deterministic, int = random
}
```

**Parameter Details:**

- **brand_context** (required if using brand familiarity):
  - Comprehensive brand information including identity, values, products, differentiators, target audience
  - Used by LLM to generate appropriate context per familiarity level
  - Example: See `scripts/test_analyze_creative.py` for full Upfront brand context

- **brand_familiarity_distribution** (optional, default: uniform):
  - **Preset string**: One of `uniform`, `new_brand`, `emerging_brand`, `established_brand`, `popular_brand`, `cult_brand`
  - **Custom dict**: `{1: 0.4, 2: 0.3, 3: 0.2, 4: 0.08, 5: 0.02}` (must sum to 1.0)

- **brand_familiarity_seed** (optional, default: None):
  - **None (default)**: Deterministic mode - assigns exact percentages (e.g., 5 personas with 20% each = exactly [1, 2, 3, 4, 5])
  - **Integer**: Random mode with seed for reproducibility (e.g., seed=42 always gives same random assignment)

#### Example Usage

**Example 1: Emerging Brand with Deterministic Assignment**

```python
import requests
import base64

with open("ad.jpg", "rb") as f:
    creative_b64 = base64.b64encode(f.read()).decode()

brand_context = """Upfront is a Dutch sports nutrition company founded in January 2020.
The brand's mission is to establish a new standard for sports nutrition with transparency
and honesty at its core. Their primary slogan is "Wat oprecht is wint" (What is genuine wins).

Key Differentiators:
- Radical Transparency: All ingredients displayed on packaging front
- No artificial flavors, colors, or sweeteners
- Minimalist design that stands out by being deliberately simple

Products: protein powders, bars, shakes, electrolytes, energy gels
Distribution: Direct-to-consumer online + 502 Albert Heijn stores
Target: Health-conscious consumers, athletes, fitness enthusiasts"""

personas = [
    "You are Sarah, a 28-year-old fitness enthusiast...",
    "You are Mike, a 35-year-old marathon runner...",
    "You are Elena, a 31-year-old yoga instructor...",
    "You are Jordan, a 26-year-old CrossFit athlete...",
    "You are Alex, a 33-year-old triathlete..."
]

response = requests.post(
    "http://localhost:8000/v1/analyze-creative",
    json={
        "creative_base64": creative_b64,
        "personas": personas,
        "brand_context": brand_context,
        "brand_familiarity_distribution": "emerging_brand",
        # No seed = deterministic (exact percentages)
    }
)

result = response.json()

# Check brand familiarity distribution
print("Brand Familiarity Distribution:")
for level_key, level_data in result["metadata"]["brand_familiarity"]["level_distribution"].items():
    print(f"  {level_data['label']}: {level_data['count']} personas ({level_data['percentage']}%)")

# Check individual persona familiarity levels
for persona_result in result["persona_results"]:
    print(f"Persona {persona_result['persona_id']}: Level {persona_result['brand_familiarity_level']}")
```

**Example 2: Custom Distribution with Random Assignment**

```python
# Test with 10 personas - custom distribution favoring low familiarity
custom_distribution = {
    1: 0.5,   # 50% never heard
    2: 0.3,   # 30% vaguely aware
    3: 0.15,  # 15% familiar
    4: 0.04,  # 4% very familiar
    5: 0.01   # 1% brand advocate
}

personas = [f"Persona {i}" for i in range(10)]

response = requests.post(
    "http://localhost:8000/v1/analyze-creative",
    json={
        "creative_base64": creative_b64,
        "personas": personas,
        "brand_context": brand_context,
        "brand_familiarity_distribution": custom_distribution,
        "brand_familiarity_seed": 42  # Reproducible random assignment
    }
)
```

**Example 3: Uniform Distribution (Test All Levels Equally)**

```python
# 5 personas with uniform distribution = exactly one at each level
personas = [f"Persona {i}" for i in range(5)]

response = requests.post(
    "http://localhost:8000/v1/analyze-creative",
    json={
        "creative_base64": creative_b64,
        "personas": personas,
        "brand_context": brand_context,
        "brand_familiarity_distribution": "uniform"
        # No seed = deterministic = [1, 2, 3, 4, 5]
    }
)
```

#### Data Leakage Prevention

The system prevents data leakage by filtering brand context based on familiarity level:

- **Level 1**: No brand information provided (hardcoded)
- **Level 2**: Only 1-2 surface-level facts (e.g., "seen at stores", "know they sell X")
- **Level 3**: Basic brand identity, positioning (~20-30% of context)
- **Level 4**: Detailed knowledge, personal experience (~60-70% of context)
- **Level 5**: Full context, emotionally invested (100% of context)

This ensures lower familiarity personas don't "leak" knowledge they wouldn't have in real life.

#### Response Format Changes

When brand familiarity is enabled:

1. **Persona results** include `brand_familiarity_level` field (1-5)
2. **Metadata** includes `brand_familiarity` object with:
   - `enabled`: True
   - `distribution_used`: Preset name or "custom"
   - `seed`: Seed value or None
   - `level_distribution`: Detailed breakdown of assignments per level

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

# Define personas (your target audience)
personas = [
    """You are Sarah, a 35-year-old marketing manager earning $85k/year in Seattle.
    You value quality and sustainability, shop online weekly, and have moderate price sensitivity.""",

    """You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco.
    You prioritize convenience, are tech-savvy, and have low price sensitivity.""",

    """You are Elena, a 42-year-old small business owner earning $65k/year in Austin.
    You're budget-conscious, shop monthly, and carefully research before purchasing."""
]

# Example 1: Analyze an image
with open("ad.jpg", "rb") as f:
    creative_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/analyze-creative",
    json={
        "creative_base64": creative_b64,
        "personas": personas
    }
)

result = response.json()

# Example 2: Analyze a video
with open("ad.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/analyze-creative",
    json={
        "creative_base64": video_b64,
        "personas": personas,
        "mime_type": "video/mp4"  # Optional - auto-detects if not provided
    }
)

result = response.json()

# Per-persona results
print("Individual Persona Feedback:\n")
for persona_result in result["persona_results"]:
    print(f"Persona {persona_result['persona_id']}:")
    print(f"  Qualitative: {persona_result['qualitative_feedback'][:100]}...")
    print(f"  Score: {persona_result['quantitative_score']}/5")
    print(f"  Rating Certainty: {persona_result['rating_certainty']:.1%}\n")

# Aggregate metrics
print("\nAggregate Results:")
print(f"  Average Score: {result['aggregate']['average_score']}/5")
print(f"  Predicted Purchase Intent: {result['aggregate']['predicted_purchase_intent']:.1%}")
print(f"  Persona Agreement: {result['aggregate']['persona_agreement']:.1%}")
print(f"  Qualitative Summary: {result['aggregate']['qualitative_summary']}")
print(f"\nMedia Type: {result['metadata']['media_type']}")
print(f"MIME Type: {result['metadata']['mime_type']}")
print(f"LLM Calls: {result['metadata']['llm_calls']}")
print(f"Model: {result['metadata']['llm_model']}")
```

### cURL

```bash
# Analyze an image
IMAGE_B64=$(base64 -i ad.jpg)

curl -X POST http://localhost:8000/v1/analyze-creative \
  -H "Content-Type: application/json" \
  -d "{
    \"creative_base64\": \"$IMAGE_B64\",
    \"personas\": [
      \"You are Sarah, a 35-year-old marketing manager earning \$85k/year in Seattle. You value quality and sustainability.\",
      \"You are Mike, a 28-year-old software engineer earning \$110k/year in San Francisco. You prioritize convenience.\"
    ]
  }"

# Analyze a video
VIDEO_B64=$(base64 -i ad.mp4)

curl -X POST http://localhost:8000/v1/analyze-creative \
  -H "Content-Type: application/json" \
  -d "{
    \"creative_base64\": \"$VIDEO_B64\",
    \"personas\": [
      \"You are Sarah, a 35-year-old marketing manager earning \$85k/year in Seattle.\"
    ],
    \"mime_type\": \"video/mp4\"
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
