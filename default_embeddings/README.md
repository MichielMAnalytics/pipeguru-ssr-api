# Default Reference Embeddings

This directory contains pre-computed embeddings for reference statement sets used in Semantic Similarity Rating (SSR).

## Why Pre-compute?

Reference statements are **static and never change**, so their embeddings should be:
1. Pre-computed once and stored on disk
2. Loaded at runtime instead of regenerated
3. Version-controlled for reproducibility

## Performance Impact

**Without caching (1000 personas):**
- 1000 LLM calls (personas)
- 6 reference sets × 5 statements = 30 embedding API calls **per request**
- Total: 1030 API calls

**With caching (1000 personas):**
- 1000 LLM calls (personas)
- 0 reference embedding calls (loaded from disk)
- Total: 1000 API calls
- **Time saved: ~10 seconds per request**

## Files

Each embedding file follows the naming convention:
```
{model_name}_{dimensions}d_purchase_intent_set_{index}.npy
```

Example:
```
models_text-embedding-004_768d_purchase_intent_set_0.npy
models_text-embedding-004_768d_purchase_intent_set_1.npy
...
models_text-embedding-004_768d_purchase_intent_set_5.npy
```

## Regenerating Embeddings

If you change the embedding model or reference statements, regenerate embeddings:

```bash
python scripts/generate_reference_embeddings.py
```

This will:
1. Load all reference statement sets
2. Generate embeddings using the configured Gemini model
3. Save embeddings as `.npy` files in this directory
4. Create a manifest file with metadata

## Manifest File

`manifest.json` contains metadata about the embeddings:
```json
{
  "model": "models/text-embedding-004",
  "dimensions": 768,
  "question_type": "purchase_intent",
  "num_sets": 6,
  "generated_at": "2025-10-27T20:00:00Z",
  "gemini_api_version": "1.0.0"
}
```
