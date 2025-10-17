"""Gemini embeddings helper for SSR."""

import os
from typing import List

import numpy as np
from google import genai
from google.genai import types


class GeminiEmbeddings:
    """Helper class for generating embeddings using Gemini."""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini embeddings client.

        Args:
            api_key: Google API key. If not provided, reads from GEMINI_API_KEY env var.
            model: Embedding model to use. Defaults to gemini-embedding-001.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor")

        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        self.client = genai.Client(api_key=self.api_key)

        # Use 768 dimensions to match all-MiniLM-L6-v2 (which has 384 dims)
        # Using 768 as a good balance between quality and cost
        self.output_dimensionality = 768

        print(f"[Embeddings] Initialized Gemini embeddings with model: {self.model}, dimensions: {self.output_dimensionality}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy.ndarray: Matrix of embeddings, shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        print(f"[Embeddings] Generating embeddings for {len(texts)} texts...")

        try:
            # Generate embeddings using Gemini
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=self.output_dimensionality,
                )
            )

            # Extract embeddings from result
            embeddings = np.array([embedding.values for embedding in result.embeddings])

            # Normalize embeddings (required when using custom dimensions)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            print(f"[Embeddings] Generated embeddings with shape: {embeddings.shape}")

            return embeddings

        except Exception as e:
            print(f"[Embeddings] Error generating embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error generating Gemini embeddings: {str(e)}")
