"""
Embedding utilities for semantic similarity calculations.

Supports both OpenAI embeddings and local sentence-transformers models.
"""

import asyncio
from typing import List, Optional
import numpy as np

# Try to import optional dependencies
try:
    from openai import AsyncOpenAI
    import os
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingClient:
    """
    Unified client for generating embeddings from different providers.
    
    Supports:
    - OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
    - Local sentence-transformers models
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding client.
        
        Args:
            provider: "openai" or "local"
            model: Model name
                - For OpenAI: "text-embedding-3-small", "text-embedding-3-large"
                - For local: any sentence-transformers model
            api_key: OpenAI API key (optional, uses env var if not provided)
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI not installed. Install with: pip install openai"
                )
            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        elif provider == "local":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            self.client = SentenceTransformer(model)
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'local'")
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.provider == "openai":
            return await self._embed_openai(texts)
        else:
            return await self._embed_local(texts)
    
    async def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    async def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local sentence-transformers."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.client.encode,
            texts
        )
        return np.array(embeddings)
    
    async def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        embeddings = await self.embed([text1, text2])
        return self.cosine_similarity(embeddings[0], embeddings[1])
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range (can be negative for opposite vectors)
        # For similarity metrics, we typically want 0-1
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))
    
    def __repr__(self) -> str:
        return f"EmbeddingClient(provider='{self.provider}', model='{self.model}')"


# Convenience function for quick similarity calculations
async def calculate_semantic_similarity(
    text1: str,
    text2: str,
    provider: str = "openai",
    model: str = "text-embedding-3-small"
) -> float:
    """
    Quick function to calculate semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        provider: "openai" or "local"
        model: Model name
    
    Returns:
        Similarity score (0.0 to 1.0)
    
    Example:
        >>> similarity = await calculate_semantic_similarity(
        ...     "The cat sat on the mat",
        ...     "A feline rested on the rug"
        ... )
        >>> print(f"Similarity: {similarity:.3f}")
    """
    client = EmbeddingClient(provider=provider, model=model)
    return await client.similarity(text1, text2)
