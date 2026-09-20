"""
Embeddings Module
=================
Loads the sentence-transformers model and generates 384-dim
embedding vectors from text.

Model: sentence-transformers/all-MiniLM-L6-v2
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Singleton model (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None


def _load_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' …", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embedding(text: str) -> list[float]:
    """
    Generate a normalised 384-dim embedding vector for the given text.

    Parameters
    ----------
    text : str
        The input text to embed.

    Returns
    -------
    list[float]
        A list of 384 floats representing the embedding vector.
    """
    model = _load_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
