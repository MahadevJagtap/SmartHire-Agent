"""
Memory Retrieve Module
======================
Retrieves relevant user memories from PostgreSQL using
pgvector cosine-distance similarity search.
"""

import os
import logging

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from embeddings import generate_embedding

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database engine (shared singleton)
# ---------------------------------------------------------------------------
_engine = None


def _get_engine():
    """Return a SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Please add it to your .env file."
            )
        _engine = create_engine(database_url)
    return _engine


def retrieve_memories(
    user_id: str, query: str, k: int = 5
) -> list[dict]:
    """
    Retrieve the top-*k* most relevant memories for a user.

    Steps:
        1. Generate an embedding for the query.
        2. Perform a vector similarity search using cosine distance.
        3. Return the top-k results.

    Parameters
    ----------
    user_id : str
        The user whose memories to search.
    query : str
        The query text to match against stored memories.
    k : int
        Maximum number of results to return (default 5).

    Returns
    -------
    list[dict]
        Each dict has keys: ``id``, ``content``, ``score``, ``created_at``.
    """
    query_embedding = generate_embedding(query)

    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT id, content, created_at,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM   memories
                WHERE  user_id = :user_id
                ORDER  BY embedding <=> CAST(:embedding AS vector)
                LIMIT  :k;
                """
            ),
            {
                "user_id": user_id,
                "embedding": str(query_embedding),
                "k": k,
            },
        )
        rows = result.fetchall()

    memories = [
        {
            "id": row[0],
            "content": row[1],
            "created_at": row[2],
            "score": float(row[3]) if row[3] is not None else 0.0,
        }
        for row in rows
    ]

    logger.info(
        "Retrieved %d memories for user '%s' (query: %.40s…)",
        len(memories),
        user_id,
        query,
    )
    return memories


def get_all_memories(user_id: str) -> list[dict]:
    """
    Return *all* memories for a user, newest first.

    Parameters
    ----------
    user_id : str
        The user whose memories to return.

    Returns
    -------
    list[dict]
        Each dict has keys: ``id``, ``content``, ``created_at``.
    """
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT id, content, created_at
                FROM   memories
                WHERE  user_id = :user_id
                ORDER  BY created_at DESC;
                """
            ),
            {"user_id": user_id},
        )
        rows = result.fetchall()

    return [
        {"id": row[0], "content": row[1], "created_at": row[2]}
        for row in rows
    ]
