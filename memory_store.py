"""
Memory Store Module
===================
Stores user memories into PostgreSQL with pgvector embeddings.

Uses SQLAlchemy for database operations and the embeddings module
for vector generation.
"""

import os
import logging

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from embeddings import generate_embedding

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database engine (created once, connections are pooled)
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


def store_memory(user_id: str, text_content: str) -> int:
    """
    Store a new memory for the given user.

    Steps:
        1. Generate an embedding for the text.
        2. Insert a row into the ``memories`` table.
        3. Commit the transaction.

    Parameters
    ----------
    user_id : str
        Unique identifier for the user.
    text_content : str
        The memory text to store.

    Returns
    -------
    int
        The id of the inserted memory row.
    """
    embedding = generate_embedding(text_content)

    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO memories (user_id, content, embedding)
                VALUES (:user_id, :content, CAST(:embedding AS vector))
                RETURNING id;
                """
            ),
            {
                "user_id": user_id,
                "content": text_content,
                "embedding": str(embedding),
            },
        )
        row_id = result.fetchone()[0]
        conn.commit()

    logger.info("Stored memory id=%d for user '%s'", row_id, user_id)
    return row_id
