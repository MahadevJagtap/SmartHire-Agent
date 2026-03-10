"""
Memory Manager — High-Level API
================================
Provides a clean, high-level interface for the AI agent to interact with
long-term memory.  All heavy lifting is delegated to the sub-modules:

  * ``embeddings.py``   — embedding generation
  * ``memory_store.py`` — writing memories
  * ``memory_retrieve.py`` — reading / searching memories

Usage::

    from memory_manager import add_memory, get_relevant_memories

    add_memory("user1", "User name is Mahadev")
    results = get_relevant_memories("user1", "What is the user's name?")
"""

import os
import logging

from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from embeddings import generate_embedding
from memory_store import store_memory
from memory_retrieve import retrieve_memories, get_all_memories

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM used for automatic memory extraction
# ---------------------------------------------------------------------------
_extraction_llm = ChatGroq(model="llama-3.3-70b-versatile")

EXTRACTION_PROMPT = """Analyze the following user message and extract NEW personal facts about the user.
Personal facts include: name, profession, preferences, interests, projects, habits, location, goals, etc.

Only extract facts that are explicitly stated by the user — do NOT infer or guess.
Return each fact on a SEPARATE LINE as a concise statement (e.g., "User name is Mahadev").
If there are NO new personal facts, return exactly: NONE

User message:
{user_message}

Extracted facts:"""


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def add_memory(user_id: str, text_content: str) -> int:
    """
    Store a new memory for the given user.

    Parameters
    ----------
    user_id : str
        Unique identifier for the user.
    text_content : str
        The memory text to store.

    Returns
    -------
    int
        The id of the newly inserted memory row.
    """
    return store_memory(user_id, text_content)


def get_relevant_memories(user_id: str, query: str, k: int = 5) -> list[dict]:
    """
    Retrieve the top-*k* most relevant memories for a user.

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
    return retrieve_memories(user_id, query, k=k)


def get_user_memories(user_id: str) -> list[dict]:
    """Return all stored memories for a user (for sidebar display)."""
    return get_all_memories(user_id)


# ---------------------------------------------------------------------------
# Automatic memory extraction
# ---------------------------------------------------------------------------

def extract_memories(user_message: str) -> list[str]:
    """
    Use the LLM to extract personal facts from a user message.

    Only facts explicitly stated by the user are extracted.
    Returns a list of concise fact strings, or an empty list
    if no personal facts are found.

    Parameters
    ----------
    user_message : str
        The raw message from the user.

    Returns
    -------
    list[str]
        Concise fact strings, e.g. ["User name is Mahadev", "User builds AI agents"].
    """
    try:
        prompt = EXTRACTION_PROMPT.format(user_message=user_message)
        result = _extraction_llm.invoke([HumanMessage(content=prompt)])
        extracted_text = result.content.strip()

        if extracted_text.upper() == "NONE" or not extracted_text:
            return []

        facts = [
            line.strip().lstrip("•-– ").strip()
            for line in extracted_text.split("\n")
            if line.strip() and line.strip().upper() != "NONE"
        ]

        # Filter out very short / empty facts
        return [f for f in facts if len(f) > 5]

    except Exception as e:
        logger.error("Memory extraction failed: %s", e)
        return []


def auto_store_memories(user_id: str, user_message: str) -> list[str]:
    """
    Extract personal facts from a user message and store any NEW ones.

    Duplicate detection: before storing a fact, a similarity search is
    performed.  If an existing memory scores above 0.85 the fact is
    considered a duplicate and skipped.

    Parameters
    ----------
    user_id : str
        Unique identifier for the user.
    user_message : str
        The raw user message to analyse.

    Returns
    -------
    list[str]
        The facts that were actually stored (excluding duplicates).
    """
    facts = extract_memories(user_message)
    stored: list[str] = []

    for fact in facts:
        # Duplicate check
        existing = retrieve_memories(user_id, fact, k=1)
        if existing and existing[0].get("score", 0) > 0.85:
            logger.info("Skipping duplicate memory: %s", fact)
            continue

        store_memory(user_id, fact)
        logger.info("Auto-stored memory for user '%s': %s", user_id, fact)
        stored.append(fact)

    return stored
