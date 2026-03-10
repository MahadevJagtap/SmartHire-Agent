"""
Test script for the modular Long-Term Memory system.
=====================================================
Verifies:
  1. Embedding generation (embeddings.py)
  2. Memory storage (memory_store.py)
  3. Semantic retrieval (memory_retrieve.py)
  4. High-level API (memory_manager.py)

Usage:
    python test_memory_system.py
"""

import sys
import os

# Fix Windows console encoding for emoji/Unicode output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()


def test_embeddings():
    """Test that embedding generation works and produces the right shape."""
    print("=" * 60)
    print("TEST 1: Embedding Generation")
    print("=" * 60)
    from embeddings import generate_embedding, EMBEDDING_DIM

    text = "Hello, my name is Mahadev."
    embedding = generate_embedding(text)

    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dims, got {len(embedding)}"
    assert all(isinstance(x, float) for x in embedding), "All values should be floats"

    print(f"  ✅ Generated embedding with {len(embedding)} dimensions")
    print(f"  ✅ First 5 values: {embedding[:5]}")
    print()


def test_store_memory():
    """Test storing a memory in the database."""
    print("=" * 60)
    print("TEST 2: Memory Storage")
    print("=" * 60)
    from memory_store import store_memory

    test_user = "__test_user__"
    row_id = store_memory(test_user, "User name is TestUser")
    assert isinstance(row_id, int), "Returned id should be an int"
    assert row_id > 0, "Row id should be positive"

    print(f"  ✅ Stored memory with id={row_id}")
    print()
    return row_id


def test_retrieve_memories():
    """Test semantic search retrieval."""
    print("=" * 60)
    print("TEST 3: Semantic Retrieval")
    print("=" * 60)
    from memory_retrieve import retrieve_memories

    test_user = "__test_user__"
    results = retrieve_memories(test_user, "What is the user's name?", k=3)

    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Should find at least one memory"

    top = results[0]
    assert "content" in top, "Result should have 'content'"
    assert "score" in top, "Result should have 'score'"
    assert top["score"] > 0.3, f"Score should be reasonable, got {top['score']}"

    print(f"  ✅ Retrieved {len(results)} memories")
    for r in results:
        print(f"     • [{r['score']:.3f}] {r['content']}")
    print()


def test_get_all_memories():
    """Test listing all memories for a user."""
    print("=" * 60)
    print("TEST 4: Get All Memories")
    print("=" * 60)
    from memory_retrieve import get_all_memories

    test_user = "__test_user__"
    memories = get_all_memories(test_user)

    assert isinstance(memories, list), "Should return a list"
    assert len(memories) > 0, "Should have at least one memory"

    print(f"  ✅ Found {len(memories)} total memories for '{test_user}'")
    print()


def test_high_level_api():
    """Test the memory_manager high-level functions."""
    print("=" * 60)
    print("TEST 5: High-Level API (memory_manager)")
    print("=" * 60)
    from memory_manager import add_memory, get_relevant_memories, get_user_memories

    test_user = "__test_user__"
    row_id = add_memory(test_user, "User builds AI agents")
    assert row_id > 0

    results = get_relevant_memories(test_user, "What does the user work on?")
    assert len(results) > 0
    assert any("AI agents" in r["content"] for r in results)

    all_mems = get_user_memories(test_user)
    assert len(all_mems) >= 2  # at least the two we stored

    print(f"  ✅ add_memory returned id={row_id}")
    print(f"  ✅ get_relevant_memories found {len(results)} results")
    print(f"  ✅ get_user_memories found {len(all_mems)} total memories")
    print()


def cleanup(test_user="__test_user__"):
    """Remove test rows from the database."""
    print("=" * 60)
    print("CLEANUP")
    print("=" * 60)
    from sqlalchemy import create_engine, text

    database_url = os.getenv("DATABASE_URL")
    engine = create_engine(database_url)
    with engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM memories WHERE user_id = :uid"),
            {"uid": test_user},
        )
        conn.commit()
    print(f"  🧹 Deleted test rows for user '{test_user}'")
    print()


def main():
    print("\n🧪 Long-Term Memory System — Integration Tests\n")

    try:
        test_embeddings()
        test_store_memory()
        test_retrieve_memories()
        test_get_all_memories()
        test_high_level_api()
        print("🎉 ALL TESTS PASSED!\n")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
