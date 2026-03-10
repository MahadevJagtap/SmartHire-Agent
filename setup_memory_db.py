"""
Database Setup Script for Long-Term Memory
===========================================
Run once to create the pgvector extension and memories table.

Usage:
    python setup_memory_db.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    # ------------------------------------------------------------------
    # 1. Check DATABASE_URL
    # ------------------------------------------------------------------
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not found in environment / .env file.")
        print("   Example: DATABASE_URL=postgresql://postgres:password@localhost:5432/chatbot_memory")
        sys.exit(1)

    print(f"📡 Connecting to: {database_url.split('@')[-1]}")  # hide credentials

    # ------------------------------------------------------------------
    # 2. Connect via SQLAlchemy
    # ------------------------------------------------------------------
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connected to PostgreSQL via SQLAlchemy.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Enable pgvector extension
    # ------------------------------------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        print("✅ pgvector extension enabled.")
    except Exception as e:
        print(f"❌ Could not enable pgvector extension: {e}")
        print("   Make sure pgvector is installed on your PostgreSQL server.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Create memories table (BIGSERIAL + TEXT as per schema)
    # ------------------------------------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          BIGSERIAL PRIMARY KEY,
                    user_id     TEXT,
                    content     TEXT,
                    embedding   VECTOR(384),
                    created_at  TIMESTAMP DEFAULT NOW()
                );
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id);
            """))
            conn.commit()
        print("✅ 'memories' table created (or already exists).")
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Verify with test insert + query
    # ------------------------------------------------------------------
    try:
        from embeddings import generate_embedding

        test_text = "Setup verification test entry"
        test_embedding = generate_embedding(test_text)

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO memories (user_id, content, embedding)
                    VALUES (:user_id, :content, CAST(:embedding AS vector))
                    RETURNING id;
                """),
                {
                    "user_id": "__setup_test__",
                    "content": test_text,
                    "embedding": str(test_embedding),
                },
            )
            test_id = result.fetchone()[0]

            # Similarity search
            row = conn.execute(
                text("""
                    SELECT id, content, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM   memories
                    WHERE  user_id = :user_id
                    ORDER  BY embedding <=> CAST(:embedding AS vector)
                    LIMIT 1;
                """),
                {
                    "embedding": str(test_embedding),
                    "user_id": "__setup_test__",
                },
            ).fetchone()

            assert row is not None, "No result returned from similarity search"
            assert row[0] == test_id, "Unexpected row returned"

            # Clean up test row
            conn.execute(
                text("DELETE FROM memories WHERE user_id = :user_id;"),
                {"user_id": "__setup_test__"},
            )
            conn.commit()
        print("✅ Verification passed — insert, vector search, and cleanup all succeeded.")
    except Exception as e:
        print(f"⚠️  Verification step failed: {e}")
        print("   Table was created but the smoke test didn't pass.")

    print("\n🎉 Memory database setup complete!")


if __name__ == "__main__":
    main()
