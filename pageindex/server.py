import os
import sys
import logging
import json
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the parent directory to sys.path to import embeddings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from embeddings import generate_embedding
except ImportError:
    # Fallback if pathing is tricky
    import logging
    logging.warning("PageIndex Server: Could not import embeddings from parent dir.")
    def generate_embedding(text): return [0.0] * 384

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("PageIndex")

# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

def init_db():
    if not engine:
        return
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                content TEXT,
                embedding VECTOR(384)
            );
        """))
        conn.commit()

# Initialize on startup
try:
    init_db()
except Exception as e:
    logging.error(f"Failed to initialize database: {e}")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def pageIndex_add_document(filename: str, content: str) -> str:
    """
    Index a document by storing its content and embedding.
    """
    if not engine:
        return "Error: DATABASE_URL not set."
    
    try:
        embedding = generate_embedding(content)
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO document_chunks (filename, content, embedding)
                    VALUES (:filename, :content, CAST(:embedding AS vector))
                """),
                {
                    "filename": filename,
                    "content": content,
                    "embedding": str(embedding)
                }
            )
            conn.commit()
        return f"Successfully indexed document: {filename}"
    except Exception as e:
        return f"Error indexing document: {str(e)}"

@mcp.tool()
def pageIndex_search(query: str, limit: int = 3) -> str:
    """
    Search indexed documents for relevant content.
    """
    if not engine:
        return "Error: DATABASE_URL not set."
    
    try:
        query_embedding = generate_embedding(query)
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT filename, content, 
                           1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
                    FROM document_chunks
                    ORDER BY embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :limit;
                """),
                {
                    "query_embedding": str(query_embedding),
                    "limit": limit
                }
            )
            rows = result.fetchall()
            
            if not rows:
                return "No matching documents found."
            
            formatted_results = []
            for row in rows:
                formatted_results.append(f"Source: {row[0]}\nSimilarity: {row[2]:.4f}\nContent: {row[1]}")
            
            return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

if __name__ == "__main__":
    mcp.run()
