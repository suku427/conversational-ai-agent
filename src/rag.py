import json
import os
from functools import lru_cache
from typing import Any, Dict, List

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_NAME = os.getenv("POSTGRES_DB", "vectordb")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
EMBEDDING_DIMENSIONS = 1536 if os.getenv("OPENAI_API_KEY") else 384


@lru_cache(maxsize=1)
def get_local_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = True
    register_vector(conn)
    return conn


def load_knowledge_base() -> List[Dict[str, str]]:
    """Read the JSON knowledge base used by the app."""
    file_path = os.path.join("data", "knowledge_base.json")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents: List[Dict[str, str]] = [
        {
            "content": (
                f"Company: {data['company_name']}\n"
                f"Product: {data['product']}\n\n"
                "PLANS:\n"
                + "\n".join(
                    [
                        f"- {plan['name']}: {plan['price']}\n  Features: {', '.join(plan['features'])}"
                        for plan in data["plans"]
                    ]
                )
                + "\n\nPOLICIES:\n"
                + f"- Refunds: {data['policies']['refund_policy']}\n"
                + f"- Support: {data['policies']['support_policy']}\n"
            )
        }
    ]
    documents.extend(
        {
            "content": (
                f"Plan: {plan['name']}\n"
                f"Price: {plan['price']}\n"
                f"Features: {', '.join(plan['features'])}"
            )
        }
        for plan in data["plans"]
    )
    return documents


def ensure_database_schema() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIMENSIONS})
        );
        """
    )
    cur.execute("TRUNCATE TABLE knowledge_base;")
    cur.close()
    conn.close()


def embed_text(text: str) -> List[float]:
    """Generate an embedding using OpenAI if configured, otherwise fallback to a local model."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    model = get_local_embedding_model()
    return model.encode(text).tolist()


def ingest_knowledge_base() -> None:
    """Populate the PostgreSQL table with vectorized knowledge base rows."""
    ensure_database_schema()
    records = load_knowledge_base()
    conn = get_connection()
    cur = conn.cursor()
    for record in records:
        embedding = embed_text(record["content"])
        cur.execute(
            "INSERT INTO knowledge_base (content, embedding) VALUES (%s, %s)",
            (record["content"], embedding),
        )
    cur.close()
    conn.close()


def retrieve_context(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Run a cosine-similarity search against PostgreSQL pgvector; fallback to JSON data if missing."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        query_vector = embed_text(query)
        cur.execute(
            """
            SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
            FROM knowledge_base
            ORDER BY embedding <=> %s::vector DESC
            LIMIT %s;
            """,
            (query_vector, query_vector, limit),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if rows:
            return [
                {"id": row[0], "content": row[1], "similarity": float(row[2])}
                for row in rows
            ]
    except Exception:
        pass

    fallback = load_knowledge_base()
    return [{"id": idx, "content": item["content"], "similarity": 1.0} for idx, item in enumerate(fallback[:limit])]


if __name__ == "__main__":
    print("Ingesting the knowledge base into PostgreSQL...")
    ingest_knowledge_base()
    print("Retrieval test:")
    for item in retrieve_context("How much does the Basic Plan cost?"):
        print(item["content"])
