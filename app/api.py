import json
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from app.main import QueryResponse, query_agent
from scripts.ingest_schema import extract_schema, upsert_schema_chunks_to_pinecone

app = FastAPI(title="SQL-RAG Agent")

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    db_uri: str | None = None  # defaults to DB_URI from .env if omitted


def _compact_sql(sql: str) -> str:
    # Deterministic API-side formatting for JSON clients.
    return re.sub(r"\s+", " ", (sql or "")).strip()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(req: IngestRequest = IngestRequest()):
    db_uri = req.db_uri or settings.db_uri
    if not db_uri:
        raise HTTPException(status_code=400, detail="db_uri required")
    try:
        chunks = extract_schema(db_uri)
        with open("data/schema_chunks.json", "w") as f:
            json.dump(chunks, f, indent=2)
        # reset retriever so it reloads fresh chunks
        import app.main as _main

        _main._retriever = None

        pinecone_status = "skipped"
        api_key = settings.pinecone_api_key
        if api_key:
            upserted = upsert_schema_chunks_to_pinecone(
                chunks=chunks,
                index_name=settings.pinecone_index_name,
                namespace=settings.pinecone_namespace,
                region=settings.pinecone_region,
                api_key=api_key,
            )
            pinecone_status = f"upserted:{upserted}"

        return {
            "ingested": len(chunks),
            "tables": [c.split("\n")[0].replace("Table: ", "") for c in chunks],
            "pinecone": pinecone_status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
def schema():
    try:
        with open("data/schema_chunks.json") as f:
            chunks = json.load(f)
        return {"count": len(chunks), "chunks": chunks}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Schema not ingested yet. Call POST /ingest first.")

@app.post("/query")
def query(req: QueryRequest):
    resp: QueryResponse = query_agent(req.question)
    return {
        "sql": _compact_sql(resp.sql),
        "sql_raw": resp.sql,
        "result": resp.result,
        "error": resp.error,
        "retries": resp.retries,
    }
