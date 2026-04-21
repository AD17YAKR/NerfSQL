from fastapi import FastAPI
from pydantic import BaseModel
from app.main import query_agent, QueryResponse

app = FastAPI(title="SQL-RAG Agent")

class QueryRequest(BaseModel):
    question: str

@app.post("/query", response_model=dict)
def query(req: QueryRequest):
    resp: QueryResponse = query_agent(req.question)
    return {"sql": resp.sql, "result": resp.result, "error": resp.error, "retries": resp.retries}

@app.get("/health")
def health():
    return {"status": "ok"}
