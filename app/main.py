from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from dataclasses import dataclass
from typing import Optional
from app.graph.graph import build_graph
from app.retriever.schema_retriever import SchemaRetriever

_graph = None
_retriever = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = SchemaRetriever()
    return _retriever

@dataclass
class QueryResponse:
    sql: str
    result: Optional[list]
    error: Optional[str]
    retries: int

def query_agent(question: str) -> QueryResponse:
    schema = _get_retriever().retrieve(question)
    state = {"question": question, "schema": schema, "sql": "", "result": None, "error": None, "retries": 0}
    final = _get_graph().invoke(state)
    return QueryResponse(sql=final["sql"], result=final["result"], error=final["error"], retries=final["retries"])
