from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage
from app.llm.client import get_llm
from app.llm.prompts import GENERATE_PROMPT, CORRECT_PROMPT
from app.db.engine import execute_query
from app.validators.sql_guard import is_safe

MAX_RETRIES = 3


def _normalize_sql(sql: str) -> str:
    cleaned = sql.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("sql\n"):
        cleaned = cleaned[4:].strip()
    return cleaned

class AgentState(TypedDict):
    question: str
    schema: str
    sql: str
    result: Optional[list]
    error: Optional[str]
    retries: int

def generate_sql(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = GENERATE_PROMPT.format(schema=state["schema"], question=state["question"])
    sql = _normalize_sql(llm.invoke([HumanMessage(content=prompt)]).content)
    return {**state, "sql": sql, "error": None}

def validate_and_execute(state: AgentState) -> AgentState:
    sql = state["sql"]
    if not is_safe(sql):
        return {**state, "error": "Blocked: destructive SQL detected", "result": None}
    try:
        result = execute_query(sql)
        return {**state, "result": result, "error": None}
    except Exception as e:
        return {**state, "error": str(e), "result": None}

def correct_sql(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = CORRECT_PROMPT.format(sql=state["sql"], error=state["error"], schema=state["schema"])
    sql = _normalize_sql(llm.invoke([HumanMessage(content=prompt)]).content)
    return {**state, "sql": sql, "retries": state["retries"] + 1}

def should_retry(state: AgentState) -> str:
    if state.get("error") and state["retries"] < MAX_RETRIES:
        return "correct"
    return "end"
