import os
from sqlalchemy import create_engine, text

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DB_URI"])
    return _engine

def execute_query(sql: str) -> list[dict]:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]
