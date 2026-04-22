from sqlalchemy import create_engine, text
from app.core.config import settings

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        if not settings.db_uri:
            raise RuntimeError("DB_URI is not configured")
        _engine = create_engine(settings.db_uri)
    return _engine

def execute_query(sql: str) -> list[dict]:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]
