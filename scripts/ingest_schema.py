import argparse
import json
from sqlalchemy import create_engine, inspect

def extract_schema(db_uri: str) -> list[str]:
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    chunks = []
    for table in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns(table)]
        fks = [f"{fk['constrained_columns']} -> {fk['referred_table']}" for fk in inspector.get_foreign_keys(table)]
        chunk = f"Table: {table}\nColumns: {', '.join(cols)}"
        if fks:
            chunk += f"\nForeign keys: {'; '.join(fks)}"
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_uri", required=True)
    parser.add_argument("--output", default="data/schema_chunks.json")
    args = parser.parse_args()

    chunks = extract_schema(args.db_uri)
    with open(args.output, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Wrote {len(chunks)} schema chunks to {args.output}")
