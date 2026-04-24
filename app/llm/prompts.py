GENERATE_PROMPT = """\
You are a SQL expert. Given the schema below, write a single read-only SQL query.
The schema is provided in TOON (Token-Oriented Object Notation).
Return ONLY a TOON object with a single field named sql.

Output format:
sql: 'SELECT ...;'

Schema:
{schema_toon}

Question: {question}
"""

CORRECT_PROMPT = """\
The following SQL query produced an error. Fix it.
The schema is provided in TOON (Token-Oriented Object Notation).
Return ONLY a TOON object with a single field named sql.

Output format:
sql: 'SELECT ...;'

SQL:
{sql}

Error:
{error}

Schema:
{schema_toon}
"""
