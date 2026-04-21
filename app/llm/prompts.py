GENERATE_PROMPT = """\
You are a SQL expert. Given the schema below, write a single read-only SQL query.
Return ONLY the SQL, no explanation.

Schema:
{schema}

Question: {question}
"""

CORRECT_PROMPT = """\
The following SQL query produced an error. Fix it and return ONLY the corrected SQL.

SQL:
{sql}

Error:
{error}

Schema:
{schema}
"""
