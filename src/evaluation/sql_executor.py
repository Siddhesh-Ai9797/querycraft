# src/evaluation/sql_executor.py

import sqlite3
import re
from typing import Optional


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string for comparison.

    Lowercases everything, collapses multiple spaces into one,
    and strips leading/trailing whitespace.

    This is necessary because the model might generate:
      "SELECT Name FROM students"
    when the gold answer is:
      "select name from students"
    These are logically identical — normalization makes them match.
    """
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    return sql


def execute_sql(sql: str, schema_context: str) -> dict:
    """
    Try to execute a SQL query against an in-memory SQLite database.

    We build the database from the CREATE TABLE statements in
    schema_context, then run the query. This tells us if the SQL
    is syntactically and structurally valid.

    Args:
        sql:            The SQL query to execute
        schema_context: CREATE TABLE statements from the dataset

    Returns:
        dict with keys:
            success (bool)  — did the query execute without error?
            error   (str)   — error message if failed, None if success
            rows    (int)   — number of rows returned if success
    """
    try:
        # Create a fresh in-memory database for each evaluation
        # In-memory means it disappears when the connection closes
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Build the schema — execute each CREATE TABLE statement
        # The schema_context may contain multiple CREATE TABLE blocks
        # split by semicolons
        statements = [
            s.strip()
            for s in schema_context.split(";")
            if s.strip()
        ]
        for stmt in statements:
            if stmt.upper().startswith("CREATE"):
                cursor.execute(stmt)

        # Now try to execute the actual query
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        return {
            "success": True,
            "error": None,
            "rows": len(rows),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "rows": 0,
        }