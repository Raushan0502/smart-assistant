"""
Apply the PL/SQL packages, triggers and views to Oracle.

    python db/apply_plsql.py            # apply everything in db/plsql
    python db/apply_plsql.py --verify   # apply, then check objects are VALID

Run after ``manage.py migrate``: the package bodies reference the tables Django
creates, so the tables have to exist first.

This is a separate step rather than a container init script because init
scripts run only on first boot. Iterating on a package would otherwise mean
destroying and recreating the database, and every statement here is written to
be safely re-runnable (``CREATE OR REPLACE``).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import oracledb
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
PLSQL_DIR = REPO_ROOT / "db" / "plsql"


def connect() -> oracledb.Connection:
    """Open a connection using the repository .env settings."""
    load_dotenv(REPO_ROOT / ".env")
    dsn = (
        f"{os.getenv('ORACLE_HOST', 'localhost')}:"
        f"{os.getenv('ORACLE_PORT', '1521')}/"
        f"{os.getenv('ORACLE_SERVICE', 'FREEPDB1')}"
    )
    return oracledb.connect(
        user=os.getenv("ORACLE_USER", "smartinbox"),
        password=os.getenv("ORACLE_PASSWORD", ""),
        dsn=dsn,
    )


def split_statements(sql: str) -> list[str]:
    """Split a script into executable statements.

    Oracle scripts cannot be split on semicolons: a PL/SQL block contains them
    internally. The convention this file follows -- a lone ``/`` on its own
    line terminating each block -- is what SQL*Plus uses, so the files stay
    runnable by hand as well as by this script.
    """
    statements: list[str] = []
    buffer: list[str] = []

    for line in sql.splitlines():
        if line.strip() == "/":
            statement = "\n".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            continue
        # Drop whole-line comments; inline ones are left for Oracle to parse.
        if line.strip().startswith("--"):
            continue
        buffer.append(line)

    trailing = "\n".join(buffer).strip()
    if trailing:
        # Anything after the last "/" may still be several plain statements.
        for part in (s.strip() for s in trailing.split(";")):
            if part:
                statements.append(part)
    return statements


def apply_file(cursor: oracledb.Cursor, path: Path) -> int:
    """Execute one SQL file, returning how many statements ran."""
    statements = split_statements(path.read_text(encoding="utf-8"))
    for statement in statements:
        try:
            cursor.execute(statement)
        except oracledb.DatabaseError as exc:
            head = re.sub(r"\s+", " ", statement)[:110]
            raise RuntimeError(f"{path.name}: failed on `{head}...`\n  {exc}") from exc
    return len(statements)


def verify(cursor: oracledb.Cursor) -> list[str]:
    """Report any package, trigger or view left in an INVALID state.

    Oracle will happily create a package body containing a compilation error
    and mark it INVALID rather than failing the statement, so a script can
    "succeed" while leaving nothing that works. This is the check that catches
    that.
    """
    cursor.execute(
        """
        SELECT object_name, object_type, status
          FROM user_objects
         WHERE object_type IN ('PACKAGE', 'PACKAGE BODY', 'TRIGGER', 'VIEW')
           AND object_name LIKE 'SI_%'
         ORDER BY object_name
        """
    )
    rows = cursor.fetchall()
    problems = []
    for name, kind, status in rows:
        marker = "ok " if status == "VALID" else "BAD"
        print(f"  [{marker}] {kind:<13} {name} ({status})")
        if status != "VALID":
            problems.append(f"{kind} {name} is {status}")
    if not rows:
        problems.append("No SI_* objects found; nothing was created.")
    return problems


def show_errors(cursor: oracledb.Cursor) -> None:
    """Print Oracle's compilation errors for anything that failed to compile."""
    cursor.execute(
        """
        SELECT name, type, line, position, text
          FROM user_errors
         WHERE name LIKE 'SI_%'
         ORDER BY name, sequence
        """
    )
    for name, kind, line, position, text in cursor.fetchall():
        print(f"    {kind} {name} line {line}:{position} - {text.strip()}")


def main() -> int:
    """Apply every PL/SQL file, optionally verifying the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check objects are VALID")
    args = parser.parse_args()

    files = sorted(PLSQL_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files in {PLSQL_DIR}")
        return 1

    with connect() as connection:
        cursor = connection.cursor()
        for path in files:
            count = apply_file(cursor, path)
            print(f"applied {path.name} ({count} statements)")
        connection.commit()

        if args.verify:
            print("\nverifying objects:")
            problems = verify(cursor)
            if problems:
                print("\nPROBLEMS:")
                for problem in problems:
                    print(f"  - {problem}")
                show_errors(cursor)
                return 1
            print("\nAll PL/SQL objects are VALID.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
