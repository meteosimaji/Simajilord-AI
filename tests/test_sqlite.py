from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from simajilord.sqlite import closing_connect


def test_sqlite_transaction_contexts_use_closing_factory() -> None:
    source_root = Path(__file__).parents[1] / "src" / "simajilord"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        uses_self_connect_context = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                expression = item.context_expr
                if not isinstance(expression, ast.Call):
                    continue
                function = expression.func
                if (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "sqlite3"
                    and function.attr == "connect"
                ):
                    violations.append(f"{path}: raw sqlite3.connect context")
                if (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "self"
                    and function.attr == "_connect"
                ):
                    uses_self_connect_context = True
        if (
            uses_self_connect_context
            and "from simajilord.sqlite import closing_connect" not in source
        ):
            violations.append(f"{path}: self._connect context is not closing")

    assert violations == []


def test_closing_connect_commits_then_closes(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    connection = closing_connect(path)

    with connection:
        connection.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
        connection.execute("INSERT INTO values_table VALUES ('committed')")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    with sqlite3.connect(path) as verifier:
        assert verifier.execute("SELECT value FROM values_table").fetchone() == (
            "committed",
        )


def test_closing_connect_rolls_back_then_closes(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as setup:
        setup.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
    connection = closing_connect(path)

    with pytest.raises(RuntimeError, match="rollback"), connection:
        connection.execute("INSERT INTO values_table VALUES ('not committed')")
        raise RuntimeError("rollback")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    with sqlite3.connect(path) as verifier:
        assert verifier.execute("SELECT COUNT(*) FROM values_table").fetchone() == (0,)
