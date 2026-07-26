from __future__ import annotations

import ast
from pathlib import Path


def test_platform_layers_do_not_import_discord() -> None:
    root = Path("src/simajilord")
    platform_roots = (
        root / "agent",
        root / "capabilities",
        root / "core",
        root / "domain",
        root / "media",
        root / "observability",
        root / "providers",
        root / "services",
    )
    offenders: list[str] = []
    for package in platform_roots:
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names = (node.module or "",)
                else:
                    continue
                if any(name == "discord" or name.startswith("discord.") for name in names):
                    offenders.append(str(path))
    assert offenders == []


def test_no_duplicate_top_level_definitions() -> None:
    collisions: list[str] = []
    for path in Path("src/simajilord").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in seen:
                    collisions.append(f"{path}:{node.name}")
                seen.add(node.name)
    assert collisions == []
