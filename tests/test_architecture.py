from __future__ import annotations

import ast
import re
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

    def inspect_scope(
        nodes: list[ast.stmt],
        *,
        path: Path,
        scope: str,
    ) -> None:
        seen: set[str] = set()
        for node in nodes:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in seen:
                collisions.append(f"{path}:{scope}:{node.name}")
            seen.add(node.name)
            if isinstance(node, ast.ClassDef):
                inspect_scope(node.body, path=path, scope=f"{scope}.{node.name}")

    for path in Path("src/simajilord").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        inspect_scope(tree.body, path=path, scope="<module>")
    assert collisions == []


def test_user_errors_use_stable_machine_codes() -> None:
    unstable: list[str] = []
    code_pattern = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")
    for path in Path("src/simajilord").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "UserError"
                or not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                continue
            if code_pattern.fullmatch(node.args[0].value) is None:
                unstable.append(f"{path}:{node.lineno}:{node.args[0].value}")
    assert unstable == []


def test_read_capabilities_declare_disclosure_classes() -> None:
    missing: list[str] = []
    declared: dict[str, str] = {}
    for path in Path("src/simajilord").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "CapabilityDescriptor"
            ):
                continue
            keywords = {item.arg: item.value for item in node.keywords if item.arg}
            name_node = keywords.get("name") or (node.args[0] if node.args else None)
            risk_node = keywords.get("risk") or (
                node.args[2] if len(node.args) > 2 else None
            )
            class_node = keywords.get("disclosure_class")
            name = (
                name_node.value
                if isinstance(name_node, ast.Constant)
                and isinstance(name_node.value, str)
                else f"{path}:{node.lineno}"
            )
            if isinstance(class_node, ast.Attribute):
                declared[name] = class_node.attr
            if (
                isinstance(risk_node, ast.Attribute)
                and risk_node.attr == "READ"
                and class_node is None
            ):
                missing.append(f"{path}:{node.lineno}:{name}")

    assert missing == []
    assert {
        "memory.search": "ACTOR_PRIVATE",
        "connector.read": "EXTERNAL_PRIVATE",
        "files.list": "ACTOR_PRIVATE",
        "files.read": "ACTOR_PRIVATE",
        "compute.run": "ACTOR_PRIVATE",
    }.items() <= declared.items()
