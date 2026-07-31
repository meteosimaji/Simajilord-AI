"""Resolve the macOS TranslationHelper without connecting to Discord."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from simajilord.providers.translation import (
    TranslationHelperResolution,
    resolve_translation_helper,
    source_translation_package,
)


def inspect_translation_helper(
    *,
    runtime_path: Path,
    executable_path: Path | None,
) -> TranslationHelperResolution:
    """Return the same helper resolution used by the composition root."""

    return resolve_translation_helper(
        source_translation_package(runtime_path),
        executable_path=executable_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check how the optional macOS on-device translation helper will be resolved."
        )
    )
    parser.add_argument(
        "--helper-path",
        type=Path,
        help="Override TRANSLATION_HELPER_PATH for this check.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable diagnostic.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configured = args.helper_path
    if configured is None:
        raw_path = os.getenv("TRANSLATION_HELPER_PATH", "").strip()
        configured = Path(raw_path).expanduser() if raw_path else None
    runtime_path = Path(__file__).resolve().parents[1] / "runtime.py"
    result = inspect_translation_helper(
        runtime_path=runtime_path,
        executable_path=configured,
    )
    payload = {
        "ready": result.ready,
        "source": result.source,
        "command": list(result.command),
        "error_code": result.error_code,
        "detail": result.detail,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Translation helper: {'ready' if result.ready else 'unavailable'}")
        print(f"Source: {result.source}")
        print(result.detail)
        if result.command:
            print(f"Command: {' '.join(result.command)}")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
