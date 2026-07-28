"""Fail CI when a tracked project file contains a credential-shaped value."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXCLUDED_PREFIXES = (
    "vendor/",
    "activity/package-lock.json",
    "src/simajilord/activity/static/assets/",
)
PATTERNS = {
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "OpenAI API key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "Discord MFA token": re.compile(rb"\bmfa\.[A-Za-z0-9_-]{20,}\b"),
    "Discord bot token": re.compile(
        rb"\b[A-Za-z0-9_-]{23,28}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{20,}\b"
    ),
}
SECRET_ASSIGNMENT = re.compile(
    rb"(?m)^(?:DISCORD_TOKEN|DISCORD_CLIENT_SECRET|HIVE_API_KEY|"
    rb"WEB_SEARCH_SHARED_SECRET)\s*=\s*([^\s#]+)"
)
SAFE_EXAMPLE_PREFIXES = (b"replace-", b"example", b"change")


def tracked_files() -> tuple[Path, ...]:
    output = subprocess.run(
        ("git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    paths: list[Path] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="strict")
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        paths.append(ROOT / relative)
    return tuple(paths)


def findings() -> tuple[str, ...]:
    results: list[str] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        data = path.read_bytes()
        relative = path.relative_to(ROOT)
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                results.append(f"{relative}: credential-shaped {label}")
        for match in SECRET_ASSIGNMENT.finditer(data):
            value = match.group(1)
            if not value.startswith(SAFE_EXAMPLE_PREFIXES):
                results.append(f"{relative}: non-placeholder secret assignment")
    return tuple(results)


def main() -> None:
    detected = findings()
    if detected:
        raise SystemExit("Secret scan failed:\n" + "\n".join(detected))
    print("Secret scan passed.")
