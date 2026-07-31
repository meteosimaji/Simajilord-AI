from __future__ import annotations

import os
from pathlib import Path

import pytest

from simajilord.providers.translation import (
    MacOSTranslationProvider,
    resolve_translation_helper,
    source_translation_package,
)
from simajilord.services.translation import TranslationProviderError


def test_source_translation_package_requires_a_real_checkout(tmp_path: Path) -> None:
    runtime_path = tmp_path / "repo" / "src" / "simajilord" / "runtime.py"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.touch()
    package = tmp_path / "repo" / "native" / "macos" / "TranslationHelper"

    assert source_translation_package(runtime_path) is None

    package.mkdir(parents=True)
    (package / "Package.swift").write_text("// package", encoding="utf-8")
    assert source_translation_package(runtime_path) == package

    wheel_runtime = (
        tmp_path
        / "venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "simajilord"
        / "runtime.py"
    )
    wheel_runtime.parent.mkdir(parents=True)
    wheel_runtime.touch()
    assert source_translation_package(wheel_runtime) is None


def test_configured_translation_helper_must_be_private_and_executable(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "TranslationHelper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")

    helper.chmod(0o700)
    ready = resolve_translation_helper(None, executable_path=helper)
    assert ready.ready is True
    assert ready.source == "configured"
    assert ready.command == (str(helper),)

    helper.chmod(0o755)
    public = resolve_translation_helper(None, executable_path=helper)
    assert public.ready is False
    assert public.error_code == "translation.helper_unavailable"
    assert "chmod 700" in public.detail

    helper.chmod(0o600)
    not_executable = resolve_translation_helper(None, executable_path=helper)
    assert not_executable.ready is False
    assert "chmod 700" in not_executable.detail


def test_installed_wheel_requires_an_external_translation_helper() -> None:
    resolution = resolve_translation_helper(None)

    assert resolution.ready is False
    assert resolution.source == "installed-wheel"
    assert resolution.error_code == "translation.helper_missing"
    assert "TRANSLATION_HELPER_PATH" in resolution.detail


def test_provider_uses_configured_helper_and_reports_missing_wheel_dependency(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "TranslationHelper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    provider = MacOSTranslationProvider(None, executable_path=helper)

    assert provider._command() == (str(helper),)

    unavailable = MacOSTranslationProvider(None)
    with pytest.raises(TranslationProviderError) as raised:
        unavailable._command()
    assert raised.value.code == "translation.helper_missing"

    assert os.access(helper, os.X_OK)
