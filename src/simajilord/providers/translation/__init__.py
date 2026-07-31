"""Offline translation providers."""

from .macos import (
    MacOSTranslationProvider,
    TranslationHelperResolution,
    resolve_translation_helper,
    source_translation_package,
)

__all__ = [
    "MacOSTranslationProvider",
    "TranslationHelperResolution",
    "resolve_translation_helper",
    "source_translation_package",
]
