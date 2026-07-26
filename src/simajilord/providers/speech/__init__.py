"""Speech synthesis providers."""

from .macos import MacOSSayProvider
from .voicevox import VoicevoxSpeechProvider

__all__ = ["MacOSSayProvider", "VoicevoxSpeechProvider"]
