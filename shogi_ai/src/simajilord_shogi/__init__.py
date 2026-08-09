"""Transport-independent Simajilord shogi engine and learning lab."""

from .branding import ENGINE_AUTHOR, ENGINE_DISPLAY_NAME, ENGINE_ROMANIZED_NAME, ENGINE_USI_NAME
from .config import ModelConfig, ReanalysisConfig, SearchConfig, model_profile, optional_max_plies

__all__ = [
    "ENGINE_AUTHOR",
    "ENGINE_DISPLAY_NAME",
    "ENGINE_ROMANIZED_NAME",
    "ENGINE_USI_NAME",
    "ModelConfig",
    "ReanalysisConfig",
    "SearchConfig",
    "model_profile",
    "optional_max_plies",
]

__version__ = "0.1.0"
