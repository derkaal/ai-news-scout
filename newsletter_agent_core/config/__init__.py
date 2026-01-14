"""
Configuration package for the newsletter agent.

This package provides configuration management for axiom-based analysis
and other agent settings.
"""

from .axiom_config import AxiomConfig
from .sovereignty_config import SovereigntyConfig

__all__ = ["AxiomConfig", "SovereigntyConfig"]