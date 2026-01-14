"""
Validation package for newsletter clustering engine.

Provides comprehensive validation and quality assessment capabilities.
"""

from .validator import ClusteringValidator, ValidationResult

__all__ = [
    "ClusteringValidator",
    "ValidationResult"
]