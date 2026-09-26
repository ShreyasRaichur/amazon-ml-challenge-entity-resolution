"""
Business Entity Resolution System
=================================
Tripartite entity resolution across heterogeneous noisy sources (S1, S2, S3)
optimizing for Macro-F0.5 with high precision and country-agnostic generalizability.
"""

from __future__ import annotations

from .agent import BusinessEntityResolutionAgent, DataResourceManager
from .config import DEFAULT_CONFIG, PipelineConfig
from .inference import EntityResolutionPipeline

__version__ = "1.0.0"
__all__ = [
    "BusinessEntityResolutionAgent",
    "DataResourceManager",
    "EntityResolutionPipeline",
    "PipelineConfig",
    "DEFAULT_CONFIG",
]
