"""
Evaluation metrics for RAG systems.

This package contains standard and advanced metrics for evaluating
RAG (Retrieval-Augmented Generation) systems.
"""

# Standard metrics
from .standard import BaseMetric, Faithfulness, AnswerRelevance, Completeness

# Advanced metrics
try:
    from .advanced import (
        ContextRelevance,
        SemanticSimilarity,
        BiasDetection,
        ToxicityDetection,
        FactualConsistency
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

# Metric registry
from .registry import MetricRegistry

# Custom metric template (for users to extend)
from .custom_template import CustomMetricTemplate

__all__ = [
    # Base
    "BaseMetric",
    
    # Standard metrics
    "Faithfulness",
    "AnswerRelevance",
    "Completeness",
    
    # Advanced metrics (if available)
    "ContextRelevance",
    "SemanticSimilarity",
    "BiasDetection",
    "ToxicityDetection",
    "FactualConsistency",
    
    # Registry
    "MetricRegistry",
    
    # Template
    "CustomMetricTemplate",
]

# Remove advanced metrics from __all__ if not available
if not ADVANCED_METRICS_AVAILABLE:
    __all__ = [item for item in __all__ if item not in [
        "ContextRelevance",
        "SemanticSimilarity",
        "BiasDetection",
        "ToxicityDetection",
        "FactualConsistency"
    ]]
