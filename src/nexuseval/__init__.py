"""
NexusEval - Production-Ready LLM Evaluation Framework

A comprehensive framework for evaluating RAG pipelines and LLM reliability
with multi-provider support, intelligent caching, and dataset management.
"""

from .core import TestCase, MetricResult, EvaluationResult, MetricConfig
from .runner import Evaluator
from .llm import LLMClient
from .metrics.standard import BaseMetric, Faithfulness, AnswerRelevance, Completeness
from .metrics.registry import MetricRegistry
from .dataset import Dataset, DatasetLoader, DatasetValidator, SampleDataGenerator
from .cache import CacheManager, InMemoryCache, FileCache, NoOpCache
from .config import NexusConfig, LLMConfig, CacheConfig, EvaluationConfig, ReportingConfig
from .providers import LLMProviderRegistry, BaseLLMProvider

# Try to import advanced metrics (optional dependencies)
try:
    from .metrics.advanced import (
        ContextRelevance,
        SemanticSimilarity,
        BiasDetection,
        ToxicityDetection,
        FactualConsistency
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

__version__ = "0.5.0"

__all__ = [
    # Core models
    "TestCase",
    "MetricResult",
    "EvaluationResult",
    "MetricConfig",
    
    # Evaluation
    "Evaluator",
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
    
    # Metric registry
    "MetricRegistry",
    
    # LLM clients
    "LLMClient",
    
    # Multi-provider support
    "LLMProviderRegistry",
    "BaseLLMProvider",
    
    # Dataset management
    "Dataset",
    "DatasetLoader",
    "DatasetValidator",
    "SampleDataGenerator",
    
    # Caching
    "CacheManager",
    "InMemoryCache",
    "FileCache",
    "NoOpCache",
    
    # Configuration
    "NexusConfig",
    "LLMConfig",
    "CacheConfig",
    "EvaluationConfig",
    "ReportingConfig",
]

# Remove advanced metrics from exports if not available
if not ADVANCED_METRICS_AVAILABLE:
    __all__ = [item for item in __all__ if item not in [
        "ContextRelevance",
        "SemanticSimilarity",
        "BiasDetection",
        "ToxicityDetection",
        "FactualConsistency"
    ]]