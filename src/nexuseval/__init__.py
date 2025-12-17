"""
NexusEval - Production-Ready LLM Evaluation Framework

A comprehensive framework for evaluating RAG pipelines and LLM reliability
with multi-provider support, intelligent caching, and dataset management.
"""

from .core import TestCase, MetricResult, EvaluationResult, MetricConfig
from .runner import Evaluator
from .llm import LLMClient
from .metrics.standard import BaseMetric, Faithfulness, AnswerRelevance, Completeness
from .dataset import Dataset, DatasetLoader, DatasetValidator, SampleDataGenerator
from .cache import CacheManager, InMemoryCache, FileCache, NoOpCache
from .config import NexusConfig, LLMConfig, CacheConfig, EvaluationConfig, ReportingConfig
from .providers import LLMProviderRegistry, BaseLLMProvider

__version__ = "0.4.0"

__all__ = [
    # Core models
    "TestCase",
    "MetricResult",
    "EvaluationResult",
    "MetricConfig",
    
    # Evaluation
    "Evaluator",
    "BaseMetric",
    "Faithfulness",
    "AnswerRelevance",
    "Completeness",
    
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