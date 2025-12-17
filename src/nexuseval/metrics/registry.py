"""
Metric registry for managing and discovering metrics.
"""

from typing import Dict, Type, List
from .standard import BaseMetric

class MetricRegistry:
    """
    Central registry for managing metrics.
    
    Allows registration of custom metrics and discovery of available metrics.
    
    Example:
        >>> from nexuseval.metrics import MetricRegistry, BaseMetric
        >>> 
        >>> class MyMetric(BaseMetric):
        ...     async def measure(self, test_case):
        ...         return MetricResult(...)
        >>> 
        >>> MetricRegistry.register("my_metric", MyMetric)
        >>> metric_class = MetricRegistry.get("my_metric")
        >>> metric = metric_class()
    """
    
    _metrics: Dict[str, Type[BaseMetric]] = {}
    _builtin_metrics: Dict[str, Type[BaseMetric]] = {}
    
    @classmethod
    def register(cls, name: str, metric_class: Type[BaseMetric], builtin: bool = False):
        """
        Register a metric.
        
        Args:
            name: Unique metric name (lowercase, underscores)
            metric_class: Metric class (must inherit from BaseMetric)
            builtin: Whether this is a built-in metric
        
        Raises:
            ValueError: If metric_class doesn't inherit from BaseMetric
            KeyError: If name is already registered
        """
        if not issubclass(metric_class, BaseMetric):
            raise ValueError(
                f"Metric class must inherit from BaseMetric, "
                f"got {metric_class}"
            )
        
        if name in cls._metrics:
            raise KeyError(f"Metric '{name}' is already registered")
        
        if builtin:
            cls._builtin_metrics[name] = metric_class
        
        cls._metrics[name] = metric_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseMetric]:
        """
        Get a metric class by name.
        
        Args:
            name: Metric name
        
        Returns:
            Metric class
        
        Raises:
            KeyError: If metric is not found
        """
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise KeyError(
                f"Metric '{name}' not found. "
                f"Available metrics: {', '.join(available)}"
            )
        
        return cls._metrics[name]
    
    @classmethod
    def list_metrics(cls, include_custom: bool = True) -> List[str]:
        """
        List all registered metric names.
        
        Args:
            include_custom: Whether to include custom (non-builtin) metrics
        
        Returns:
            List of metric names
        """
        if include_custom:
            return list(cls._metrics.keys())
        else:
            return list(cls._builtin_metrics.keys())
    
    @classmethod
    def get_metric_info(cls, name: str) -> Dict[str, any]:
        """
        Get information about a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Dict with metric information
        """
        metric_class = cls.get(name)
        return {
            "name": name,
            "class_name": metric_class.__name__,
            "description": metric_class.__doc__,
            "is_builtin": name in cls._builtin_metrics
        }
    
    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a custom metric.
        
        Args:
            name: Metric name
        
        Raises:
            KeyError: If metric is not found
            ValueError: If trying to unregister a built-in metric
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' not found")
        
        if name in cls._builtin_metrics:
            raise ValueError(f"Cannot unregister built-in metric '{name}'")
        
        del cls._metrics[name]


# Auto-register built-in metrics
def _register_builtin_metrics():
    """Register all built-in metrics."""
    from .standard import Faithfulness, AnswerRelevance, Completeness
    
    MetricRegistry.register("faithfulness", Faithfulness, builtin=True)
    MetricRegistry.register("answer_relevance", AnswerRelevance, builtin=True)
    MetricRegistry.register("completeness", Completeness, builtin=True)
    
    # Register advanced metrics if available
    try:
        from .advanced import (
            ContextRelevance,
            SemanticSimilarity,
            BiasDetection,
            ToxicityDetection,
            FactualConsistency
        )
        
        MetricRegistry.register("context_relevance", ContextRelevance, builtin=True)
        MetricRegistry.register("semantic_similarity", SemanticSimilarity, builtin=True)
        MetricRegistry.register("bias_detection", BiasDetection, builtin=True)
        MetricRegistry.register("toxicity_detection", ToxicityDetection, builtin=True)
        MetricRegistry.register("factual_consistency", FactualConsistency, builtin=True)
    except ImportError:
        pass  # Advanced metrics not available


# Auto-register on import
_register_builtin_metrics()
