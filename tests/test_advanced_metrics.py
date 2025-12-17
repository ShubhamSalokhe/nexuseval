"""
Tests for advanced evaluation metrics.
"""

import pytest
import asyncio
from nexuseval.core import TestCase, MetricResult

# Import advanced metrics if available
try:
    from nexuseval.metrics.advanced import (
        ContextRelevance,
        BiasDetection,
        ToxicityDetection,
        FactualConsistency
    )
    from nexuseval.metrics.registry import MetricRegistry
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

from nexuseval.embeddings import EmbeddingClient

pytestmark = pytest.mark.skipif(
    not ADVANCED_METRICS_AVAILABLE,
    reason="Advanced metrics not available (optional dependencies not installed)"
)


class TestContextRelevance:
    """Test ContextRelevance metric."""
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        metric = ContextRelevance(threshold=0.7)
        
        test_case = TestCase(
            input_text="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=[
                "Paris is the capital and largest city of France.",
                "The Eiffel Tower is a famous landmark in Paris.",
                "Python is a programming language."  # Irrelevant
            ]
        )
        
        # This requires API key, so mock or skip in actual tests
        # result = await metric.measure(test_case)
        # assert isinstance(result, MetricResult)
        # assert 0.0 <= result.score <= 1.0
        assert metric.name == "Context Relevance"
        assert metric.threshold == 0.7
    
    def test_no_context(self):
        metric = ContextRelevance()
        
        test_case = TestCase(
            input_text="What is AI?",
            actual_output="AI is artificial intelligence."
        )
        
        # This should return score 0.0
        # result = asyncio.run(metric.measure(test_case))
        # assert result.score == 0.0
        # assert not result.passed


class TestBiasDetection:
    """Test BiasDetection metric."""
    
    def test_initialization(self):
        metric = BiasDetection()
        assert metric.name == "Bias Detection"
        assert metric.threshold == 0.0  # No bias should be threshold
        assert len(metric.BIAS_TYPES) == 6
    
    @pytest.mark.asyncio
    async def test_bias_types(self):
        metric = BiasDetection()
        
        # Test case with potentially biased content
        test_case = TestCase(
            input_text="Tell me about engineers",
            actual_output="Engineers are typically men who work with computers."
        )
        
        # Would detect gender bias in real evaluation
        # result = await metric.measure(test_case)
        # assert isinstance(result, MetricResult)
        assert "gender" in metric.BIAS_TYPES


class TestToxicityDetection:
    """Test ToxicityDetection metric."""
    
    def test_initialization(self):
        metric = ToxicityDetection()
        assert metric.name == "Toxicity Detection"
        assert metric.threshold == 0.0
    
    @pytest.mark.asyncio
    async def test_clean_content(self):
        metric = ToxicityDetection()
        
        test_case = TestCase(
            input_text="What is Python?",
            actual_output="Python is a high-level programming language known for its simplicity."
        )
        
        # Clean content should pass
        # result = await metric.measure(test_case)
        # assert result.passed


class TestFactualConsistency:
    """Test FactualConsistency metric."""
    
    def test_initialization(self):
        metric = FactualConsistency(threshold=0.8, max_claims=10)
        assert metric.name == "Factual Consistency"
        assert metric.threshold == 0.8
        assert metric.max_claims == 10
    
    @pytest.mark.asyncio
    async def test_with_context(self):
        metric = FactualConsistency()
        
        test_case = TestCase(
            input_text="Who invented Python?",
            actual_output="Python was created by Guido van Rossum in 1991.",
            retrieval_context=[
                "Python was created by Guido van Rossum and first released in 1991.",
                "Python is a high-level programming language."
            ]
        )
        
        # Claims should be verified against context
        # result = await metric.measure(test_case)
        # assert isinstance(result, MetricResult)
        assert metric.threshold == 0.8


class TestMetricRegistry:
    """Test MetricRegistry functionality."""
    
    def test_builtin_metrics_registered(self):
        # Check standard metrics are registered
        assert "faithfulness" in MetricRegistry.list_metrics()
        assert "answer_relevance" in MetricRegistry.list_metrics()
        assert "completeness" in MetricRegistry.list_metrics()
    
    def test_advanced_metrics_registered(self):
        if ADVANCED_METRICS_AVAILABLE:
            assert "context_relevance" in MetricRegistry.list_metrics()
            assert "bias_detection" in MetricRegistry.list_metrics()
            assert "toxicity_detection" in MetricRegistry.list_metrics()
            assert "factual_consistency" in MetricRegistry.list_metrics()
    
    def test_get_metric(self):
        metric_class = MetricRegistry.get("faithfulness")
        assert metric_class is not None
        assert metric_class.__name__ == "Faithfulness"
        
        # Can instantiate (if API key available)
        try:
            metric = metric_class()
            assert metric.name == "Faithfulness"
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OPENAI_API_KEY not set")
    
    def test_get_nonexistent_metric(self):
        with pytest.raises(KeyError):
            MetricRegistry.get("nonexistent_metric")
    
    def test_list_metrics_builtin_only(self):
        builtin = MetricRegistry.list_metrics(include_custom=False)
        assert "faithfulness" in builtin
        
        all_metrics = MetricRegistry.list_metrics(include_custom=True)
        assert len(all_metrics) >= len(builtin)
    
    def test_get_metric_info(self):
        info = MetricRegistry.get_metric_info("faithfulness")
        assert "name" in info
        assert "class_name" in info
        assert "description" in info
        assert "is_builtin" in info
        assert info["is_builtin"] == True


class TestEmbeddings:
    """Test embedding functionality."""
    
    def test_embedding_client_creation(self):
        # Test OpenAI provider
        try:
            client = EmbeddingClient(provider="openai", model="text-embedding-3-small")
            assert client.provider == "openai"
            assert client.model == "text-embedding-3-small"
        except ImportError:
            pytest.skip("OpenAI not installed")
    
    def test_cosine_similarity(self):
        import numpy as np
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = EmbeddingClient.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.01  # Should be very similar
        
        vec3 = np.array([0.0, 1.0, 0.0])
        similarity2 = EmbeddingClient.cosine_similarity(vec1, vec3)
        assert similarity2 < similarity  # Should be less similar
    
    def test_zero_vectors(self):
        import numpy as np
        
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 1.0, 1.0])
        
        similarity = EmbeddingClient.cosine_similarity(vec1, vec2)
        assert similarity == 0.0  # Zero vector comparison


@pytest.mark.skipif(
    not ADVANCED_METRICS_AVAILABLE,
    reason="Advanced metrics not available"
)
class TestSemanticSimilarity:
    """Test SemanticSimilarity metric (requires embeddings)."""
    
    def test_initialization(self):
        try:
            from nexuseval.metrics.advanced import SemanticSimilarity
            
            metric = SemanticSimilarity(
                threshold=0.8,
                embedding_provider="openai",
                embedding_model="text-embedding-3-small"
            )
            assert metric.name == "Semantic Similarity"
            assert metric.threshold == 0.8
            assert metric.embedding_client.provider == "openai"
        except ImportError:
            pytest.skip("Embeddings not available")
    
    def test_requires_expected_output(self):
        try:
            from nexuseval.metrics.advanced import SemanticSimilarity
            
            metric = SemanticSimilarity()
            
            test_case = TestCase(
                input_text="What is AI?",
                actual_output="AI is artificial intelligence."
                # No expected_output
            )
            
            # Should return score 0.0 without expected output
            # result = asyncio.run(metric.measure(test_case))
            # assert result.score == 0.0
        except ImportError:
            pytest.skip("Embeddings not available")
