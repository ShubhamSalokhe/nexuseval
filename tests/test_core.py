"""
Unit tests for core data models.
"""

import pytest
from datetime import datetime
from nexuseval.core import TestCase, MetricResult, EvaluationResult, MetricConfig

class TestTestCase:
    """Test TestCase model."""
    
    def test_basic_creation(self):
        """Test creating a basic test case."""
        tc = TestCase(
            input_text="What is 2+2?",
            actual_output="2+2 equals 4.",
        )
        assert tc.input_text == "What is 2+2?"
        assert tc.actual_output == "2+2 equals 4."
        assert tc.retrieval_context is None
        assert tc.expected_output is None
    
    def test_with_context(self):
        """Test test case with retrieval context."""
        tc = TestCase(
            input_text="What is the capital of France?",
            actual_output="Paris",
            retrieval_context=["France is in Europe", "Paris is the capital"]
        )
        assert len(tc.retrieval_context) == 2
        assert "Paris is the capital" in tc.retrieval_context
    
    def test_with_conversation_history(self):
        """Test test case with conversation history."""
        tc = TestCase(
            input_text="And what about Germany?",
            actual_output="Berlin",
            conversation_history=[
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"}
            ]
        )
        assert len(tc.conversation_history) == 2
        assert tc.conversation_history[0]["role"] == "user"
    
    def test_with_metadata(self):
        """Test test case with metadata."""
        tc = TestCase(
            input_text="Test",
            actual_output="Response",
            metadata={"model": "gpt-4", "latency_ms": 250}
        )
        assert tc.metadata["model"] == "gpt-4"
        assert tc.metadata["latency_ms"] == 250
    
    def test_validation_fails_missing_required(self):
        """Test that validation fails without required fields."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TestCase(input_text="Only input")


class TestMetricResult:
    """Test MetricResult model."""
    
    def test_basic_creation(self):
        """Test creating a basic metric result."""
        result = MetricResult(
            metric_name="Faithfulness",
            score=0.85,
            reason="Output is well supported",
            passed=True
        )
        assert result.metric_name == "Faithfulness"
        assert result.score == 0.85
        assert result.passed is True
    
    def test_with_execution_time(self):
        """Test metric result with execution time."""
        result = MetricResult(
            metric_name="AnswerRelevance",
            score=0.92,
            reason="Highly relevant",
            passed=True,
            execution_time=1.23
        )
        assert result.execution_time == 1.23
    
    def test_with_metadata(self):
        """Test metric result with additional metadata."""
        result = MetricResult(
            metric_name="Test",
            score=0.5,
            reason="Test",
            passed=False,
            metadata={"debug_info": "some data"}
        )
        assert result.metadata["debug_info"] == "some data"


class TestEvaluationResult:
    """Test EvaluationResult model."""
    
    def test_basic_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            test_case_id="test_001",
            input_text="What is AI?",
            actual_output="Artificial Intelligence",
            metric_results={
                "Faithfulness": MetricResult(
                    metric_name="Faithfulness",
                    score=0.9,
                    reason="Good",
                    passed=True
                )
            },
            aggregate_score=0.9,
            execution_time=2.5
        )
        assert result.test_case_id == "test_001"
        assert result.aggregate_score == 0.9
        assert "Faithfulness" in result.metric_results
        assert isinstance(result.timestamp, datetime)
    
    def test_with_cost(self):
        """Test evaluation result with cost tracking."""
        result = EvaluationResult(
            test_case_id="test_002",
            input_text="Test",
            actual_output="Response",
            metric_results={},
            aggregate_score=0.5,
            execution_time=1.0,
            cost=0.001
        )
        assert result.cost == 0.001


class TestMetricConfig:
    """Test MetricConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MetricConfig()
        assert config.threshold == 0.5
        assert config.weight == 1.0
        assert config.enabled is True
        assert config.llm_model is None
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = MetricConfig(
            threshold=0.7,
            weight=2.0,
            enabled=False,
            llm_model="gpt-4o"
        )
        assert config.threshold == 0.7
        assert config.weight == 2.0
        assert config.enabled is False
        assert config.llm_model == "gpt-4o"
    
    def test_threshold_validation(self):
        """Test that threshold is validated to be between 0 and 1."""
        # Valid thresholds
        MetricConfig(threshold=0.0)
        MetricConfig(threshold=1.0)
        MetricConfig(threshold=0.5)
        
        # Invalid thresholds
        with pytest.raises(Exception):  # Pydantic ValidationError
            MetricConfig(threshold=-0.1)
        
        with pytest.raises(Exception):
            MetricConfig(threshold=1.5)
