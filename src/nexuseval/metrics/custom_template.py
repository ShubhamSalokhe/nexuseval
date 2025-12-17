"""
Template for creating custom metrics.

Use this template as a starting point for creating your own metrics.
"""

from abc import ABC
from ..core import TestCase, MetricResult
from .standard import BaseMetric
from ..llm import LLMClient

class CustomMetricTemplate(BaseMetric):
    """
    Template for creating custom metrics.
    
    To create a custom metric:
    1. Copy this template
    2. Rename the class
    3. Implement the measure() method
    4. Optionally override __init__() for custom configuration
    5. Register with MetricRegistry if needed
    
    Example:
        >>> class MyCustomMetric(BaseMetric):
        ...     def __init__(self):
        ...         super().__init__("My Custom Metric", threshold=0.75)
        ...     
        ...     async def measure(self, test_case: TestCase) -> MetricResult:
        ...         # Your custom logic here
        ...         score = self._calculate_score(test_case)
        ...         reason = "Explanation of the score"
        ...         
        ...         return MetricResult(
        ...             metric_name=self.name,
        ...             score=score,
        ...             reason=reason,
        ...             passed=score >= self.threshold
        ...         )
        ...     
        ...     def _calculate_score(self, test_case):
        ...         # Helper method
        ...         return 0.8
    """
    
    def __init__(
        self,
        name: str = "Custom Metric",
        threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the custom metric.
        
        Args:
            name: Metric name (will appear in results)
            threshold: Pass/fail threshold (0.0 to 1.0)
            **kwargs: Additional configuration options
        """
        super().__init__(name, threshold)
        
        # Add custom initialization here
        # Example: self.custom_param = kwargs.get("custom_param", default_value)
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        """
        Evaluate a test case and return a metric result.
        
        This is where you implement your custom evaluation logic.
        
        Args:
            test_case: TestCase object containing:
                - input_text: User's query
                - actual_output: LLM's response
                - retrieval_context: Retrieved chunks (optional)
                - expected_output: Ground truth answer (optional)
                - metadata: Additional data (optional)
        
        Returns:
            MetricResult with score, reason, and pass/fail status
        """
        # TODO: Implement your custom evaluation logic
        
        # Example 1: Simple rule-based scoring
        # score = 1.0 if len(test_case.actual_output) > 100 else 0.5
        
        # Example 2: LLM-based evaluation
        # prompt = f"Evaluate this response: {test_case.actual_output}"
        # result = await self.llm.get_score(prompt)
        # score = result.get("score", 0.0)
        
        # Example 3: Custom calculation
        # score = self._calculate_custom_score(test_case)
        
        score = 0.0  # Replace with your logic
        reason = "Not implemented - replace with actual evaluation"
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=score >= self.threshold,
            metadata={}  # Optional: add extra data
        )
    
    # Add helper methods as needed
    def _calculate_custom_score(self, test_case: TestCase) -> float:
        """
        Helper method for custom score calculation.
        
        Args:
            test_case: Test case to evaluate
        
        Returns:
            Score between 0.0 and 1.0
        """
        # Your custom logic here
        return 0.0


# Example: Length-based metric
class ResponseLengthMetric(BaseMetric):
    """
    Example metric that checks if response length is within acceptable range.
    
    This is a simple example to demonstrate the pattern.
    """
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 500,
        threshold: float = 1.0
    ):
        super().__init__("Response Length", threshold)
        self.min_length = min_length
        self.max_length = max_length
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        length = len(test_case.actual_output)
        
        if self.min_length <= length <= self.max_length:
            score = 1.0
            reason = f"Length {length} is within range [{self.min_length}, {self.max_length}]"
            passed = True
        else:
            score = 0.0
            if length < self.min_length:
                reason = f"Length {length} is too short (min: {self.min_length})"
            else:
                reason = f"Length {length} is too long (max: {self.max_length})"
            passed = False
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=passed,
            metadata={"length": length}
        )


# Example: Keyword presence metric
class KeywordPresenceMetric(BaseMetric):
    """
    Example metric that checks for required keywords in the response.
    """
    
    def __init__(
        self,
        required_keywords: list[str],
        threshold: float = 0.8
    ):
        super().__init__("Keyword Presence", threshold)
        self.required_keywords = [kw.lower() for kw in required_keywords]
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        output_lower = test_case.actual_output.lower()
        
        found_keywords = [
            kw for kw in self.required_keywords
            if kw in output_lower
        ]
        
        score = len(found_keywords) / len(self.required_keywords) if self.required_keywords else 1.0
        
        missing_keywords = [
            kw for kw in self.required_keywords
            if kw not in found_keywords
        ]
        
        if missing_keywords:
            reason = f"Missing keywords: {', '.join(missing_keywords)}"
        else:
            reason = "All required keywords present"
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=score >= self.threshold,
            metadata={
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords
            }
        )
