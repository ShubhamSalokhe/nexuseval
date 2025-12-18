import asyncio
import time
import sys
from io import StringIO
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

from nexuseval.core import TestCase, MetricResult
from nexuseval.runner import Evaluator
from nexuseval.metrics.standard import BaseMetric
from nexuseval.retry import retry_with_exponential_backoff
import json

# --- Mocks ---

class MockLLMProvider:
    def __init__(self, *args, **kwargs):
        self.model = "mock-model"
        self.total_usage = {"input": 0, "output": 0}
        
    async def generate_json(self, prompt, **kwargs):
        # Simulate network delay
        await asyncio.sleep(0.1)
        return {"score": 0.9, "reason": "Mock reasoning"}

    def get_usage_stats(self):
        return {"total_cost_usd": 0.0}

class MockMetric(BaseMetric):
    def __init__(self, name="MockMetric"):
        self.name = name
        self.threshold = 0.5
        # Mock LLMClient to avoid API key check
        self.llm = MagicMock()
        self.llm.verbose = False
        self.llm.provider = MockLLMProvider() 
        
        # We need to mock get_score because BaseMetric uses it
        async def mock_get_score(prompt, **kwargs):
             return await self.llm.provider.generate_json(prompt, **kwargs)
        
        self.llm.get_score = mock_get_score

    async def measure(self, test_case: TestCase) -> MetricResult:
        # Simulate work
        await asyncio.sleep(0.1) 
        return MetricResult(
            metric_name=self.name,
            score=0.9,
            reason="Mock reason",
            passed=True
        )

# --- Tests ---

async def test_concurrency():
    print("Testing Concurrency...")
    
    # Create 10 test cases
    cases = [TestCase(input_text=f"Q{i}", actual_output=f"A{i}") for i in range(10)]
    
    # Create evaluator with concurrency = 2
    metric = MockMetric()
    evaluator = Evaluator(metrics=[metric], max_concurrency=2)
    
    start_time = time.time()
    await evaluator._run_all(cases)
    end_time = time.time()
    
    duration = end_time - start_time
    # Each task takes 0.1s. 
    # With concurrency 2, we have 5 batches of 2.
    # Theoretical min time = 5 * 0.1 = 0.5s.
    # Without concurrency limit (unbounded), it would be ~0.1s (all at once).
    
    print(f"Duration with concurrency 2: {duration:.4f}s")
    
    if duration >= 0.5:
        print("✅ Concurrency limiting appears effective (took >= 0.5s)")
    else:
        print("❌ Concurrency limiting might be failing (too fast)")

async def test_retry_logic():
    print("\nTesting Retry Logic...")
    
    attempts = 0
    
    @retry_with_exponential_backoff(
        max_retries=3,
        initial_delay=0.1,
        errors=(ValueError,),
        on_error=lambda e, i: print(f"  caught error: {e}, attempt {i}")
    )
    async def unstable_function():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Fail!")
        return "Success"
        
    result = await unstable_function()
    print(f"Result: {result}, Total Attempts: {attempts}")
    
    if result == "Success" and attempts == 3:
        print("✅ Retry logic worked (3 attempts)")
    else:
        print("❌ Retry logic failed")

def test_verbose_logging():
    print("\nTesting Verbose Logging...")
    
    # Capture stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    try:
        # We need to rely on the actual LLMClient plumbing for this test
        # So we'll patch the LLMClient.__init__ to avoid Api Key check
        with patch('nexuseval.llm.LLMClient.__init__', return_value=None), \
             patch('nexuseval.llm.LLMClient.provider', create=True) as mock_provider:
             
            from nexuseval.llm import LLMClient
            client = LLMClient(verbose=True)
            
            # Re-set verbose because our init patch wiped it
            client.verbose = True
            client.provider_name = "mock"
            client.model = "test"
            client.enable_cache = False
            
            # Make generate_json awaitable
            async def async_return(*args, **kwargs):
                return {"foo": "bar"}
            mock_provider.generate_json.side_effect = async_return
            
            # This needs to be an async call
            asyncio.run(client.get_score("test prompt"))
            
    finally:
        sys.stdout = sys.__stdout__
        
    output = captured_output.getvalue()
    if "[NexusEval]" in output and "mock/test" in output:
         print("✅ Verbose logging detected")
    else:
         print("❌ Verbose logging not found in output")
         print("Output was:", output)


if __name__ == "__main__":
    asyncio.run(test_concurrency())
    asyncio.run(test_retry_logic())
    test_verbose_logging()
