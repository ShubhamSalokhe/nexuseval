import asyncio
from typing import List, Dict
from tqdm.asyncio import tqdm
from .core import TestCase
from .metrics.standard import BaseMetric

class Evaluator:
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    async def _evaluate_single(self, test_case: TestCase) -> Dict:
        """Helper to run all metrics for ONE test case in parallel."""
        tasks = [metric.measure(test_case) for metric in self.metrics]
        results = await asyncio.gather(*tasks)
        
        return {
            "input": test_case.input_text,
            "output": test_case.actual_output,
            "metrics": {res.metric_name: {"score": res.score, "reason": res.reason} for res in results}
        }

    def evaluate(self, test_cases: List[TestCase]):
        """
        The main entry point. Runs the async loop seamlessly for the user.
        """
        print(f"🚀 NexusEval: Evaluating {len(test_cases)} cases with {len(self.metrics)} metrics...")
        
        # Run the async loop
        results = asyncio.run(self._run_all(test_cases))
        return results

    async def _run_all(self, test_cases: List[TestCase]):
        tasks = [self._evaluate_single(case) for case in test_cases]
        # tqdm creates a progress bar
        return await tqdm.gather(*tasks)