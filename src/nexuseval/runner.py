import asyncio
from typing import List, Dict, Optional
from tqdm.auto import tqdm
from .cache import CacheManager
from .config import NexusConfig
from .core import TestCase
from .llm import LLMClient
from .metrics.standard import BaseMetric


class Evaluator:
    def __init__(
        self,
        metrics: List[BaseMetric],
        max_concurrency: Optional[int] = None,
        verbose: Optional[bool] = None,
        config: Optional[NexusConfig] = None,
    ):
        self.config = config or NexusConfig()
        self.metrics = metrics
        self.max_concurrency = (
            self.config.evaluation.max_concurrency if max_concurrency is None else max_concurrency
        )
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError("Metric names must be unique within an evaluator")
        self.verbose = (config.reporting.verbose if config else False) if verbose is None else verbose
        self._llm = None

        for metric in self.metrics:
            if isinstance(metric, BaseMetric):
                metric._verbose = self.verbose
                if config is not None and getattr(metric, "_llm", None) is None:
                    metric._llm_factory = self._get_llm

    def _get_llm(self):
        # Share a configured client across metrics, without creating clients for
        # local metrics or checks that can complete without an API request.
        if self._llm is None:
            llm_config = self.config.llm
            retry_defaults = {}
            for old_name, new_name in (("retry_attempts", "max_retries"), ("retry_delay", "retry_delay")):
                if (
                    old_name in self.config.evaluation.model_fields_set
                    and new_name not in llm_config.model_fields_set
                ):
                    retry_defaults[new_name] = getattr(self.config.evaluation, old_name)
            if retry_defaults:
                llm_config = llm_config.model_copy(update=retry_defaults)
            self._llm = LLMClient(
                config=llm_config,
                cache_manager=CacheManager.from_config(self.config.cache),
                enable_cache=self.config.cache.enabled and self.config.cache.backend != "none",
                enable_cost_tracking=self.config.evaluation.enable_cost_tracking,
                verbose=self.verbose,
            )
        return self._llm

    async def _evaluate_single(self, test_case: TestCase) -> Dict:
        """Run all metrics for one test case in parallel."""
        tasks = [asyncio.create_task(metric.measure(test_case)) for metric in self.metrics]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "input": test_case.input_text,
            "output": test_case.actual_output,
            "metrics": {
                res.metric_name: res.model_dump(exclude={"metric_name"}) for res in results
            },
        }

    def evaluate(self, test_cases: List[TestCase]):
        """Evaluate from synchronous code; use evaluate_async in an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(test_cases))
        raise RuntimeError("An event loop is running; use 'await evaluator.evaluate_async(test_cases)'")

    async def evaluate_async(self, test_cases: List[TestCase]):
        """Evaluate from async applications and notebooks, preserving input order."""
        if self.verbose:
            print(f"🚀 NexusEval: Evaluating {len(test_cases)} cases with {len(self.metrics)} metrics...")
        return await self._run_all(test_cases)

    async def _run_all(self, test_cases: List[TestCase]):
        if not test_cases:
            return []
        results = [None] * len(test_cases)
        cases = iter(enumerate(test_cases))
        progress = tqdm(total=len(test_cases), disable=not self.config.evaluation.progress_bar)

        async def worker():
            for index, case in cases:
                results[index] = await self._evaluate_single(case)
                progress.update(1)

        # Bound task creation as well as execution for large datasets.
        workers = [asyncio.create_task(worker()) for _ in range(min(self.max_concurrency, len(test_cases)))]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            progress.close()
        return results
