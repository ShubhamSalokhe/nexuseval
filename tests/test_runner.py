"""Offline regressions for evaluation, concurrency, and configuration."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nexuseval import (
    AnswerRelevance, CacheConfig, EvaluationConfig, Evaluator, Faithfulness,
    LLMConfig, MetricResult, NexusConfig, TestCase as Case,
)
from nexuseval.metrics.custom_template import ResponseLengthMetric
from nexuseval.metrics.standard import BaseMetric


def quiet_config(**kwargs):
    return NexusConfig(evaluation=EvaluationConfig(progress_bar=False), **kwargs)


class RecordingMetric(BaseMetric):
    def __init__(self):
        super().__init__("Recording")
        self.active = 0
        self.peak = 0

    async def measure(self, case):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001 * (4 - int(case.input_text) % 4))
            return MetricResult(metric_name=self.name, score=1.0, reason="OK", passed=True)
        finally:
            self.active -= 1


def test_rule_based_evaluation_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    evaluator = Evaluator([ResponseLengthMetric(min_length=1)], config=quiet_config())
    results = evaluator.evaluate([Case(input_text="Question", actual_output="Answer")])
    assert results[0]["metrics"]["Response Length"]["passed"] is True
    assert evaluator._llm is None


@pytest.mark.asyncio
async def test_async_evaluation_preserves_order_and_limits_tasks():
    metric = RecordingMetric()
    evaluator = Evaluator([metric], max_concurrency=3, config=quiet_config())
    cases = [Case(input_text=str(i), actual_output="Answer") for i in range(60)]
    original_create = asyncio.create_task
    task_count = 0

    def create(coro, **kwargs):
        nonlocal task_count
        task_count += 1
        return original_create(coro, **kwargs)

    with patch("nexuseval.runner.asyncio.create_task", side_effect=create):
        results = await evaluator.evaluate_async(cases)
    assert [result["input"] for result in results] == [case.input_text for case in cases]
    assert metric.peak == 3
    # Three workers plus one task per metric evaluation, no task per queued case.
    assert task_count == len(cases) + 3
    assert await evaluator.evaluate_async([]) == []


@pytest.mark.asyncio
async def test_sync_entrypoint_inside_event_loop_has_clear_error():
    evaluator = Evaluator([], config=quiet_config())
    with pytest.raises(RuntimeError, match="await evaluator.evaluate_async"):
        evaluator.evaluate([])


@pytest.mark.parametrize("concurrency", [0, -1, 1.5, True])
def test_invalid_concurrency_rejected(concurrency):
    with pytest.raises(ValueError, match="positive integer"):
        Evaluator([], max_concurrency=concurrency)


def test_duplicate_metric_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        Evaluator([Faithfulness(), Faithfulness()])


@pytest.mark.asyncio
async def test_configured_evaluation_shares_client_and_persistent_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = AsyncMock()
    provider.generate_json.return_value = {"score": 0.9, "reason": "Supported"}
    config = quiet_config(
        llm=LLMConfig(provider="custom", model="test-model", temperature=0.6, max_tokens=42),
        cache=CacheConfig(backend="file", cache_dir=str(tmp_path / "nested" / "cache")),
    )
    case = Case(input_text="Q", actual_output="A", retrieval_context=["Context"])
    with patch("nexuseval.llm.LLMProviderRegistry.create", return_value=provider) as factory:
        metrics = [Faithfulness(), AnswerRelevance()]
        evaluator = Evaluator(metrics, config=config)
        await evaluator.evaluate_async([case])
        assert metrics[0].llm is metrics[1].llm
        assert factory.call_count == 1
        assert factory.call_args.kwargs["temperature"] == 0.6
        assert provider.generate_json.await_count == 2
        await evaluator.evaluate_async([case])
        assert provider.generate_json.await_count == 2
        # A new evaluator reads the same persistent responses.
        await Evaluator([Faithfulness(), AnswerRelevance()], config=config).evaluate_async([case])
        assert provider.generate_json.await_count == 2


@pytest.mark.asyncio
async def test_explicit_metric_client_is_preserved():
    client = AsyncMock()
    client.get_score.return_value = {"score": 0.8, "reason": "OK"}
    metric = AnswerRelevance(llm=client)
    await Evaluator([metric], config=quiet_config()).evaluate_async([Case(input_text="Q", actual_output="A")])
    assert metric.llm is client


@pytest.mark.asyncio
async def test_failure_cancels_other_metrics_and_workers():
    started = asyncio.Event()
    stopped = asyncio.Event()

    class WaitingMetric(BaseMetric):
        async def measure(self, case):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class FailingMetric(BaseMetric):
        async def measure(self, case):
            await started.wait()
            raise ValueError("Invalid provider response")

    evaluator = Evaluator(
        [WaitingMetric("Waiting"), FailingMetric("Failing")], max_concurrency=2, config=quiet_config()
    )
    with pytest.raises(ValueError, match="Invalid provider response"):
        await evaluator.evaluate_async([Case(input_text="Q", actual_output="A")] * 4)
    assert stopped.is_set()
    assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


@pytest.mark.parametrize("llm_retries,expected", [(None, 5), (0, 0)])
def test_retry_configuration_precedence(llm_retries, expected):
    llm = LLMConfig(**({} if llm_retries is None else {"max_retries": llm_retries}))
    config = NexusConfig(llm=llm, evaluation=EvaluationConfig(retry_attempts=5))
    evaluator = Evaluator([Faithfulness()], config=config)
    with patch("nexuseval.runner.LLMClient") as client:
        evaluator._get_llm()
    assert client.call_args.kwargs["config"].max_retries == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_name", ["ContextRelevance", "FactualConsistency"])
async def test_metric_failure_drains_nested_requests(metric_name):
    from nexuseval import ContextRelevance, FactualConsistency

    started = asyncio.Event()
    stopped = asyncio.Event()

    async def request(*args):
        if "slow" in args:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        await started.wait()
        raise RuntimeError("Provider unavailable")

    if metric_name == "ContextRelevance":
        metric = ContextRelevance()
        metric._evaluate_chunk = request
    else:
        metric = FactualConsistency()
        metric._extract_claims = AsyncMock(return_value=["slow", "fail"])
        metric._verify_claim = request
    evaluator = Evaluator([metric], config=quiet_config())
    case = Case(input_text="Q", actual_output="A", retrieval_context=["slow", "fail"])
    existing_tasks = asyncio.all_tasks()
    try:
        with pytest.raises(RuntimeError, match="Provider unavailable"):
            await evaluator.evaluate_async([case])
        assert stopped.is_set()
        assert not (asyncio.all_tasks() - existing_tasks)
    finally:
        pending = asyncio.all_tasks() - existing_tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
