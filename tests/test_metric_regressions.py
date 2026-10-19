"""Scoring regressions using structured provider doubles."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from nexuseval import (
    AnswerRelevance, BiasDetection, Completeness, ContextRelevance, Faithfulness,
    FactualConsistency, SemanticSimilarity, TestCase as Case, ToxicityDetection,
)
from nexuseval.embeddings import EmbeddingClient


@pytest.fixture
def case():
    return Case(input_text="Q", actual_output="A", retrieval_context=["One", "Two"], expected_output="Expected")


@pytest.mark.parametrize("metric_class", [Faithfulness, AnswerRelevance, Completeness, ContextRelevance,
                                          BiasDetection, ToxicityDetection, FactualConsistency])
def test_initialization_needs_no_openai_key(metric_class, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    metric = metric_class()
    assert metric._llm is None


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_class,payload", [
    (metric, {}) for metric in (Faithfulness, AnswerRelevance, Completeness, ContextRelevance,
                               BiasDetection, ToxicityDetection, FactualConsistency)
] + [
    (metric, {"score": 0.0, "reason": "Provider failed"})
    for metric in (ContextRelevance, BiasDetection, ToxicityDetection, FactualConsistency)
])
async def test_missing_response_fields_cannot_become_success(metric_class, payload, case):
    client = AsyncMock()
    client.get_score.return_value = payload
    with pytest.raises(ValidationError):
        await metric_class(llm=client).measure(case)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_class,flag,score_field", [
    (BiasDetection, "has_bias", "severity"), (ToxicityDetection, "is_toxic", "toxicity_score")
])
async def test_safety_scores_are_lower_is_better_and_use_threshold(metric_class, flag, score_field, case):
    client = AsyncMock()
    client.get_score.return_value = {flag: False, score_field: 0.0}
    clean = await metric_class(llm=client).measure(case)
    assert clean.score == 0.0
    assert clean.passed
    client.get_score.return_value = {flag: True, score_field: 0.4}
    detected = await metric_class(llm=client).measure(case)
    assert detected.score == 0.4
    assert not detected.passed
    tolerated = await metric_class(threshold=0.5, llm=client).measure(case)
    assert tolerated.passed


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_class,payload", [
    (ContextRelevance, {"relevant": "false", "reason": "Bad type"}),
    (BiasDetection, {"has_bias": "false", "severity": 0.0}),
    (ToxicityDetection, {"is_toxic": "false", "toxicity_score": 0.0}),
    (AnswerRelevance, {"score": 2.0, "reason": "Out of range"}),
    (Faithfulness, {"score": float("nan"), "reason": "Not finite"}),
])
async def test_malformed_scores_and_boolean_strings_rejected(metric_class, payload, case):
    client = AsyncMock()
    client.get_score.return_value = payload
    with pytest.raises(ValidationError):
        await metric_class(llm=client).measure(case)


@pytest.mark.asyncio
async def test_context_relevance_computes_fraction(case):
    client = AsyncMock()
    client.get_score.side_effect = [{"relevant": True, "reason": "Useful"}, {"relevant": False, "reason": "Unrelated"}]
    result = await ContextRelevance(llm=client).measure(case)
    assert result.score == 0.5
    assert not result.passed
    assert len(result.metadata["chunk_evaluations"]) == 2


@pytest.mark.asyncio
async def test_factual_consistency_verifies_claims_and_enforces_limit(case):
    client = AsyncMock()
    client.get_score.side_effect = [
        {"claims": ["First", "Second", "Ignored"]},
        {"verified": True, "confidence": 0.9, "reason": "Supported"},
        {"verified": False, "confidence": 0.2, "reason": "Unsupported"},
    ]
    result = await FactualConsistency(max_claims=2, llm=client).measure(case)
    assert result.score == 0.5
    assert result.metadata["total_claims"] == 2
    assert not result.passed
    assert client.get_score.await_count == 3


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_claim_limits_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        FactualConsistency(max_claims=count)


@pytest.mark.parametrize("first,second,expected", [
    ([1.0, 0.0], [1.0, 0.0], 1.0),
    ([1.0, 0.0], [0.0, 1.0], 0.0),
    ([1.0, 0.0], [-1.0, 0.0], 0.0),
    ([0.0, 0.0], [1.0, 1.0], 0.0),
])
def test_cosine_similarity_matches_geometry(first, second, expected):
    assert EmbeddingClient.cosine_similarity(np.array(first), np.array(second)) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_local_similarity_needs_no_openai_client(case, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("nexuseval.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", True), patch(
        "nexuseval.embeddings.SentenceTransformer", create=True
    ) as transformer, patch("nexuseval.embeddings.AsyncOpenAI") as openai:
        transformer.return_value.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
        metric = SemanticSimilarity(embedding_provider="local")
        result = await metric.measure(case)
        assert result.score == 1.0
        transformer.assert_called_once_with("all-MiniLM-L6-v2")
        openai.assert_not_called()
