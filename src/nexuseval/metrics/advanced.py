"""
Advanced evaluation metrics for RAG systems.

This module contains advanced metrics beyond the basic triad:
- Context Relevance: Measures retrieval precision
- Semantic Similarity: Embedding-based similarity
- Bias Detection: Detects various bias types
- Toxicity Detection: Flags harmful content
- Factual Consistency: Verifies claims against context
"""

import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC
from ..core import TestCase, MetricResult
from ..llm import LLMClient
from ..templates import (
    CONTEXT_RELEVANCE_PROMPT,
    BIAS_DETECTION_PROMPT,
    TOXICITY_DETECTION_PROMPT,
    EXTRACT_CLAIMS_PROMPT,
    VERIFY_CLAIM_PROMPT
)
from .standard import BaseMetric

# Try to import optional dependencies
try:
    from ..embeddings import EmbeddingClient
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class ContextRelevance(BaseMetric):
    """
    Measures how relevant the retrieved context is to the user's query.
    
    This metric evaluates retrieval precision by checking what proportion
    of retrieved chunks are actually relevant to answering the query.
    
    Score: relevant_chunks / total_chunks
    """
    
    def __init__(self, threshold: float = 0.7):
        super().__init__("Context Relevance", threshold)
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        if not test_case.retrieval_context:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason="No context provided",
                passed=False
            )
        
        # Evaluate each context chunk for relevance
        relevant_count = 0
        total_chunks = len(test_case.retrieval_context)
        evaluations = []
        
        tasks = [
            self._evaluate_chunk(test_case.input_text, chunk)
            for chunk in test_case.retrieval_context
        ]
        chunk_results = await asyncio.gather(*tasks)
        
        for i, (is_relevant, reason) in enumerate(chunk_results):
            if is_relevant:
                relevant_count += 1
            evaluations.append({
                "chunk_index": i,
                "relevant": is_relevant,
                "reason": reason
            })
        
        score = relevant_count / total_chunks if total_chunks > 0 else 0.0
        
        reason = (
            f"{relevant_count}/{total_chunks} chunks relevant. "
            f"{'Passed' if score >= self.threshold else 'Failed'} threshold of {self.threshold}"
        )
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=score >= self.threshold,
            metadata={"chunk_evaluations": evaluations}
        )
    
    async def _evaluate_chunk(self, query: str, chunk: str) -> tuple[bool, str]:
        """Evaluate if a single chunk is relevant to the query."""
        formatted_prompt = CONTEXT_RELEVANCE_PROMPT.format(
            query=query,
            context_chunk=chunk
        )
        
        result = await self.llm.get_score(formatted_prompt)
        is_relevant = result.get("relevant", False)
        reason = result.get("reason", "No reason provided")
        
        return is_relevant, reason


class SemanticSimilarity(BaseMetric):
    """
    Calculates semantic similarity between actual and expected output.
    
    Uses embeddings to measure how semantically similar the generated
    answer is to the expected answer, regardless of exact wording.
    
    Requires expected_output in TestCase.
    """
    
    def __init__(
        self,
        threshold: float = 0.8,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small"
    ):
        super().__init__("Semantic Similarity", threshold)
        
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embeddings module not available. "
                "Install with: pip install openai or pip install sentence-transformers"
            )
        
        self.embedding_client = EmbeddingClient(
            provider=embedding_provider,
            model=embedding_model
        )
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        if not test_case.expected_output:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason="No expected output provided for comparison",
                passed=False
            )
        
        # Calculate cosine similarity using embeddings
        similarity = await self.embedding_client.similarity(
            test_case.actual_output,
            test_case.expected_output
        )
        
        reason = (
            f"Semantic similarity: {similarity:.3f}. "
            f"{'Passed' if similarity >= self.threshold else 'Failed'} "
            f"threshold of {self.threshold}"
        )
        
        return MetricResult(
            metric_name=self.name,
            score=similarity,
            reason=reason,
            passed=similarity >= self.threshold,
            metadata={
                "embedding_provider": self.embedding_client.provider,
                "embedding_model": self.embedding_client.model
            }
        )


class BiasDetection(BaseMetric):
    """
    Detects potential biases in generated responses.
    
    Checks for various bias types:
    - Gender bias
    - Racial/ethnic bias
    - Religious bias
    - Age bias
    - Disability bias
    - Nationality bias
    
    Score: 0.0 if no bias, higher scores indicate more bias.
    Lower is better for this metric.
    """
    
    BIAS_TYPES = ["gender", "race", "religion", "age", "disability", "nationality"]
    
    def __init__(self, threshold: float = 0.0):
        # For bias, we want score to be 0 (no bias detected)
        super().__init__("Bias Detection", threshold)
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        formatted_prompt = BIAS_DETECTION_PROMPT.format(
            output=test_case.actual_output
        )
        
        result = await self.llm.get_score(formatted_prompt)
        
        has_bias = result.get("has_bias", False)
        bias_types = result.get("bias_types", [])
        severity = result.get("severity", 0.0)
        examples = result.get("examples", [])
        explanation = result.get("reason", "No bias detected")
        
        # Invert score - lower is better for bias
        score = 1.0 - severity if has_bias else 1.0
        
        if has_bias:
            reason = (
                f"Bias detected: {', '.join(bias_types)}. "
                f"Severity: {severity:.2f}. "
                f"Examples: {'; '.join(examples[:2])}"
            )
        else:
            reason = "No bias detected"
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=not has_bias,  # Passes if no bias detected
            metadata={
                "has_bias": has_bias,
                "bias_types": bias_types,
                "severity": severity,
                "examples": examples
            }
        )


class ToxicityDetection(BaseMetric):
    """
    Detects toxic, harmful, or inappropriate content in responses.
    
    Checks for:
    - Profanity/vulgar language
    - Threats or violence
    - Hate speech or discrimination
    - Sexual content
    - Harassment or bullying
    
    Score: 0.0 if no toxicity, higher scores indicate more toxicity.
    Lower is better for this metric.
    """
    
    def __init__(self, threshold: float = 0.0):
        # For toxicity, we want score to be 0 (no toxicity)
        super().__init__("Toxicity Detection", threshold)
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        formatted_prompt = TOXICITY_DETECTION_PROMPT.format(
            output=test_case.actual_output
        )
        
        result = await self.llm.get_score(formatted_prompt)
        
        is_toxic = result.get("is_toxic", False)
        toxicity_score = result.get("toxicity_score", 0.0)
        categories = result.get("categories", [])
        severity = result.get("severity", "low")
        explanation = result.get("reason", "No toxicity detected")
        
        # Invert score - lower is better for toxicity
        score = 1.0 - toxicity_score if is_toxic else 1.0
        
        if is_toxic:
            reason = (
                f"Toxicity detected: {', '.join(categories)}. "
                f"Severity: {severity}. "
                f"Score: {toxicity_score:.2f}"
            )
        else:
            reason = "No toxic content detected"
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=not is_toxic,  # Passes if no toxicity
            metadata={
                "is_toxic": is_toxic,
                "toxicity_score": toxicity_score,
                "categories": categories,
                "severity": severity
            }
        )


class FactualConsistency(BaseMetric):
    """
    Verifies factual claims in the response against the context.
    
    Process:
    1. Extract factual claims from the output
    2. Verify each claim against the retrieval context
    3. Calculate consistency score
    
    Score: verified_claims / total_claims
    """
    
    def __init__(self, threshold: float = 0.8, max_claims: int = 10):
        super().__init__("Factual Consistency", threshold)
        self.max_claims = max_claims
    
    async def measure(self, test_case: TestCase) -> MetricResult:
        if not test_case.retrieval_context:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason="No context provided for verification",
                passed=False
            )
        
        # Step 1: Extract claims
        claims = await self._extract_claims(test_case.actual_output)
        
        if not claims:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                reason="No factual claims to verify",
                passed=True
            )
        
        # Limit number of claims
        claims = claims[:self.max_claims]
        
        # Step 2: Verify each claim
        context = "\n".join(test_case.retrieval_context)
        
        verification_tasks = [
            self._verify_claim(claim, context)
            for claim in claims
        ]
        verifications = await asyncio.gather(*verification_tasks)
        
        # Step 3: Calculate score
        verified_count = sum(1 for verified, _, _ in verifications if verified)
        total_claims = len(claims)
        score = verified_count / total_claims if total_claims > 0 else 0.0
        
        # Compile results
        claim_details = [
            {
                "claim": claims[i],
                "verified": verified,
                "confidence": confidence,
                "reason": reason
            }
            for i, (verified, confidence, reason) in enumerate(verifications)
        ]
        
        unverified_claims = [
            claim for claim, (verified, _, _) in zip(claims, verifications)
            if not verified
        ]
        
        if unverified_claims:
            reason = (
                f"{verified_count}/{total_claims} claims verified. "
                f"Unverified: {'; '.join(unverified_claims[:2])}"
            )
        else:
            reason = f"All {total_claims} claims verified against context"
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            passed=score >= self.threshold,
            metadata={
                "total_claims": total_claims,
                "verified_count": verified_count,
                "claim_details": claim_details
            }
        )
    
    async def _extract_claims(self, output: str) -> List[str]:
        """Extract factual claims from output."""
        formatted_prompt = EXTRACT_CLAIMS_PROMPT.format(output=output)
        result = await self.llm.get_score(formatted_prompt)
        return result.get("claims", [])
    
    async def _verify_claim(self, claim: str, context: str) -> tuple[bool, float, str]:
        """Verify a single claim against context."""
        formatted_prompt = VERIFY_CLAIM_PROMPT.format(
            claim=claim,
            context=context
        )
        
        result = await self.llm.get_score(formatted_prompt)
        verified = result.get("verified", False)
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "No reason provided")
        
        return verified, confidence, reason
