from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field
from ..core import TestCase, MetricResult
from ..llm import LLMClient
from ..templates import FAITHFULNESS_PROMPT, RELEVANCE_PROMPT, COMPLETENESS_PROMPT

class ScoreResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str


class BaseMetric(ABC):
    def __init__(self, name: str, threshold: float = 0.5, llm: Optional[LLMClient] = None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.name = name
        self.threshold = threshold
        self._llm = llm
        self._llm_factory = LLMClient
        self._verbose = None

    @property
    def llm(self) -> LLMClient:
        # Rule-based metrics and missing-context checks need no API credentials.
        if self._llm is None:
            self._llm = self._llm_factory()
        if getattr(self, "_verbose", None) is not None:
            self._llm.verbose = self._verbose
        return self._llm

    @llm.setter
    def llm(self, client: LLMClient):
        self._llm = client

    @abstractmethod
    async def measure(self, test_case: TestCase) -> MetricResult:
        pass

class Faithfulness(BaseMetric):
    def __init__(self, threshold: float = 0.7, llm: Optional[LLMClient] = None):
        super().__init__("Faithfulness", threshold, llm)

    async def measure(self, test_case: TestCase) -> MetricResult:
        if not test_case.retrieval_context:
            return MetricResult(metric_name=self.name, score=0.0, reason="No context provided", passed=False)
        
        # Format the prompt
        formatted_prompt = FAITHFULNESS_PROMPT.format(
            context="\n".join(test_case.retrieval_context),
            output=test_case.actual_output
        )
        
        # Call LLM
        result = ScoreResponse.model_validate(
            await self.llm.get_score(formatted_prompt, response_model=ScoreResponse)
        )
        score = result.score
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=result.reason,
            passed=score >= self.threshold
        )

class AnswerRelevance(BaseMetric):
    def __init__(self, threshold: float = 0.7, llm: Optional[LLMClient] = None):
        super().__init__("Answer Relevance", threshold, llm)

    async def measure(self, test_case: TestCase) -> MetricResult:
        formatted_prompt = RELEVANCE_PROMPT.format(
            input_text=test_case.input_text,
            output=test_case.actual_output
        )
        
        result = ScoreResponse.model_validate(
            await self.llm.get_score(formatted_prompt, response_model=ScoreResponse)
        )
        score = result.score
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=result.reason,
            passed=score >= self.threshold
        )


class Completeness(BaseMetric):
    """
    Checks if the LLM answered ALL parts of the user's question.
    """
    def __init__(self, threshold: float = 0.7, llm: Optional[LLMClient] = None):
        super().__init__("Completeness", threshold, llm)

    async def measure(self, test_case: TestCase) -> MetricResult:
        # Format the prompt using the user input and the model output
        formatted_prompt = COMPLETENESS_PROMPT.format(
            input_text=test_case.input_text,
            output=test_case.actual_output
        )
        
        # Call the Async LLM Client
        result = ScoreResponse.model_validate(
            await self.llm.get_score(formatted_prompt, response_model=ScoreResponse)
        )
        score = result.score
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=result.reason,
            passed=score >= self.threshold
        )
