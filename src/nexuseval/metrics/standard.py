from abc import ABC, abstractmethod
from ..core import TestCase, MetricResult
from ..llm import LLMClient
from ..templates import FAITHFULNESS_PROMPT, RELEVANCE_PROMPT

class BaseMetric(ABC):
    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self.llm = LLMClient()

    @abstractmethod
    async def measure(self, test_case: TestCase) -> MetricResult:
        pass

class Faithfulness(BaseMetric):
    def __init__(self):
        super().__init__("Faithfulness", threshold=0.7)

    async def measure(self, test_case: TestCase) -> MetricResult:
        if not test_case.retrieval_context:
            return MetricResult(metric_name=self.name, score=0.0, reason="No context provided", passed=False)
        
        # Format the prompt
        formatted_prompt = FAITHFULNESS_PROMPT.format(
            context="\n".join(test_case.retrieval_context),
            output=test_case.actual_output
        )
        
        # Call LLM
        result = await self.llm.get_score(formatted_prompt)
        score = result.get("score", 0.0)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=result.get("reason", "No reason"),
            passed=score >= self.threshold
        )

class AnswerRelevance(BaseMetric):
    def __init__(self):
        super().__init__("Answer Relevance", threshold=0.7)

    async def measure(self, test_case: TestCase) -> MetricResult:
        formatted_prompt = RELEVANCE_PROMPT.format(
            input_text=test_case.input_text,
            output=test_case.actual_output
        )
        
        result = await self.llm.get_score(formatted_prompt)
        score = result.get("score", 0.0)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=result.get("reason", "No reason"),
            passed=score >= self.threshold
        )