# src/nexuseval/templates.py

FAITHFULNESS_PROMPT = """
You are a strict judge evaluating a RAG system.
Your task is to check if the 'Actual Output' is fully supported by the 'Context'.

CONTEXT:
{context}

ACTUAL OUTPUT:
{output}

STEPS:
1. Break down the Actual Output into individual statements.
2. For each statement, check if it is supported by the Context.
3. If ANY statement is not supported, it is a Hallucination.

Return valid JSON only:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<concise explanation>"
}}
"""

RELEVANCE_PROMPT = """
You are an expert evaluator. Determine if the 'Actual Output' is relevant to the 'User Input'.
Ignore whether the answer is factually true; focus ONLY on whether it addresses the question asked.

USER INPUT:
{input_text}

ACTUAL OUTPUT:
{output}

Return valid JSON only:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<concise explanation>"
}}
"""