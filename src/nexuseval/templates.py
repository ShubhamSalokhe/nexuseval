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

COMPLETENESS_PROMPT = """
You are a strict evaluator checking for logical completeness.
Your task is to determine if the 'Actual Output' addresses ALL parts of the 'User Input'.

USER INPUT:
{input_text}

ACTUAL OUTPUT:
{output}

STEPS:
1. Analyze the User Input and list all distinct questions, constraints, or instructions.
2. Check if the Actual Output addresses each item from Step 1.
3. If any part is missed or ignored, the answer is incomplete.

Return valid JSON only:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<List what was missed, if anything>"
}}
"""