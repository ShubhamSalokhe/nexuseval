# NexusEval 🧠

[![PyPI version](https://badge.fury.io/py/nexuseval.svg)](https://badge.fury.io/py/nexuseval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**The robust, open-source framework for evaluating RAG pipelines and LLM reliability.**

NexusEval helps developers measure the quality of their Large Language Model applications using the "Golden Triad" of evaluation: **Faithfulness**, **Answer Relevance**, and **Completeness**. It is built for speed with native asynchronous support.

---

## 🚀 Features

- **⚡ Async-First Architecture:** Evaluate hundreds of test cases in parallel using Python's `asyncio`.
- **🛡️ Hallucination Detection:** The `Faithfulness` metric strictly checks if the output is supported by your retrieval context.
- **🎯 Intent Verification:** The `Completeness` metric ensures the model answered *every part* of the user's complex query.
- **🔌 Drop-in Ready:** Compatible with OpenAI (GPT-4, GPT-3.5) out of the box.
- **✅ Pydantic Validation:** Type-safe data handling to prevent runtime errors.

---

## 📦 Installation

Install the latest version from PyPI:

```bash
pip install nexuseval

```

*Note: You must have an `OPENAI_API_KEY` set in your environment variables.*

```bash
export OPENAI_API_KEY="sk-..."

```

---

## ⚡ Quick Start

Here is how to evaluate a RAG response in 30 seconds.

```python
import asyncio
from nexuseval import TestCase, Evaluator, Faithfulness, AnswerRelevance

# 1. Define your test case
# (This simulates a user asking a question and your RAG system answering)
case = TestCase(
    input_text="What is the capital of Mars?",
    actual_output="Mars has no capital city because it has no government.",
    retrieval_context=[
        "Mars is the fourth planet from the Sun.",
        "Mars is uninhabited and has no political structure."
    ]
)

# 2. Select the metrics you want to run
evaluator = Evaluator(metrics=[
    Faithfulness(),      # Checks for hallucinations
    AnswerRelevance()    # Checks if the answer matches the question
])

# 3. Run the evaluation
results = evaluator.evaluate([case])

# 4. View results
print(results)

```

---

## 📊 Supported Metrics

NexusEval focuses on the **RAG Triad** standard.

| Metric | What it Measures | Why use it? |
| --- | --- | --- |
| **Faithfulness** | **Hallucinations.** Does the answer contain information *not* present in the retrieved context? | Essential for RAG. Prevents the model from making up facts. |
| **Answer Relevance** | **Focus.** Does the answer actually address the user's query? | Ensures the model isn't rambling or dodging the question. |
| **Completeness** | **Coverage.** Did the answer address *all* constraints and sub-questions in the prompt? | Critical for complex queries (e.g., "Give me pros AND cons"). |

### Using the Completeness Metric

Useful for checking if the model missed instructions.

```python
from nexuseval import Completeness

# User asked for TWO things, but Model gave ONE.
bad_case = TestCase(
    input_text="Who is the CEO of Tesla and Space X?",
    actual_output="The CEO of Tesla is Elon Musk." # Missed SpaceX
)

evaluator = Evaluator(metrics=[Completeness()])
results = evaluator.evaluate([bad_case])

# Result: Score will be low (~0.5) with a reason: "Failed to mention SpaceX."

```

---

## 🛠️ Advanced Usage

### Running Bulk Evaluations (Async)

The `Evaluator` automatically uses `asyncio` to run metrics in parallel. You don't need to change your code—just pass a list of 100+ cases, and it will process them concurrently.

```python
# Create a list of 50 test cases
cases = [TestCase(...) for _ in range(50)]

# NexusEval will show a progress bar and finish quickly
results = evaluator.evaluate(cases)

```

### Customizing the Judge Model

By default, NexusEval uses `gpt-4-turbo`. You can change this to save costs (e.g., `gpt-4o-mini`).

```python
from nexuseval.metrics.standard import Faithfulness

# Use a cheaper/faster model for evaluation
metric = Faithfulness()
metric.llm.model = "gpt-4o-mini"

```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/NewMetric`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.
