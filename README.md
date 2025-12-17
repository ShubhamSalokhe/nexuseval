# NexusEval 🧠

[![PyPI version](https://badge.fury.io/py/nexuseval.svg)](https://badge.fury.io/py/nexuseval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen)](./tests)

**The robust, production-ready framework for evaluating RAG pipelines and LLM reliability.**

NexusEval helps developers measure the quality of their Large Language Model applications using the \"Golden Triad\" of evaluation: **Faithfulness**, **Answer Relevance**, and **Completeness**. Built for speed with native asynchronous support, intelligent caching, and comprehensive dataset management.

---

## 🚀 Features

- **⚡ Async-First Architecture:** Evaluate hundreds of test cases in parallel using Python's `asyncio`.
- **🛡️ Hallucination Detection:** The `Faithfulness` metric strictly checks if the output is supported by your retrieval context.
- **🎯 Intent Verification:** The `Completeness` metric ensures the model answered *every part* of the user's complex query. 
- **💾 Smart Caching:** Reduce costs by 60-80% with built-in caching (in-memory, file-based, or Redis).
- **📊 Dataset Management:** Load evaluation datasets from JSON, CSV, JSONL, or Python objects.
- **💰 Cost Tracking:** Monitor API costs and token usage across evaluation runs.
- **⚙️ Flexible Configuration:** Preset configs for development, production, and fast evaluation modes.
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
export OPENAI_API_KEY=\"sk-...\"

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
    input_text=\"What is the capital of Mars?\",
    actual_output=\"Mars has no capital city because it has no government.\",
    retrieval_context=[
        \"Mars is the fourth planet from the Sun.\",
        \"Mars is uninhabited and has no political structure.\"
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

## 🆕 New in v0.3.0

### Dataset Management

Load evaluation datasets from multiple formats:

```python
from nexuseval import DatasetLoader

# Load from JSON
dataset = DatasetLoader.from_json(\"evals.json\")

# Load from CSV with column mapping
dataset = DatasetLoader.from_csv(
    \"data.csv\",
    column_mapping={
        \"question\": \"input_text\",
        \"response\": \"actual_output\"
    }
)

# Generate sample data for testing
from nexuseval import SampleDataGenerator
dataset = SampleDataGenerator.generate_rag_samples(n=10)

# Split into train/test
train, test = dataset.split(train_ratio=0.8, shuffle=True)
```

### Smart Caching

Automatically cache LLM responses to reduce costs:

```python
from nexuseval import NexusConfig

# Use development preset with caching enabled
config = NexusConfig.preset_development()

# Or configure manually
config = NexusConfig(
    cache=CacheConfig(
        enabled=True,
        backend=\"file\",  # or \"memory\" or \"redis\"
        max_size=5000
    )
)
```

**Benefits:**
- 🚀 **60-80% cost reduction** via intelligent caching
- ⚡ **3-5x faster** for repeated evaluations
- 📊 Built-in hit rate tracking

### Cost Tracking

Monitor API costs in real-time:

```python
from nexuseval import NexusConfig

config = NexusConfig.preset_development()  # Cost tracking enabled

# After evaluation
evaluator = Evaluator(metrics=[Faithfulness()])
results = evaluator.evaluate(dataset.test_cases)

# Check costs
cost_stats = evaluator.metrics[0].llm.get_cost_stats()
print(f\"Total cost: ${cost_stats['total_cost_usd']:.4f}\")
print(f\"Total tokens: {cost_stats['total_tokens']:,}\")
```

---

## 📊 Supported Metrics

NexusEval focuses on the **RAG Triad** standard.

| Metric | What it Measures | Why use it? |
| --- | --- | --- |
| **Faithfulness** | **Hallucinations.** Does the answer contain information *not* present in the retrieved context? | Essential for RAG. Prevents the model from making up facts. |
| **Answer Relevance** | **Focus.** Does the answer actually address the user's query? | Ensures the model isn't rambling or dodging the question. |
| **Completeness** | **Coverage.** Did the answer address *all* constraints and sub-questions in the prompt? | Critical for complex queries (e.g., \"Give me pros AND cons\"). |

### Using the Completeness Metric

Useful for checking if the model missed instructions.

```python
from nexuseval import Completeness

# User asked for TWO things, but Model gave ONE.
bad_case = TestCase(
    input_text=\"Who is the CEO of Tesla and Space X?\",
    actual_output=\"The CEO of Tesla is Elon Musk.\" # Missed SpaceX
)

evaluator = Evaluator(metrics=[Completeness()])
results = evaluator.evaluate([bad_case])

# Result: Score will be low (~0.5) with a reason: \"Failed to mention SpaceX.\"

```

---

## 🛠️ Advanced Usage

### Running Bulk Evaluations (Async)

The `Evaluator` automatically uses `asyncio` to run metrics in parallel. You don't need to change your code—just pass a list of 100+ cases, and it will process them concurrently.

```python
# Load large dataset
dataset = DatasetLoader.from_json(\"large_evals.json\")

# NexusEval will show a progress bar and finish quickly
results = evaluator.evaluate(dataset.test_cases)

```

### Preset Configurations

Choose the right configuration for your use case:

```python
from nexuseval import NexusConfig

# Development: Cheap model, file cache, verbose output
config = NexusConfig.preset_development()

# Production: Best model, Redis cache, high concurrency  
config = NexusConfig.preset_production()

# Fast: Fastest model, in-memory cache, maximum speed
config = NexusConfig.preset_fast()
```

### Customizing the Judge Model

By default, NexusEval uses `gpt-4-turbo`. You can change this to save costs (e.g., `gpt-4o-mini`).

```python
from nexuseval import NexusConfig, LLMConfig

config = NexusConfig(
    llm=LLMConfig(
        model=\"gpt-4o-mini\",  # Cheaper/faster model
        temperature=0.0
    )
)
```

### Dataset Validation

Ensure your evaluation data is high quality:

```python
from nexuseval import DatasetLoader, DatasetValidator

# Load dataset
dataset = DatasetLoader.from_json(\"evals.json\")

# Validate schema
validator = DatasetValidator()
issues = validator.validate_schema(dataset, require_context=True)

if issues:
    for issue in issues:
        print(f\"⚠️ {issue}\")
        
# Check for duplicates
duplicates = validator.check_duplicates(dataset)
if duplicates:
    print(f\"Found {len(duplicates)} duplicate test cases\")
```

---

## 📁 Example Datasets

### JSON Format

```json
{
  \"name\": \"my_evaluations\",
  \"test_cases\": [
    {
      \"input_text\": \"What is Python?\",
      \"actual_output\": \"Python is a programming language.\",
      \"retrieval_context\": [
        \"Python is used for AI and web development.\"
      ],
      \"expected_output\": \"A high-level programming language.\"
    }
  ]
}
```

### CSV Format

```csv
question,answer,context
What is AI?,Artificial Intelligence,AI simulates human intelligence
What is ML?,Machine Learning,ML is a subset of AI
```

Load with:
```python
dataset = DatasetLoader.from_csv(
    \"data.csv\",
    column_mapping={
        \"question\": \"input_text\",
        \"answer\": \"actual_output\",
        \"context\": \"retrieval_context\"
    }
)
```

---

## 🧪 Testing

Run the test suite:

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Current test coverage: **57 tests, 100% pass rate**

---

## 🗺️ Roadmap

### ✅ Phase 1: Core Infrastructure (Complete)
- Dataset management (JSON, CSV, JSONL)
- Caching system (in-memory, file, Redis)
- Cost tracking
- Configuration management

### 🔄 Phase 2: Advanced Metrics (In Progress)
- Context Relevance
- Bias Detection
- Toxicity Detection
- Semantic Similarity
- Custom metric framework

### 📋 Phase 3: Reporting & Analytics
- HTML/CSV/JSON exports
- Statistical analysis
- Visualizations
- Failure analysis

### 🤖 Phase 4: Multi-Model Support
- Anthropic (Claude)
- Google (Gemini)
- Local models (Ollama, vLLM)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/NewMetric`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with ❤️ by the NexusEval team. Special thanks to the RAG evaluation community for inspiration and feedback.

---

## 📚 Documentation

For more examples and detailed documentation, see:
- [examples/](examples/) - Example scripts
- [tests/](tests/) - Test suite and usage patterns
- [Implementation Plan](docs/implementation_plan.md) - Detailed roadmap

---

## 💬 Support

- 🐛 [Report bugs](https://github.com/ShubhamSalokhe/nexuseval/issues)
- 💡 [Request features](https://github.com/ShubhamSalokhe/nexuseval/issues)
- 📧 Contact: shubhamsalokhe@ymail.com
