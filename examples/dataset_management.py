"""
Example: Creating and loading custom datasets.

This example demonstrates:
1. Creating datasets from various formats
2. Validating datasets
3. Splitting datasets for train/test
4. Saving and loading datasets
"""

import json
from pathlib import Path
from nexuseval import (
    Dataset,
    DatasetLoader,
    DatasetValidator,
    TestCase
)

# Example 1: Create dataset from Python dict/list
print("=" * 60)
print("Example 1: Creating dataset from Python list")
print("=" * 60)

test_cases = [
    {
        "input_text": "What is Python?",
        "actual_output": "Python is a high-level programming language.",
        "retrieval_context": ["Python is popular for AI and web development."],
        "expected_output": "Python is a programming language.",
    },
    {
        "input_text": "Who created Python?",
        "actual_output": "Python was created by Guido van Rossum.",
        "retrieval_context": ["Guido van Rossum started Python in 1991."],
    }
]

dataset = DatasetLoader.from_list(
    test_cases,
    name="python_qa",
    description="Questions about Python programming language"
)

print(f"✅ Created dataset with {len(dataset)} test cases")
print(f"Dataset name: {dataset.name}\n")

# Example 2: Validate dataset
print("=" * 60)
print("Example 2: Validating dataset")
print("=" * 60)

validator = DatasetValidator()
issues = validator.validate_schema(dataset, require_context=True)

if issues:
    print(f"⚠️  Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Dataset is valid!")

# Check for duplicates
duplicates = validator.check_duplicates(dataset)
if duplicates:
    print(f"\n⚠️  Found {len(duplicates)} duplicate(s)")
else:
    print("✅ No duplicates found\n")

# Example 3: Split dataset
print("=" * 60)
print("Example 3: Splitting dataset into train/test")
print("=" * 60)

# Create a larger dataset first
from nexuseval import SampleDataGenerator
large_dataset = SampleDataGenerator.generate_rag_samples(n=20)

train_set, test_set = large_dataset.split(train_ratio=0.8, shuffle=True)

print(f"Original dataset: {len(large_dataset)} cases")
print(f"Train set: {len(train_set)} cases")
print(f"Test set: {len(test_set)} cases\n")

# Example 4: Save and load dataset
print("=" * 60)
print("Example 4: Saving and loading datasets")
print("=" * 60)

# Save as JSON
output_dir = Path("sample_data")
output_dir.mkdir(exist_ok=True)

# Save dataset
output_file = output_dir / "my_dataset.json"
dataset_dict = {
    "name": dataset.name,
    "description": dataset.description,
    "metadata": dataset.metadata,
    "test_cases": [tc.model_dump() for tc in dataset.test_cases]
}

with open(output_file, 'w') as f:
    json.dump(dataset_dict, f, indent=2)

print(f"✅ Saved dataset to: {output_file}")

# Load it back
loaded_dataset = DatasetLoader.from_json(output_file)
print(f"✅ Loaded dataset: {len(loaded_dataset)} test cases")
print(f"Dataset name: {loaded_dataset.name}\n")

# Example 5: Create dataset from CSV format
print("=" * 60)
print("Example 5: Creating CSV dataset")
print("=" * 60)

# Create a sample CSV file
csv_file = output_dir / "questions.csv"
with open(csv_file, 'w') as f:
    f.write("question,answer,context\n")
    f.write("What is AI?,Artificial Intelligence,AI is the simulation of human intelligence\n")
    f.write("What is ML?,Machine Learning,ML is a subset of AI\n")

# Load with column mapping
csv_dataset = DatasetLoader.from_csv(
    csv_file,
    column_mapping={
        "question": "input_text",
        "answer": "actual_output",
        "context": "retrieval_context"
    }
)

print(f"✅ Loaded CSV dataset: {len(csv_dataset)} test cases")
for i, tc in enumerate(csv_dataset.test_cases):
    print(f"\nTest Case {i+1}:")
    print(f"  Input: {tc.input_text}")
    print(f"  Output: {tc.actual_output}")
    print(f"  Context: {tc.retrieval_context}")

print("\n" + "=" * 60)
print("✅ All examples completed!")
print("=" * 60)
