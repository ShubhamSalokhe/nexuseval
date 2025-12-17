"""
Unit tests for dataset management.
"""

import pytest
import json
import csv
from pathlib import Path
from nexuseval.dataset import (
    Dataset,
    DatasetLoader,
    DatasetValidator,
    SampleDataGenerator
)
from nexuseval.core import TestCase

@pytest.fixture
def sample_test_cases():
    """Fixture providing sample test cases."""
    return [
        TestCase(
            input_text="What is Python?",
            actual_output="A programming language",
            retrieval_context=["Python is used for AI"]
        ),
        TestCase(
            input_text="What is Java?",
            actual_output="A programming language",
            retrieval_context=["Java is used for enterprise apps"]
        )
    ]

@pytest.fixture
def sample_dataset(sample_test_cases):
    """Fixture providing a sample dataset."""
    return Dataset(
        test_cases=sample_test_cases,
        name="test_dataset",
        description="Test dataset"
    )


class TestDataset:
    """Test Dataset model."""
    
    def test_creation(self, sample_test_cases):
        """Test creating a dataset."""
        dataset = Dataset(test_cases=sample_test_cases)
        assert len(dataset) == 2
        assert dataset[0].input_text == "What is Python?"
    
    def test_empty_dataset_fails(self):
        """Test that empty dataset raises error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Dataset(test_cases=[])
    
    def test_dataset_split(self, sample_dataset):
        """Test splitting dataset."""
        # Create larger dataset for meaningful split
        large_dataset = SampleDataGenerator.generate_rag_samples(n=10)
        train, test = large_dataset.split(train_ratio=0.8, shuffle=False)
        
        assert len(train) == 8
        assert len(test) == 2
        assert train.name == f"{large_dataset.name}_train"


class TestDatasetLoader:
    """Test DatasetLoader functionality."""
    
    def test_from_list_with_test_cases(self, sample_test_cases):
        """Test creating dataset from list of TestCase objects."""
        dataset = DatasetLoader.from_list(
            sample_test_cases,
            name="my_dataset"
        )
        assert len(dataset) == 2
        assert dataset.name == "my_dataset"
    
    def test_from_list_with_dicts(self):
        """Test creating dataset from list of dictionaries."""
        data = [
            {
                "input_text": "Question 1",
                "actual_output": "Answer 1"
            },
            {
                "input_text": "Question 2",
                "actual_output": "Answer 2"
            }
        ]
        dataset = DatasetLoader.from_list(data, name="dict_dataset")
        assert len(dataset) == 2
        assert isinstance(dataset[0], TestCase)
    
    def test_from_json(self, tmp_path, sample_test_cases):
        """Test loading dataset from JSON file."""
        # Create JSON file
        json_file = tmp_path / "test.json"
        data = {
            "name": "json_dataset",
            "test_cases": [tc.model_dump() for tc in sample_test_cases]
        }
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        # Load dataset
        dataset = DatasetLoader.from_json(json_file)
        assert len(dataset) == 2
        assert dataset.name == "json_dataset"
    
    def test_from_jsonl(self, tmp_path):
        """Test loading dataset from JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"input_text": "Q1", "actual_output": "A1"}) + "\n")
            f.write(json.dumps({"input_text": "Q2", "actual_output": "A2"}) + "\n")
        
        dataset = DatasetLoader.from_json(jsonl_file, is_jsonl=True)
        assert len(dataset) == 2
    
    def test_from_csv(self, tmp_path):
        """Test loading dataset from CSV file."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer", "context"])
            writer.writeheader()
            writer.writerow({
                "question": "What is AI?",
                "answer": "Artificial Intelligence",
                "context": "AI simulates human intelligence"
            })
        
        dataset = DatasetLoader.from_csv(
            csv_file,
            column_mapping={
                "question": "input_text",
                "answer": "actual_output",
                "context": "retrieval_context"
            }
        )
        assert len(dataset) == 1
        assert dataset[0].input_text == "What is AI?"
    
    def test_from_dict(self, sample_test_cases):
        """Test creating dataset from dictionary."""
        data = {
            "name": "dict_dataset",
            "description": "Test",
            "test_cases": [tc.model_dump() for tc in sample_test_cases]
        }
        dataset = DatasetLoader.from_dict(data)
        assert len(dataset) == 2
        assert dataset.name == "dict_dataset"


class TestDatasetValidator:
    """Test DatasetValidator functionality."""
    
    def test_validate_schema_valid(self, sample_dataset):
        """Test validation of valid dataset."""
        validator = DatasetValidator()
        issues = validator.validate_schema(sample_dataset)
        assert len(issues) == 0
    
    def test_validate_schema_empty_input(self):
        """Test validation catches empty input_text."""
        dataset = Dataset(
            test_cases=[
                TestCase(input_text="", actual_output="Some output")
            ]
        )
        validator = DatasetValidator()
        issues = validator.validate_schema(dataset)
        assert len(issues) > 0
        assert "Empty input_text" in issues[0]
    
    def test_validate_requires_context(self):
        """Test validation for required context."""
        dataset = Dataset(
            test_cases=[
                TestCase(
                    input_text="Question",
                    actual_output="Answer"
                    # No retrieval_context
                )
            ]
        )
        validator = DatasetValidator()
        issues = validator.validate_schema(dataset, require_context=True)
        assert len(issues) > 0
        assert "Missing retrieval_context" in issues[0]
    
    def test_check_duplicates(self):
        """Test duplicate detection."""
        dataset = Dataset(
            test_cases=[
                TestCase(input_text="Question 1", actual_output="Answer 1"),
                TestCase(input_text="Question 2", actual_output="Answer 2"),
                TestCase(input_text="Question 1", actual_output="Answer 3"),  # Duplicate
            ]
        )
        validator = DatasetValidator()
        duplicates = validator.check_duplicates(dataset)
        assert len(duplicates) == 1
        assert duplicates[0] == (0, 2)


class TestSampleDataGenerator:
    """Test sample data generation."""
    
    def test_generate_rag_samples(self):
        """Test generating sample RAG data."""
        dataset = SampleDataGenerator.generate_rag_samples(n=5)
        assert len(dataset) == 5
        assert dataset.name == "sample_dataset"
        
        # Check that test cases have required fields
        for tc in dataset.test_cases:
            assert tc.input_text
            assert tc.actual_output
            assert tc.retrieval_context
            assert tc.test_case_id
