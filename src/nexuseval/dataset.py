"""
Dataset management for NexusEval.

This module provides utilities for loading, validating, and managing
evaluation datasets from various sources and formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from .core import TestCase

class Dataset(BaseModel):
    """
    Container for evaluation test cases with metadata.
    
    Attributes:
        test_cases: List of TestCase objects to evaluate
        metadata: Dataset-level metadata (name, version, description, etc.)
        name: Dataset name for identification
        description: Human-readable description
    """
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None
    description: Optional[str] = None
    
    @validator('test_cases')
    def validate_non_empty(cls, v):
        if not v:
            raise ValueError("Dataset must contain at least one test case")
        return v
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __getitem__(self, idx: int) -> TestCase:
        return self.test_cases[idx]
    
    def split(self, train_ratio: float = 0.8, shuffle: bool = True) -> tuple['Dataset', 'Dataset']:
        """
        Split dataset into train and test sets.
        
        Args:
            train_ratio: Ratio of training data (0.0 to 1.0)
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random
        
        cases = self.test_cases.copy()
        if shuffle:
            random.shuffle(cases)
        
        split_idx = int(len(cases) * train_ratio)
        train_cases = cases[:split_idx]
        test_cases = cases[split_idx:]
        
        train_dataset = Dataset(
            test_cases=train_cases,
            metadata={**self.metadata, "split": "train"},
            name=f"{self.name}_train" if self.name else None,
            description=f"Training split of {self.name or 'dataset'}"
        )
        
        test_dataset = Dataset(
            test_cases=test_cases,
            metadata={**self.metadata, "split": "test"},
            name=f"{self.name}_test" if self.name else None,
            description=f"Test split of {self.name or 'dataset'}"
        )
        
        return train_dataset, test_dataset


class DatasetLoader:
    """
    Factory class for loading datasets from various sources.
    
    Supported formats:
    - JSON/JSONL (with automatic schema mapping)
    - CSV (with configurable column mapping)
    - Python lists/dicts
    """
    
    @staticmethod
    def from_json(
        path: Union[str, Path],
        is_jsonl: bool = False,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Dataset:
        """
        Load dataset from JSON or JSONL file.
        
        Args:
            path: Path to JSON/JSONL file
            is_jsonl: Whether file is JSONL (one JSON object per line)
            column_mapping: Optional mapping of file columns to TestCase fields
                          e.g., {"question": "input_text", "answer": "actual_output"}
        
        Returns:
            Dataset object
            
        Example:
            >>> dataset = DatasetLoader.from_json("evals.json")
            >>> dataset = DatasetLoader.from_json("evals.jsonl", is_jsonl=True)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Load data
        if is_jsonl:
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # If root is dict, look for test_cases key or use as single case
            if "test_cases" in data:
                test_cases_data = data["test_cases"]
                metadata = {k: v for k, v in data.items() if k != "test_cases"}
            else:
                test_cases_data = [data]
                metadata = {}
        elif isinstance(data, list):
            test_cases_data = data
            metadata = {}
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")
        
        # Apply column mapping if provided
        if column_mapping:
            test_cases_data = [
                {column_mapping.get(k, k): v for k, v in case.items()}
                for case in test_cases_data
            ]
        
        # Convert to TestCase objects
        test_cases = [TestCase(**case) for case in test_cases_data]
        
        return Dataset(
            test_cases=test_cases,
            metadata=metadata,
            name=metadata.get("name", path.stem),
            description=metadata.get("description")
        )
    
    @staticmethod
    def from_csv(
        path: Union[str, Path],
        column_mapping: Dict[str, str],
        **csv_kwargs
    ) -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            path: Path to CSV file
            column_mapping: Required mapping of CSV columns to TestCase fields
                          e.g., {"Question": "input_text", "Answer": "actual_output"}
            **csv_kwargs: Additional arguments passed to csv.DictReader
        
        Returns:
            Dataset object
            
        Example:
            >>> dataset = DatasetLoader.from_csv(
            ...     "evals.csv",
            ...     column_mapping={"query": "input_text", "response": "actual_output"}
            ... )
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        test_cases = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, **csv_kwargs)
            for row in reader:
                # Map columns
                mapped_row = {column_mapping.get(k, k): v for k, v in row.items()}
                
                # Handle special fields that might be comma-separated strings
                if 'retrieval_context' in mapped_row and isinstance(mapped_row['retrieval_context'], str):
                    mapped_row['retrieval_context'] = [
                        ctx.strip() for ctx in mapped_row['retrieval_context'].split('|')
                        if ctx.strip()
                    ]
                
                test_cases.append(TestCase(**mapped_row))
        
        return Dataset(
            test_cases=test_cases,
            name=path.stem,
            description=f"Dataset loaded from {path.name}"
        )
    
    @staticmethod
    def from_list(
        test_cases: List[Union[TestCase, Dict[str, Any]]],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dataset:
        """
        Create dataset from Python list.
        
        Args:
            test_cases: List of TestCase objects or dicts
            name: Optional dataset name
            description: Optional description
        
        Returns:
            Dataset object
        """
        # Convert dicts to TestCase if needed
        cases = [
            tc if isinstance(tc, TestCase) else TestCase(**tc)
            for tc in test_cases
        ]
        
        return Dataset(
            test_cases=cases,
            name=name or "custom_dataset",
            description=description
        )
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Dataset:
        """
        Create dataset from dictionary.
        
        Args:
            data: Dictionary with 'test_cases' key and optional metadata
        
        Returns:
            Dataset object
        """
        test_cases_data = data.get("test_cases", [])
        metadata = {k: v for k, v in data.items() if k != "test_cases"}
        
        test_cases = [
            tc if isinstance(tc, TestCase) else TestCase(**tc)
            for tc in test_cases_data
        ]
        
        return Dataset(
            test_cases=test_cases,
            metadata=metadata,
            name=metadata.get("name"),
            description=metadata.get("description")
        )


class DatasetValidator:
    """
    Validator for ensuring dataset quality and consistency.
    """
    
    @staticmethod
    def validate_schema(dataset: Dataset, require_context: bool = False) -> List[str]:
        """
        Validate dataset schema and return list of warnings/errors.
        
        Args:
            dataset: Dataset to validate
            require_context: Whether retrieval_context is required
        
        Returns:
            List of validation messages (empty if valid)
        """
        issues = []
        
        for idx, test_case in enumerate(dataset.test_cases):
            # Check required fields
            if not test_case.input_text.strip():
                issues.append(f"Test case {idx}: Empty input_text")
            
            if not test_case.actual_output.strip():
                issues.append(f"Test case {idx}: Empty actual_output")
            
            if require_context and not test_case.retrieval_context:
                issues.append(f"Test case {idx}: Missing retrieval_context")
        
        return issues
    
    @staticmethod
    def check_duplicates(dataset: Dataset) -> List[tuple[int, int]]:
        """
        Find duplicate test cases based on input_text.
        
        Returns:
            List of (idx1, idx2) tuples indicating duplicates
        """
        duplicates = []
        seen = {}
        
        for idx, test_case in enumerate(dataset.test_cases):
            key = test_case.input_text.strip().lower()
            if key in seen:
                duplicates.append((seen[key], idx))
            else:
                seen[key] = idx
        
        return duplicates


# Example dataset generator for testing/demos
class SampleDataGenerator:
    """
    Generate sample datasets for testing and demonstrations.
    """
    
    @staticmethod
    def generate_rag_samples(n: int = 10) -> Dataset:
        """
        Generate sample RAG evaluation test cases.
        
        Args:
            n: Number of samples to generate
        
        Returns:
            Dataset with synthetic test cases
        """
        samples = [
            {
                "input_text": "What is the capital of France?",
                "actual_output": "The capital of France is Paris.",
                "retrieval_context": ["France is a country in Europe.", "Paris is the capital city of France."],
                "expected_output": "Paris is the capital of France.",
            },
            {
                "input_text": "Who wrote Romeo and Juliet?",
                "actual_output": "William Shakespeare wrote Romeo and Juliet.",
                "retrieval_context": ["Shakespeare was an English playwright.", "Romeo and Juliet is a famous tragedy."],
            },
            {
                "input_text": "What is photosynthesis?",
                "actual_output": "Photosynthesis is the process by which plants convert light into energy.",
                "retrieval_context": ["Plants use chlorophyll to absorb light.", "Photosynthesis produces oxygen."],
            },
        ]
        
        # Cycle through samples to reach n
        test_cases = []
        for i in range(n):
            sample = samples[i % len(samples)].copy()
            sample["test_case_id"] = f"sample_{i:03d}"
            test_cases.append(TestCase(**sample))
        
        return Dataset(
            test_cases=test_cases,
            name="sample_dataset",
            description=f"Sample dataset with {n} test cases for demonstration"
        )
