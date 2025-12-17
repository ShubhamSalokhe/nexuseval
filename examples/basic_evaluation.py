"""
Example: Basic evaluation with new dataset management and caching.

This example demonstrates:
1. Loading a dataset from JSON
2. Creating an evaluator with caching enabled
3. Running evaluation
4. Viewing cache statistics and cost tracking
"""

import asyncio
from nexuseval import (
    Dataset,
    DatasetLoader,
    Evaluator,
    Faithfulness,
    AnswerRelevance,
    Completeness,
    NexusConfig
)

async def main():
    # Option 1: Load dataset from file
    # dataset = DatasetLoader.from_json("your_dataset.json")
    
    # Option 2: Use sample data generator for demo
    from nexuseval import SampleDataGenerator
    dataset = SampleDataGenerator.generate_rag_samples(n=5)
    
    print(f"📊 Loaded dataset: {len(dataset)} test cases")
    print(f"Dataset name: {dataset.name}\n")
    
    # Create configuration with caching and cost tracking
    config = NexusConfig.preset_development()  # Uses gpt-4o-mini, file cache, cost tracking
    
    # Or create custom config
    # config = NexusConfig(
    #     llm=LLMConfig(model="gpt-4o-mini", provider="openai"),
    #     cache=CacheConfig(enabled=True, backend="file"),
    #     evaluation=EvaluationConfig(enable_cost_tracking=True)
    # )
    
    # Create evaluator
    evaluator = Evaluator(
        metrics=[
            Faithfulness(),
            AnswerRelevance(),
            Completeness()
        ]
    )
    
    print("🚀 Running evaluation (first run - will call API)...")
    results_1 = evaluator.evaluate(dataset.test_cases)
    
    print("\n" + "="*60)
    print("📈 RESULTS (First Run)")
    print("="*60)
    for result in results_1[:2]:  # Show first 2
        print(f"\n❓ Input: {result['input']}")
        print(f"💬 Output: {result['output']}")
        print("Metrics:")
        for metric_name, metric_data in result['metrics'].items():
            print(f"  • {metric_name}: {metric_data['score']:.2f} - {metric_data['reason']}")
    
    # Run again to demonstrate caching
    print("\n" + "="*60)
    print("🚀 Running evaluation again (should use cache)...")
    print("="*60)
    results_2 = evaluator.evaluate(dataset.test_cases)
    
    print("\n✅ Second run completed (faster due to cache)")
    
    # Show cache statistics
    if hasattr(evaluator.metrics[0], 'llm') and hasattr(evaluator.metrics[0].llm, 'cache_manager'):
        cache_stats = evaluator.metrics[0].llm.cache_manager.get_stats()
        print(f"\n📦 Cache Statistics:")
        print(f"  Backend: {cache_stats['backend']}")
        print(f"  Hits: {cache_stats.get('hits', 0)}")
        print(f"  Misses: {cache_stats.get('misses', 0)}")
        print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
    
    # Show cost tracking (if enabled)
    if hasattr(evaluator.metrics[0], 'llm') and hasattr(evaluator.metrics[0].llm, 'get_cost_stats'):
        cost_stats = evaluator.metrics[0].llm.get_cost_stats()
        print(f"\n💰 Cost Statistics:")
        print(f"  Total Cost: ${cost_stats['total_cost_usd']:.4f}")
        print(f"  Total Tokens: {cost_stats['total_tokens']:,}")
        print(f"  Model: {cost_stats['model']}")

if __name__ == "__main__":
    asyncio.run(main())
