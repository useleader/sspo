"""MT-Bench evaluator for SSPO reproduction.

Paper reference: MT-Bench evaluates multi-turn dialogue quality across 8 categories.
Paper Table 2 (page 8) shows SSPO results per category.

Categories: reasoning, math, coding, writing, roleplay, extraction, STEM, humanities
"""
from typing import Optional


def evaluate_mtbench(model_outputs_path: str, output_dir: Optional[str] = None) -> dict:
    """Evaluate model outputs using MT-Bench.

    Args:
        model_outputs_path: Path to JSON file with model responses
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with per-category scores and average

    Paper metrics (from page 8, Table 2):
        - MT-Bench categories: 8
        - First turn evaluation only
        - GPT-4 as judge
    """
    try:
        from mtbench import MTBenchEvaluator
    except ImportError:
        raise ImportError(
            "mtbench not installed. Install with: pip install mtbench"
        )

    evaluator = MTBenchEvaluator(output_dir=output_dir)
    results = evaluator.evaluate(model_outputs_path)

    category_scores = {}
    total_score = 0.0

    for category, score in results.items():
        category_scores[category] = score
        total_score += score

    avg_score = total_score / 8

    return {
        "categories": category_scores,
        "average_score": avg_score,
        "reasoning": category_scores.get("reasoning", 0.0),
        "math": category_scores.get("math", 0.0),
        "coding": category_scores.get("coding", 0.0),
        "writing": category_scores.get("writing", 0.0),
    }


def evaluate_mtbench_cli(model_outputs_path: str, output_dir: str) -> None:
    """CLI for MT-Bench evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="MT-Bench evaluator for SSPO")
    parser.add_argument("--model-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = evaluate_mtbench(args.model_outputs, args.output_dir)

    print("MT-Bench Results:")
    for cat, score in results["categories"].items():
        print(f"  {cat}: {score:.2f}")
    print(f"Average: {results['average_score']:.2f}")


if __name__ == "__main__":
    evaluate_mtbench_cli()