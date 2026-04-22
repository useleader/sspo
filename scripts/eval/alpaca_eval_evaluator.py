"""AlpacaEval evaluator for SSPO reproduction.

Paper reference: AlpacaEval uses Length-Controlled Win Rate (LC-Win Rate)
evaluated by GPT-4 as judge. Paper Table 1 shows SSPO achieves 32.4% on Mistral.

Reference: https://github.com/tatsu-lab/alpaca_eval
"""
import json
from pathlib import Path
from typing import Optional


def evaluate_alpacaeval(
    model_outputs_path: str,
    output_dir: Optional[str] = None,
    reference_output: str = "alpaca_eval_gpt4_turbo",
) -> dict:
    """Evaluate model outputs using AlpacaEval with LC-Win Rate.

    Args:
        model_outputs_path: Path to JSON file with model responses
        output_dir: Directory to save evaluation results
        reference_output: Reference model name for comparison

    Returns:
        Dictionary with evaluation metrics including LC-Win Rate

    Paper metrics (from page 8):
        - AlpacaEval samples: 805
        - Max length: 2048 tokens
        - Judge model: GPT-4
    """
    try:
        from alpaca_eval import AlpacaEvaluator
    except ImportError:
        raise ImportError(
            "alpaca_eval not installed. Install with: pip install alpaca_eval"
        )

    evaluator = AlpacaEvaluator(
        model_outputs=model_outputs_path,
        reference_outputs=reference_output,
        output_dir=output_dir,
    )

    results = evaluator.evaluate()

    return {
        "lc_win_rate": results["lc_win_rate"],
        "win_rate": results["win_rate"],
        "length": results["length"],
        "n_samples": 805,
    }


def evaluate_alpacaeval_cli(model_outputs_path: str, output_dir: str) -> None:
    """CLI for AlpacaEval evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="AlpacaEval evaluator for SSPO")
    parser.add_argument("--model-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = evaluate_alpacaeval(args.model_outputs, args.output_dir)

    print(f"LC-Win Rate: {results['lc_win_rate']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Avg Length: {results['length']:.1f} tokens")


if __name__ == "__main__":
    evaluate_alpacaeval_cli()