"""AlpacaEval evaluator for SSPO reproduction.

Paper reference: AlpacaEval uses Length-Controlled Win Rate (LC-Win Rate)
evaluated by GPT-4 as judge. Paper Table 1 shows SSPO achieves 32.4% on Mistral.

Reference: https://github.com/tatsu-lab/alpaca_eval

Supports custom judge models via environment variables:
- JUDGE_MODEL: Model name (e.g., gpt-4o)
- OPENAI_API_BASE: API base URL (e.g., https://aihubmix.com/v1)
- OPENAI_API_KEY: API key
"""
import json
import os
from pathlib import Path
from typing import Optional

# Clear proxy settings to avoid connection issues with custom API endpoints
for _var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(_var, None)

# Load environment variables from .env file
from dotenv import load_dotenv
# Find .env in project root (parent of scripts/eval/)
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent
load_dotenv(_project_root / ".env")


def evaluate_alpacaeval(
    model_outputs_path: str,
    output_dir: Optional[str] = None,
    reference_output: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> dict:
    """Evaluate model outputs using AlpacaEval with LC-Win Rate.

    Args:
        model_outputs_path: Path to JSON file with model responses
        output_dir: Directory to save evaluation results
        reference_output: Path to reference outputs JSON file (optional)
        judge_model: Custom judge model (overrides env var JUDGE_MODEL)

    Returns:
        Dictionary with evaluation metrics including LC-Win Rate

    Paper metrics (from page 8):
        - AlpacaEval samples: 805
        - Max length: 2048 tokens
        - Judge model: GPT-4
    """
    from alpaca_eval import evaluate

    # Configure API from environment
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model = judge_model or os.getenv("JUDGE_MODEL", "gpt-4")

    # Set environment for API access
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    print(f"Using judge model: {model}")
    print(f"API base: {api_base or 'default (OpenAI)'}")

    # Load reference outputs from alpaca_eval dataset if not provided
    if reference_output is None:
        # Try to load from local cache first
        ref_cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "reference_outputs.json")
        if os.path.exists(ref_cache_path):
            print("Loading reference outputs from cache...")
            with open(ref_cache_path) as f:
                reference_outputs = json.load(f)
        else:
            print("Loading reference outputs from alpaca_eval dataset...")
            # Set HF_ENDPOINT for Chinese network
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from datasets import load_dataset
            ref_data = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)
            reference_outputs = [
                {"instruction": item["instruction"], "output": item["output"]}
                for item in ref_data["eval"]
            ]
    else:
        reference_outputs = reference_output

    # Use evaluate function with custom annotator config for gpt-4o via aihubmix
    annotators_config = "gpt4o_aihubmix"

    results = evaluate(
        model_outputs=model_outputs_path,
        reference_outputs=reference_outputs,
        annotators_config=annotators_config,
        output_path=output_dir or "auto",
        is_return_instead_of_print=True,
    )

    return {
        "lc_win_rate": results.get("length_controlled_winrate", 0),
        "win_rate": results.get("winrate", 0),
        "length": results.get("avg_length", 0),
        "n_samples": results.get("n_samples", 805),
        "judge_model": model,
    }


def evaluate_alpacaeval_cli(model_outputs_path: str, output_dir: str, judge_model: str = None) -> None:
    """CLI for AlpacaEval evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="AlpacaEval evaluator for SSPO")
    parser.add_argument("--model-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default=judge_model, help="Judge model (overrides .env)")
    args = parser.parse_args()

    results = evaluate_alpacaeval(args.model_outputs, args.output_dir, judge_model=args.judge_model)

    print(f"LC-Win Rate: {results['lc_win_rate']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Avg Length: {results['length']:.1f} tokens")
    print(f"Judge Model: {results['judge_model']}")


if __name__ == "__main__":
    import sys
    judge_model = os.getenv("JUDGE_MODEL")
    if len(sys.argv) >= 3:
        evaluate_alpacaeval_cli(sys.argv[1], sys.argv[2], judge_model)
    else:
        evaluate_alpacaeval_cli(
            model_outputs_path="results/test_responses_5.json",
            output_dir="results/alpaca_eval_output",
            judge_model=judge_model
        )