"""Results aggregation for SSPO reproduction.

Aggregates evaluation results across models and methods,
generates comparison tables matching paper format.

Paper reference: Table 1 (page 8) and Table 2 (page 8) format.
"""
import json
from pathlib import Path
from typing import Optional


def aggregate_results(
    results_dir: str,
    output_path: Optional[str] = None,
) -> dict:
    """Aggregate evaluation results into paper-matching tables.

    Args:
        results_dir: Directory containing evaluation results
        output_path: Optional path to save aggregated results

    Returns:
        Dictionary with aggregated tables matching paper format

    Paper Table 1 (AlpacaEval LC-Win Rate):
        | Method | LC-Win Rate (%) |
        |--------|-----------------|
        | SFT    | 12.1           |
        | DPO    | 26.2           |
        | SimPO  | 26.8           |
        | ORPO   | 25.6           |
        | KTO    | 25.9           |
        | SSPO   | 32.4           |
    """
    results_path = Path(results_dir)
    aggregated = {
        "alpaca_eval": {},
        "mtbench": {},
    }

    for result_file in results_path.glob("**/*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            method = data.get("method", "unknown")
            model = data.get("model", "unknown")

            if "lc_win_rate" in data:
                key = f"{model}_{method}"
                aggregated["alpaca_eval"][key] = data["lc_win_rate"]

            if "average_score" in data:
                key = f"{model}_{method}"
                aggregated["mtbench"][key] = data["average_score"]

        except (json.JSONDecodeError, KeyError):
            continue

    if output_path:
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)

    return aggregated


def generate_comparison_table(
    aggregated: dict,
    metric: str = "alpaca_eval",
) -> str:
    """Generate markdown comparison table matching paper format."""
    if metric not in aggregated:
        return "No data available"

    data = aggregated[metric]
    methods = ["SFT", "DPO", "SimPO", "ORPO", "KTO", "SSPO"]
    models = ["mistral", "llama3", "qwen2"]

    lines = []
    if metric == "alpaca_eval":
        lines.append("| Method | " + " | ".join(models) + " |")
        lines.append("|--------|" + "|".join(["---"] * (len(models) + 1)) + "|")
        for method in methods:
            row = [method]
            for model in models:
                key = f"{model}_{method.lower()}"
                value = data.get(key, "N/A")
                row.append(f"{value:.1f}" if isinstance(value, float) else value)
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| Category | " + " | ".join(methods[1:]) + " |")
        lines.append("|----------|" + "|".join(["---"] * len(methods)) + "|")
        categories = ["reasoning", "math", "coding", "writing"]
        for cat in categories:
            row = [cat.capitalize()]
            for method in methods[1:]:
                key = f"{cat}_{method.lower()}"
                value = data.get(key, "N/A")
                row.append(f"{value:.1f}" if isinstance(value, float) else value)
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SSPO evaluation results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    results = aggregate_results(args.results_dir, args.output)
    print("AlpacaEval Results:")
    print(generate_comparison_table(results, "alpaca_eval"))
    print("\nMT-Bench Results:")
    print(generate_comparison_table(results, "mtbench"))
