"""SSPO Evaluation package.

Modules:
- generate_responses: Generate model responses for evaluation
- alpaca_eval_evaluator: AlpacaEval LC-Win Rate evaluation
- mtbench_evaluator: MT-Bench multi-turn dialogue evaluation
- aggregate_results: Aggregate and compare results against paper baselines
"""
from .generate_responses import GenerationConfig, generate_responses
from .alpaca_eval_evaluator import evaluate_alpacaeval
from .mtbench_evaluator import evaluate_mtbench
from .aggregate_results import aggregate_results, generate_comparison_table

__all__ = [
    "GenerationConfig",
    "generate_responses",
    "evaluate_alpacaeval",
    "evaluate_mtbench",
    "aggregate_results",
    "generate_comparison_table",
]
