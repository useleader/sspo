#!/usr/bin/env python3
"""
Generate responses from trained model for evaluation.

This script:
1. Loads trained model with LoRA weights
2. Generates responses for benchmark prompts
3. Saves responses for AlpacaEval / MT-Bench evaluation

Usage:
    python scripts/eval/generate_responses.py \
        --model_path saves/mistral-7b-it/sspo/fb0.01_ch0.1/best_model \
        --dataset alpacaeval \
        --output results/responses.json
"""

import argparse
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


@dataclass
class GenerationConfig:
    """Generation configuration from paper."""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses from trained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (with LoRA weights)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpacaeval",
        choices=["alpacaeval", "mtbench"],
        help="Benchmark dataset to generate responses for",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for responses",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from path."""
    # This is a placeholder - actual implementation would use
    # the LLaMA-Factory model loading utilities
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer


def load_benchmark_prompts(dataset: str) -> List[dict]:
    """Load benchmark prompts."""
    if dataset == "alpacaeval":
        # Load AlpacaEval prompts
        try:
            from alpaca_eval import load_benchmark
            benchmark = load_benchmark(configs=[], include_ground_truth=False)
            return benchmark
        except ImportError:
            print("Warning: alpaca_eval not installed, using mock data")
            return generate_mock_prompts(100)
    elif dataset == "mtbench":
        # Load MT-Bench prompts
        # MT-Bench has 8 categories with multi-turn conversations
        print("Warning: Using mock MT-Bench prompts (mtbench library not installed)")
        return generate_mtbench_prompts()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def generate_mock_prompts(n: int) -> List[dict]:
    """Generate mock prompts for testing."""
    return [
        {"instruction": f"Test prompt {i}", "input": ""}
        for i in range(n)
    ]


def generate_mtbench_prompts() -> List[dict]:
    """Generate MT-Bench prompts (placeholder)."""
    # MT-Bench has 8 categories, 10 questions each
    categories = [
        "reasoning", "math", "coding", "writing",
        "roleplay", "extraction", "stem", "humanities"
    ]
    
    prompts = []
    for category in categories:
        for i in range(10):
            prompts.append({
                "category": category,
                "question_id": f"{category}_{i}",
                "instruction": f"{category.capitalize()} question {i}",
                "input": "",
            })
    
    return prompts


def generate_responses(
    model,
    tokenizer,
    prompts: List[dict],
    config: GenerationConfig,
    batch_size: int,
) -> List[dict]:
    """Generate responses for prompts."""
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i+batch_size]
        
        # Prepare inputs
        instructions = [p["instruction"] for p in batch]
        inputs = tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - config.max_new_tokens,
        )
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
            )
        
        # Decode
        responses = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Collect results
        for prompt, response in zip(batch, responses):
            results.append({
                "instruction": prompt["instruction"],
                "input": prompt.get("input", ""),
                "output": response,
                "category": prompt.get("category", ""),
            })
    
    return results


def save_responses(results: List[dict], output_path: Path) -> None:
    """Save responses to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(results)} responses to {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Generating Responses for Evaluation")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output}")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load prompts
    print(f"Loading {args.dataset} prompts...")
    prompts = load_benchmark_prompts(args.dataset)
    print(f"  Loaded {len(prompts)} prompts")
    
    # Generate
    config = GenerationConfig()
    results = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        config=config,
        batch_size=args.batch_size,
    )
    
    # Save
    save_responses(results, Path(args.output))
    
    print("Done!")


if __name__ == "__main__":
    main()
