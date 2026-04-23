#!/usr/bin/env python3
"""
Data Analysis Script - Sample Length Distribution and Characteristics.

This script analyzes the downloaded datasets to understand:
1. Sample counts
2. Length distributions (instruction, response, messages)
3. Token estimation for batching
4. Data quality metrics

Run with:
    python scripts/analyze_data.py
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def analyze_ultrafeedback(data: List[dict]) -> Dict[str, Any]:
    """Analyze UltraFeedback dataset."""
    print("\n" + "=" * 60)
    print("UltraFeedback Analysis")
    print("=" * 60)

    n_samples = len(data)
    print(f"\nTotal samples: {n_samples:,}")

    # Instruction length analysis
    instruction_lengths = [count_words(s["instruction"]) for s in data]
    print(f"\n--- Instruction Length (words) ---")
    print(f"  Mean:     {statistics.mean(instruction_lengths):.1f}")
    print(f"  Median:   {statistics.median(instruction_lengths):.1f}")
    print(f"  Std Dev:  {statistics.stdev(instruction_lengths):.1f}")
    print(f"  Min:      {min(instruction_lengths)}")
    print(f"  Max:      {max(instruction_lengths)}")
    print(f"  P25:      {np.percentile(instruction_lengths, 25):.1f}")
    print(f"  P75:      {np.percentile(instruction_lengths, 75):.1f}")
    print(f"  P95:      {np.percentile(instruction_lengths, 95):.1f}")

    # Chosen response length analysis
    chosen_lengths = [count_words(s["chosen_response"]) for s in data]
    print(f"\n--- Chosen Response Length (words) ---")
    print(f"  Mean:     {statistics.mean(chosen_lengths):.1f}")
    print(f"  Median:   {statistics.median(chosen_lengths):.1f}")
    print(f"  Std Dev:  {statistics.stdev(chosen_lengths):.1f}")
    print(f"  Min:      {min(chosen_lengths)}")
    print(f"  Max:      {max(chosen_lengths)}")
    print(f"  P25:      {np.percentile(chosen_lengths, 25):.1f}")
    print(f"  P75:      {np.percentile(chosen_lengths, 75):.1f}")
    print(f"  P95:      {np.percentile(chosen_lengths, 95):.1f}")

    # Rejected response length analysis (non-empty only)
    rejected_lengths = [count_words(s["rejected_response"]) for s in data if s.get("rejected_response")]
    if rejected_lengths:
        print(f"\n--- Rejected Response Length (words, non-empty only) ---")
        print(f"  Count:    {len(rejected_lengths):,} ({len(rejected_lengths)/n_samples*100:.1f}% of samples)")
        print(f"  Mean:     {statistics.mean(rejected_lengths):.1f}")
        print(f"  Median:   {statistics.median(rejected_lengths):.1f}")
        print(f"  P75:      {np.percentile(rejected_lengths, 75):.1f}")
        print(f"  P95:      {np.percentile(rejected_lengths, 95):.1f}")

    # Source distribution
    sources = {}
    for s in data:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"\n--- Source Distribution ---")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {count:,} ({count/n_samples*100:.1f}%)")

    # Rating analysis
    chosen_ratings = [s["chosen_avg_rating"] for s in data]
    rejected_ratings = [s["rejected_avg_rating"] for s in data if s.get("rejected_avg_rating")]
    print(f"\n--- Rating Analysis ---")
    print(f"  Chosen rating - Mean: {statistics.mean(chosen_ratings):.2f}, Std: {statistics.stdev(chosen_ratings):.2f}")
    if rejected_ratings:
        print(f"  Rejected rating - Mean: {statistics.mean(rejected_ratings):.2f}, Std: {statistics.stdev(rejected_ratings):.2f}")

    # Model distribution
    models = {}
    for s in data:
        model = s.get("chosen_model", "unknown")
        models[model] = models.get(model, 0) + 1
    print(f"\n--- Model Distribution (chosen responses) ---")
    for model, count in sorted(models.items(), key=lambda x: -x[1])[:10]:
        print(f"  {model}: {count:,} ({count/n_samples*100:.1f}%)")

    # Token estimation (rough: 1 word ≈ 1.3 tokens for English)
    avg_total_words = statistics.mean(instruction_lengths) + statistics.mean(chosen_lengths)
    est_tokens = avg_total_words * 1.3
    print(f"\n--- Token Estimation (per sample) ---")
    print(f"  Avg instruction tokens: ~{statistics.mean(instruction_lengths) * 1.3:.0f}")
    print(f"  Avg response tokens: ~{statistics.mean(chosen_lengths) * 1.3:.0f}")
    print(f"  Total per sample: ~{est_tokens:.0f} tokens")
    print(f"  At context 1024 tokens: {1024/est_tokens:.1f} samples per context window")

    return {
        "n_samples": n_samples,
        "instruction_mean": statistics.mean(instruction_lengths),
        "chosen_mean": statistics.mean(chosen_lengths),
        "total_words_mean": avg_total_words,
        "est_tokens_per_sample": est_tokens,
    }


def analyze_ultrachat(data: List[dict]) -> Dict[str, Any]:
    """Analyze UltraChat dataset."""
    print("\n" + "=" * 60)
    print("UltraChat Analysis")
    print("=" * 60)

    n_samples = len(data)
    print(f"\nTotal samples: {n_samples:,}")

    # Message count per conversation
    msg_counts = [len(s["messages"]) for s in data]
    print(f"\n--- Messages per Conversation ---")
    print(f"  Mean:     {statistics.mean(msg_counts):.1f}")
    print(f"  Median:   {statistics.median(msg_counts):.1f}")
    print(f"  Std Dev:  {statistics.stdev(msg_counts):.1f}")
    print(f"  Min:      {min(msg_counts)}")
    print(f"  Max:      {max(msg_counts)}")
    print(f"  P25:      {np.percentile(msg_counts, 25):.1f}")
    print(f"  P75:      {np.percentile(msg_counts, 75):.1f}")
    print(f"  P95:      {np.percentile(msg_counts, 95):.1f}")

    # Role distribution
    roles = {}
    for s in data:
        for msg in s["messages"]:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
    print(f"\n--- Role Distribution ---")
    for role, count in sorted(roles.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count:,}")

    # Message content length
    user_lengths = []
    assistant_lengths = []
    for s in data:
        for msg in s["messages"]:
            content = msg.get("content", "")
            if msg.get("role") in ("user", "human"):
                user_lengths.append(count_words(content))
            elif msg.get("role") in ("assistant", "gpt"):
                assistant_lengths.append(count_words(content))

    print(f"\n--- User Message Length (words) ---")
    if user_lengths:
        print(f"  Count:    {len(user_lengths):,}")
        print(f"  Mean:     {statistics.mean(user_lengths):.1f}")
        print(f"  Median:   {statistics.median(user_lengths):.1f}")
        print(f"  P95:      {np.percentile(user_lengths, 95):.1f}")

    print(f"\n--- Assistant Message Length (words) ---")
    if assistant_lengths:
        print(f"  Count:    {len(assistant_lengths):,}")
        print(f"  Mean:     {statistics.mean(assistant_lengths):.1f}")
        print(f"  Median:   {statistics.median(assistant_lengths):.1f}")
        print(f"  P95:      {np.percentile(assistant_lengths, 95):.1f}")

    # Estimate total tokens per conversation
    total_words = sum(user_lengths + assistant_lengths)
    est_tokens = total_words * 1.3
    print(f"\n--- Token Estimation ---")
    print(f"  Total words: ~{total_words:,}")
    print(f"  Total tokens: ~{est_tokens:,.0f}")
    print(f"  Avg tokens per conversation: ~{est_tokens/n_samples:.0f}")

    return {
        "n_samples": n_samples,
        "msg_count_mean": statistics.mean(msg_counts),
        "est_tokens_per_sample": est_tokens / n_samples,
    }


def estimate_batch_sizes(uf_stats: Dict, uc_stats: Dict):
    """Estimate batch sizes based on context length."""
    print("\n" + "=" * 60)
    print("Batch Size Estimation")
    print("=" * 60)

    context_len = 1024  # From paper

    print(f"\n--- UltraChat (for reference model training) ---")
    tokens_per_sample = uc_stats.get("est_tokens_per_sample", 500)
    samples_per_batch = max(1, int(context_len / tokens_per_sample))
    print(f"  Avg tokens/sample: ~{tokens_per_sample:.0f}")
    print(f"  Samples/batch at context {context_len}: ~{samples_per_batch}")
    print(f"  With 4 GPU x 4 batch size: {samples_per_batch * 16} samples/batch")

    print(f"\n--- UltraFeedback (for SSPO training) ---")
    # SSPO uses longer context for preference learning
    tokens_per_sample = uf_stats.get("total_words_mean", 300) * 1.3
    samples_per_batch = max(1, int(context_len / tokens_per_sample))
    print(f"  Avg tokens/sample: ~{tokens_per_sample:.0f}")
    print(f"  Samples/batch at context {context_len}: ~{samples_per_batch}")


def main():
    print("=" * 60)
    print("SSPO Data Analysis")
    print("=" * 60)

    # Analyze UltraFeedback
    uf_path = Path("data/ultrafeedback/train.json")
    if uf_path.exists():
        uf_data = load_jsonl(uf_path)
        uf_stats = analyze_ultrafeedback(uf_data)
    else:
        print("\nUltraFeedback not found at data/ultrafeedback/train.json")
        uf_stats = {}

    # Analyze UltraChat
    uc_path = Path("data/ultrachat/train_sft.json")
    if uc_path.exists():
        uc_data = load_jsonl(uc_path)
        uc_stats = analyze_ultrachat(uc_data)
    else:
        print("\nUltraChat not found at data/ultrachat/train_sft.json")
        uc_stats = {}

    # Batch size estimation
    if uf_stats and uc_stats:
        estimate_batch_sizes(uf_stats, uc_stats)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
