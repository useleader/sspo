# ADR-0004: MT-Bench and AlpacaEval for Evaluation

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper evaluates on MT-Bench (multi-turn dialogue) and AlpacaEval (instruction following) using Length-Controlled Win Rate (LC-Win Rate). The current `src_sspo/llamafactory/eval/evaluator.py` only supports academic benchmarks (MMLU, etc.), not alignment benchmarks.

## Decision

We implement evaluation pipeline in `src/examples/eval/` with the following metrics from the paper:

### Primary Metrics (from paper page 8)

**AlpacaEval (Length-Controlled Win Rate)**:
- Length-controlled win rate vs reference (GPT-4 as judge)
- Evaluation using `alpaca_eval` library
- Max length: 2048 tokens

**MT-Bench**:
- Multi-turn conversation quality
- First turn evaluation (average across 8 categories)
- Categories: reasoning, math, coding, writing, roleplay, extraction, STEM, humanities

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| Standard Win Rate | Raw win rate without length control |
| Length | Average response length |
| GPT-4 Score | Per-sample GPT-4 evaluation |

### Evaluation Configuration (from paper page 7)

| Parameter | Value |
|-----------|-------|
| AlpacaEval samples | 805 |
| MT-Bench categories | 8 |
| Generation max tokens | 2048 |
| Temperature | 0.7 |
| Judge model | GPT-4 |

## Implementation Components

### 1. Response Generation (`generate_response.py`)
```python
# Generate responses using trained model on benchmark prompts
def generate_responses(model_path, dataset, output_dir):
    # Load model with LoRA weights
    # Generate responses for AlpacaEval / MT-Bench
    # Save to JSON for evaluation
```

### 2. AlpacaEval Evaluator
```python
# Use alpaca_eval library
from alpaca_eval import Evaluator

evaluator = Evaluator(model_outputs=outputs, reference_outputs=references)
results = evaluator.evaluate()
```

### 3. MT-Bench Evaluator
```python
# Multi-turn conversation scoring
# Category-wise evaluation
# First turn only for consistency with paper
```

### 4. Results Aggregation (`aggregate_results.py`)
```python
# Aggregate results across models and methods
# Generate comparison tables matching paper format
# Compute win rate vs baselines
```

## Benchmark Details (from paper page 8)

### AlpacaEval Results (Table 1)

| Method | LC-Win Rate (%) |
|--------|-----------------|
| SFT | 12.1 |
| DPO | 26.2 |
| SimPO | 26.8 |
| ORPO | 25.6 |
| KTO | 25.9 |
| **SSPO** | **32.4** |

### MT-Bench Results (Table 2, page 8)

| Category | DPO | SimPO | SSPO |
|----------|-----|-------|------|
| Reasoning | 6.2 | 6.3 | 7.1 |
| Math | 5.8 | 5.9 | 6.5 |
| Coding | 5.5 | 5.6 | 6.2 |
| Writing | 6.8 | 6.9 | 7.5 |
| ... | ... | ... | ... |

## Alternatives Considered

### Alternative 1: Human Evaluation
- **Pros**: Gold standard for helpfulness/harmlessness
- **Cons**: Expensive, slow, not reproducible
- **Why not**: Paper uses AutoEval for scalability

### Alternative 2: Only Use Academic Benchmarks (MMLU, etc.)
- **Pros**: Already implemented in current evaluator.py
- **Cons**: Does not measure alignment quality
- **Why not**: Alignment benchmarks are required for SSPO paper claims

## Consequences

### Positive
- Matches paper evaluation: MT-Bench + AlpacaEval are paper benchmarks
- Reproducible: automated scoring reduces human bias
- Comprehensive: measures both dialogue quality and instruction following

### Negative
- External API dependency: GPT-4 scoring requires OpenAI API access
- Cost: AlpacaEval with GPT-4 is expensive (~800 samples × API cost)

### Risks
- **Risk**: API costs and rate limits
- **Mitigation**: Cache results, use GPT-3.5 for development, GPT-4 only for final evaluation
