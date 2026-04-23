# Test Documentation - SSPO Project

## Overview

This document describes the test strategy, coverage, and usage for the SSPO reproduction project. Tests follow a TDD workflow and are organized in the `tests/` directory.

## Test Organization

```
tests/
├── data/                              # Data pipeline tests
│   ├── test_download.py              # Download functionality
│   ├── test_preprocess.py            # Basic preprocessing
│   ├── test_preprocess_comprehensive.py  # Business logic & edge cases
│   ├── test_model_configs.py          # Config generation
│   └── test_data_analysis.py         # Analysis script
└── eval/                              # Evaluation tests
    ├── __init__.py
    ├── test_generate_responses.py     # Response generation
    ├── test_alpaca_eval_evaluator.py # AlpacaEval evaluation
    ├── test_mtbench_evaluator.py     # MT-Bench evaluation
    └── test_aggregate_results.py     # Results aggregation
```

## Running Tests

```bash
# Run all tests
source .venv/bin/activate
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/data/test_preprocess_comprehensive.py -v

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=term-missing
```

## Test Coverage Matrix

### Business Problem Coverage

| Problem Area | Test Class | Test Cases | Status |
|--------------|------------|------------|--------|
| **Ratio Edge Cases** | `TestRatioCalculationEdgeCases` | 4 cases | ✅ |
| **Data Quality - Missing Fields** | `TestDataQualityMissingFields` | 4 cases | ✅ |
| **Data Integrity** | `TestDataIntegrity` | 4 cases | ✅ |
| **Save/Load** | `TestSaveLoad` | 4 cases | ✅ |
| **Scale Performance** | `TestScale` | 2 cases | ✅ |
| **Download** | `TestDownload` | 5 cases | ✅ |
| **Config Generation** | `TestModelConfigGeneration` | 6 cases | ✅ |
| **Generation Config** | `TestGenerationConfig` | 4 cases | ✅ |

### Edge Cases Covered

| Edge Case | Description | Detected Bug |
|-----------|-------------|--------------|
| `ratio = 0.0` | Keep zero samples | - |
| `ratio = 1.0` | Keep all samples | - |
| `ratio = 0.001` | Very small ratio | - |
| `ratio = 0.1` | Fractional rounding | - |
| Missing assistant message | UltraChat sample skipped | ✅ Fixed |
| Empty messages list | Skip sample | - |
| Empty rejected_response | Included as empty string | - |
| Missing instruction | Uses empty string | - |

### Data Integrity Checks

| Check | Implementation |
|-------|----------------|
| Labeled data has empty unlabeled | `test_labeled_data_has_no_unlabeled` |
| Unlabeled data has empty chosen/rejected | `test_unlabeled_data_has_empty_chosen_rejected` |
| Combined data is shuffled | `test_combined_data_is_shuffled` |
| All samples have instruction field | `test_all_samples_have_instruction` |

## Test Statistics

- **Total Tests**: 73 (1 skipped)
- **Execution Time**: ~64 seconds
- **Coverage**: Core business logic in `scripts/`

### Additional Test Coverage (Eval)

| Problem Area | Test Class | Test Cases |
|--------------|------------|------------|
| **AlpacaEval** | `TestAlpacaEvalEvaluator` | 3 cases |
| **MT-Bench** | `TestMTBenchEvaluator` | 4 cases (1 skipped) |
| **Aggregation** | `TestAggregateResults` + `TestGenerateComparisonTable` | 10 cases |

## Bug Detection Log

| Date | Bug | Test | Fix |
|------|-----|------|-----|
| Phase 1 | UltraChat samples with no assistant message were added with empty response | `test_ultrachat_missing_assistant_message` | Added check `if not response: continue` in `preprocess_data.py` |
| Phase 1 | Tests used JSON format but `load_jsonl()` expects JSONL | `test_json_save_and_load` | Updated tests to write JSONL format |

## CI/CD Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Create environment
        run: uv venv --python 3.10 .venv && source .venv/bin/activate
      - name: Install dependencies
        run: uv pip install tqdm numpy requests
      - name: Run tests
        run: python -m pytest tests/ -q
```

## Key Test Files

### `tests/data/test_preprocess_comprehensive.py`

Comprehensive business logic tests covering:
- Edge cases in ratio calculation
- Missing assistant messages in UltraChat
- Empty rejected_response in UltraFeedback
- JSON encoding issues
- dataset_info.json overwrite protection
- Empty dataset after sampling
- Memory efficiency with large datasets

### `tests/eval/test_generate_responses.py`

Response generation tests:
- Configuration defaults match paper
- Sampling settings validation
- Benchmark prompt loading (AlpacaEval, MT-Bench)
- Single response generation
- JSON save/load

### `tests/eval/test_alpaca_eval_evaluator.py`

AlpacaEval evaluator tests:
- Function signature validation
- CLI function existence
- ImportError handling when library missing

### `tests/eval/test_mtbench_evaluator.py`

MT-Bench evaluator tests:
- Function signature validation
- CLI function existence
- ImportError handling when library missing

### `tests/eval/test_aggregate_results.py`

Results aggregation tests:
- Directory-based result aggregation
- Nested directory support
- Invalid JSON skipping
- Missing method field handling
- Output path saving
- Comparison table generation

## Test Data

Tests use mock data to avoid dependencies on downloaded datasets:

```python
# Example mock data patterns
mock_data = [{"id": i} for i in range(100)]
ultrafeedback = [{"instruction": "Q1", "chosen_response": "A1", "rejected_response": "A2"}]
ultrachat = [{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}]
```

## Debugging Failed Tests

```bash
# Run with verbose output
python -m pytest tests/data/test_preprocess_comprehensive.py -v -s

# Run specific test
python -m pytest tests/data/test_preprocess_comprehensive.py::TestRatioCalculationEdgeCases::test_ratio_zero_keeps_nothing -v

# Show local variables on failure
python -m pytest tests/ -l
```
