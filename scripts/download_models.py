#!/usr/bin/env python3
"""
Download pre-trained models from HuggingFace or ModelScope.

Supports 8 models across 3 domains:
    General: mistral, llama3, qwen2, phi2
    Medical: meerkat, ultramedical
    Business: mistral-business, finance

Usage:
    # Download from HuggingFace (default)
    python scripts/download_models.py --model mistral --output ./cache/

    # Download all models
    python scripts/download_models.py --model all --output ./cache/

    # Use ModelScope instead (may be faster in China)
    python scripts/download_models.py --model mistral --source modelscope --output ./models/

    # List available models
    python scripts/download_models.py --list

Note:
    Llama-3 and some models require HF_TOKEN:
    export HF_TOKEN=your_huggingface_token
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add scripts to path for logging module
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logging import (
    Colors,
    StepTracker,
    get_logger,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
)


# ANSI color shortcuts
GREEN = Colors.GREEN
RED = Colors.RED
YELLOW = Colors.YELLOW
CYAN = Colors.CYAN
BOLD = Colors.BOLD
RESET = Colors.RESET


# Model configurations
MODELS = {
    # General domain models
    "mistral": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "ms_path": "AI-ModelScope/Mistral-7B-Instruct-v0.2",
        "name": "Mistral-7B-Instruct-v0.2",
        "size_gb": 14,
        "description": "General purpose instruction-following model",
    },
    "llama3": {
        "hf_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "ms_path": "LLM-Research/Meta-Llama-3-8B-Instruct",
        "name": "Meta-Llama-3-8B-Instruct",
        "size_gb": 8,
        "description": "General purpose instruction-following model (requires token)",
    },
    "qwen2": {
        "hf_path": "Qwen/Qwen2-7B-Instruct",
        "ms_path": "Qwen/Qwen2-7B-Instruct",
        "name": "Qwen2-7B-Instruct",
        "size_gb": 8,
        "description": "General purpose instruction-following model",
    },
    "phi2": {
        "hf_path": "microsoft/Phi-2",
        "ms_path": None,  # Not available on ModelScope
        "name": "Phi-2",
        "size_gb": 5,
        "description": "2.7B parameter reasoning model",
    },
    # Medical domain models
    "meerkat": {
        "hf_path": "CognitiveLLVM/Meerkat-7B-v1.0",
        "ms_path": None,  # Not available on ModelScope
        "name": "Meerkat-7B-v1.0",
        "size_gb": 14,
        "description": "Medical domain model for healthcare applications",
    },
    "ultramedical": {
        "hf_path": "zelrn/llama3-8b-ultramedical",
        "ms_path": None,
        "name": "Llama3-8B-UltraMedical",
        "size_gb": 16,
        "description": "Medical domain model fine-tuned on medical data",
    },
    # Business domain models
    "mistral-business": {
        "hf_path": "sadavart/finrl-chatbot-mistral-7b",
        "ms_path": None,
        "name": "FinRL-Chatbot-Mistral-7B",
        "size_gb": 14,
        "description": "Business/Finance chatbot based on Mistral-7B",
    },
    "finance": {
        "hf_path": "Derivative/Finance-LLaMA-8B",
        "ms_path": None,
        "name": "Finance-LLaMA-8B",
        "size_gb": 16,
        "description": "Finance domain model for financial tasks",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./cache",
        help="Output directory for cached models",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="Download source",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace/ModelScope token",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files",
    )
    return parser.parse_args()


def get_token(args) -> str:
    """Get token from args or environment."""
    return args.token or os.environ.get("HF_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")


def download_from_huggingface(
    model_key: str,
    output_dir: Path,
    token: str = None,
    logger=None,
    tracker: StepTracker = None,
    step_num: int = None,
) -> tuple:
    """Download a single model from HuggingFace. Returns (success, size_gb)."""
    model_info = MODELS[model_key]
    hf_path = model_info["hf_path"]
    model_dir = Path(output_dir) / model_key

    print(f"\n  {BOLD}HF path:{RESET} {hf_path}")
    print(f"  {BOLD}Output:{RESET} {model_dir}")

    try:
        from huggingface_hub import snapshot_download

        print(f"  {CYAN}Downloading from HuggingFace...{RESET}")

        start_time = time.time()
        local_path = snapshot_download(
            repo_id=hf_path,
            cache_dir=str(output_dir / model_key),
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
        elapsed = time.time() - start_time

        # Verify download
        model_files = list(model_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in model_files if f.is_file())
        size_gb = total_size / (1024 ** 3)

        print(f"  {GREEN}✓{RESET} Downloaded {len(model_files)} files ({size_gb:.2f} GB, {elapsed:.1f}s)")

        if logger:
            logger.info(f"{model_key}: {len(model_files)} files, {size_gb:.2f} GB, {elapsed:.1f}s")

        return True, size_gb

    except Exception as e:
        print_error(f"Failed: {e}")
        if logger:
            logger.error(f"{model_key}: {e}")
        return False, 0


def download_from_modelscope(
    model_key: str,
    output_dir: Path,
    token: str = None,
    logger=None,
    tracker: StepTracker = None,
    step_num: int = None,
) -> tuple:
    """Download a single model from ModelScope. Returns (success, size_gb)."""
    model_info = MODELS[model_key]
    ms_path = model_info["ms_path"]

    if ms_path is None:
        print_error(f"Model {model_key} not available on ModelScope, use --source huggingface")
        return False, 0

    model_dir = Path(output_dir) / model_key

    print(f"\n  {BOLD}ModelScope ID:{RESET} {ms_path}")
    print(f"  {BOLD}Output:{RESET} {model_dir}")

    try:
        from modelscope import snapshot_download

        print(f"  {CYAN}Downloading from ModelScope...{RESET}")

        start_time = time.time()
        local_path = snapshot_download(
            model_id=ms_path,
            cache_dir=str(output_dir),
            local_dir=str(model_dir),
            token=token,
        )
        elapsed = time.time() - start_time

        # Verify download
        model_files = list(model_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in model_files if f.is_file())
        size_gb = total_size / (1024 ** 3)

        print(f"  {GREEN}✓{RESET} Downloaded {len(model_files)} files ({size_gb:.2f} GB, {elapsed:.1f}s)")

        if logger:
            logger.info(f"{model_key}: {len(model_files)} files, {size_gb:.2f} GB, {elapsed:.1f}s")

        return True, size_gb

    except Exception as e:
        print_error(f"Failed: {e}")
        if logger:
            logger.error(f"{model_key}: {e}")
        return False, 0


def list_models():
    """List all available models."""
    print_header("Available Models for SSPO")
    print()

    domains = {
        "General": ["mistral", "llama3", "qwen2", "phi2"],
        "Medical": ["meerkat", "ultramedical"],
        "Business": ["mistral-business", "finance"],
    }

    for domain, models in domains.items():
        print(f"{BOLD}{domain} Domain:{RESET}")
        for model_key in models:
            info = MODELS[model_key]
            print(f"  {CYAN}{model_key:20s}{RESET} - {info['description']}")
            print(f"    {BOLD}HF:{RESET} {info['hf_path']}")
            if info["ms_path"]:
                print(f"    {BOLD}MS:{RESET} {info['ms_path']}")
            print(f"    {BOLD}Size:{RESET} ~{info['size_gb']}GB")
            print()
        print()


def main():
    args = parse_args()

    # List mode
    if args.list:
        list_models()
        return

    output_dir = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_models_{timestamp}.log"
    logger = get_logger("download_models", log_file)

    # Get token
    token = get_token(args)

    # Determine which models to download
    if args.model == "all":
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]

    # Setup step tracker
    tracker = StepTracker("download_models", total_steps=len(models_to_download))

    print_header(f"SSPO Model Downloader ({args.source.capitalize()})")
    print(f"{CYAN}Output directory:{RESET} {output_dir}")
    print(f"{CYAN}Log file:{RESET} {log_file}")
    print(f"{CYAN}Models:{RESET} {', '.join(models_to_download)}")

    if not token:
        print_warning("No token provided. Some models may fail.")
        if args.source == "huggingface":
            print(f"  Set HF_TOKEN env var or use --token")
        else:
            print(f"  Set MODELSCOPE_TOKEN env var or use --token")

    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    tracker.step("Checking dependencies", step_num=1)
    if args.source == "huggingface":
        try:
            from huggingface_hub import snapshot_download
            print_success("HuggingFace SDK available")
        except ImportError:
            print_error("HuggingFace SDK not installed")
            print(f"  Install with: uv pip install huggingface_hub")
            sys.exit(1)
    else:
        try:
            from modelscope import snapshot_download
            print_success("ModelScope SDK available")
        except ImportError:
            print_error("ModelScope SDK not installed")
            print(f"  Install with: uv pip install modelscope")
            sys.exit(1)

    overall_start = time.time()
    results = {}
    total_size = 0

    # Download each model
    for i, model in enumerate(models_to_download, 2):
        tracker.step(f"Downloading {model}", step_num=i)

        if args.source == "huggingface":
            success, size_gb = download_from_huggingface(
                model, output_dir, token, logger, tracker, i
            )
        else:
            success, size_gb = download_from_modelscope(
                model, output_dir, token, logger, tracker, i
            )

        results[model] = success
        if success:
            total_size += size_gb

    # Summary
    elapsed = time.time() - overall_start
    print_header("Download Summary")
    success_count = sum(1 for v in results.values() if v)

    for model, success in results.items():
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"  {status} {model}")

    print()
    if success_count == len(results):
        print_success(f"All {success_count} models downloaded ({total_size:.2f} GB, {elapsed:.1f}s)")
    else:
        print_error(f"{success_count}/{len(results)} downloaded ({elapsed:.1f}s)")

    logger.info(f"Summary: {success_count}/{len(results)}, {total_size:.2f} GB, {elapsed:.1f}s")

    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
