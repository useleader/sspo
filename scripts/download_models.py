#!/usr/bin/env python3
"""
Download models from ModelScope.

ModelScope is a Chinese ML platform that provides faster downloads in China.

Usage:
    python scripts/download_models.py --model mistral --output models/
    python scripts/download_models.py --model all --output models/
"""

import argparse
import os
from pathlib import Path

from modelscope import snapshot_download


# ModelScope model IDs
MODELSCOPE_MODELS = {
    "mistral": "AI-ModelScope/Mistral-7B-Instruct-v0.2",
    "llama3": "AI-ModelScope/Meta-Llama-3-8B-Instruct",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download models from ModelScope")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        choices=["mistral", "llama3", "qwen2", "all"],
        help="Model to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="ModelScope token (or set MODELSCOPE_TOKEN env var)",
    )
    return parser.parse_args()


def download_model(model_name: str, output_dir: str, token: str = None):
    """Download a single model from ModelScope."""
    if model_name not in MODELSCOPE_MODELS:
        print(f"Unknown model: {model_name}")
        return False

    model_id = MODELSCOPE_MODELS[model_name]
    model_dir = Path(output_dir) / model_name

    print(f"\n{'='*60}")
    print(f"Downloading {model_name} from ModelScope")
    print(f"Model ID: {model_id}")
    print(f"Output: {model_dir}")
    print(f"{'='*60}")

    try:
        # Download model files
        print(f"Downloading model files... (this may take a while)")
        local_path = snapshot_download(
            model_id=model_id,
            cache_dir=output_dir,
            local_dir=str(model_dir),
            token=token,
        )

        # Verify download
        model_files = list(model_dir.glob("*"))
        print(f"\nDownloaded files ({len(model_files)}):")
        total_size = 0
        for f in model_files[:10]:
            size = f.stat().st_size / (1024 * 1024 * 1024)  # GB
            total_size += size
            print(f"  {f.name}: {size:.2f} GB")
        if len(model_files) > 10:
            print(f"  ... and {len(model_files) - 10} more files")

        print(f"\n✅ {model_name} downloaded successfully!")
        print(f"   Local path: {local_path}")
        return True

    except Exception as e:
        print(f"\n❌ Failed to download {model_name}: {e}")
        return False


def main():
    args = parse_args()

    # Get token from env or argument
    token = args.token or os.environ.get("MODELSCOPE_TOKEN")

    if not token:
        print("Warning: No ModelScope token provided.")
        print("Some models may require authentication.")
        print("Set MODELSCOPE_TOKEN environment variable or use --token")
        print("Get your token at: https://www.modelscope.cn/my/myaccesstoken\n")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Determine which models to download
    if args.model == "all":
        models_to_download = list(MODELSCOPE_MODELS.keys())
    else:
        models_to_download = [args.model]

    # Download each model
    results = {}
    for model in models_to_download:
        results[model] = download_model(model, args.output, token)

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model}")

    # Check if all successful
    if all(results.values()):
        print(f"\nAll models downloaded to: {args.output}/")
        print("\nNext steps:")
        print("  1. Update configs to use local paths:")
        print(f"     export MODEL_PATH={args.output}/mistral")
        print("     # Then use MODEL_PATH in training configs")
    else:
        print("\nSome downloads failed. Please check errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
