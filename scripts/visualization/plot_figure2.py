"""Figure 2: Loss Contribution Ratio Visualization.

Plots labeled vs unlabeled loss contribution ratio over training steps.
"""

import argparse
import json

import matplotlib.pyplot as plt


def plot_figure2(log_file: str, output: str) -> None:
    """Plot labeled vs unlabeled loss contribution ratio over steps.

    Args:
        log_file: Path to JSON training log with fields:
                  steps, loss_contrib_labeled, loss_contrib_unlabeled
        output: Path to save the PNG figure
    """
    try:
        with open(log_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {log_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in log file: {log_file}")

    try:
        steps = data["steps"]
        labeled = data["loss_contrib_labeled"]
        unlabeled = data["loss_contrib_unlabeled"]
    except KeyError as e:
        raise KeyError(f"Missing required field: {e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, labeled, label="Labeled", linewidth=2)
    ax.plot(steps, unlabeled, label="Unlabeled", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss Contribution")
    ax.set_title("Loss Contribution Ratio: Labeled vs Unlabeled")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 2 loss contribution")
    parser.add_argument("--log", required=True, help="Path to training log JSON")
    parser.add_argument("--output", required=True, help="Path to output PNG")
    args = parser.parse_args()
    plot_figure2(args.log, args.output)


if __name__ == "__main__":
    main()