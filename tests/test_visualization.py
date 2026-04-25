"""Tests for scripts/visualization/plot_figure2.py"""

import json
import os
import tempfile
import pytest


def test_plot_figure2():
    """Test plot_figure2 creates PNG from mock log data."""
    from scripts.visualization.plot_figure2 import plot_figure2

    # Create mock log data
    log_data = {
        "steps": list(range(100)),
        "loss_contrib_labeled": [0.8 + 0.2 * (i / 100) for i in range(100)],
        "loss_contrib_unlabeled": [0.2 - 0.1 * (i / 100) for i in range(100)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "log.json")
        output_file = os.path.join(tmpdir, "figure2.png")

        with open(log_file, "w") as f:
            json.dump(log_data, f)

        plot_figure2(log_file, output_file)

        assert os.path.exists(output_file), f"PNG not created at {output_file}"
        assert os.path.getsize(output_file) > 0, "PNG file is empty"


def test_plot_figure3():
    """Test plot_figure3 creates PNG from mock log data."""
    from scripts.visualization.plot_figure2 import plot_figure3

    # Create mock log data with reward distribution lists
    log_data = {
        "reward_chosen_mean_step100": [0.5 + 0.1 * i for i in range(50)],
        "reward_chosen_mean_step500": [0.6 + 0.08 * i for i in range(50)],
        "reward_chosen_mean_step1000": [0.7 + 0.05 * i for i in range(50)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "log.json")
        output_file = os.path.join(tmpdir, "figure3.png")

        with open(log_file, "w") as f:
            json.dump(log_data, f)

        plot_figure3(log_file, output_file)

        assert os.path.exists(output_file), f"PNG not created at {output_file}"
        assert os.path.getsize(output_file) > 0, "PNG file is empty"