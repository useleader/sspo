"""
SSPO Logging Utilities

Provides consistent logging across all SSPO pipeline scripts with:
- Step tracking
- Elapsed time
- Stage indicators
- Color support
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD = "\033[1m"


class StepTracker:
    """Track progress through multi-step pipeline."""

    def __init__(self, name: str, total_steps: int = None):
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_start_time = self.start_time
        self.logger = get_logger(name)

    def step(self, message: str, step_num: int = None):
        """Log a step with timing."""
        self.current_step = step_num or self.current_step + 1
        elapsed = time.time() - self.step_start_time
        self.step_start_time = time.time()

        if self.total_steps:
            self.logger.info(
                f"[{self.current_step}/{self.total_steps}] {message} "
                f"(+{elapsed:.1f}s)"
            )
        else:
            self.logger.info(f"[{self.current_step}] {message} (+{elapsed:.1f}s)")

    def complete(self, message: str = None):
        """Log completion with total time."""
        total = time.time() - self.start_time
        msg = message or "Complete"
        self.logger.info(f"✓ {msg} (total: {total:.1f}s)")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Get or create a logger with consistent formatting."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Console handler with color
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"{Colors.CYAN}[%(asctime)s]{Colors.RESET} "
        f"%(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logging(
    name: str,
    log_dir: Optional[Path] = None,
    verbose: bool = False
) -> tuple:
    """Setup logging and return (logger, log_file)."""
    log_file = None
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

    logger = get_logger(name, log_file)

    if verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    return logger, log_file


def print_header(title: str, width: int = 60):
    """Print a section header."""
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * width}{Colors.RESET}")
    print()


def print_step(step: int, total: int, message: str, duration: float = None):
    """Print a formatted step message."""
    duration_str = f" (+{duration:.1f}s)" if duration else ""
    print(f"{Colors.CYAN}[{step}/{total}]{Colors.RESET} {message}{duration_str}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.WHITE}ℹ {message}{Colors.RESET}")
