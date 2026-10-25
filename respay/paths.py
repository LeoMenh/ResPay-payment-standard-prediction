"""Repository-relative paths shared by ResPay workflows."""

from pathlib import Path

# This module lives at ``<repo>/respay/paths.py``; deriving the root from the
# module location keeps commands independent of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DATA_ROOT = REPO_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = RESULTS_ROOT / "processed"

__all__ = [
    "DATA_ROOT",
    "PROCESSED_DATA_ROOT",
    "RAW_DATA_ROOT",
    "REPO_ROOT",
    "RESULTS_ROOT",
]
