#!/usr/bin/env python3
"""Thin CLI wrapper. All logic lives in src/data/verification.py (so it is importable and testable)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.verification import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
