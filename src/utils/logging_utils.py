"""Basic logging setup. Never log raw audio or participant names."""
from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    level = (level or os.environ.get("PD_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
