"""Tiny in-process operational metrics (request counts, errors, latency, model version). No audio or personal data."""
from __future__ import annotations

import threading
import time
from collections import Counter, deque

import numpy as np


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: Counter = Counter()
        self.latencies_ms: dict[str, deque] = {}

    def record(self, endpoint: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests[(endpoint, status)] += 1
            self.latencies_ms.setdefault(endpoint, deque(maxlen=1000)).append(ms)

    def snapshot(self, model_version: str | None) -> dict:
        with self._lock:
            lat = {}
            for ep, d in self.latencies_ms.items():
                a = np.array(d)
                lat[ep] = {"n": int(a.size), "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1)}
            total = sum(self.requests.values())
            errors = sum(n for (_, s), n in self.requests.items() if s >= 400)
            return {"uptime_sec": round(time.time() - self.started, 1), "request_count": total, "error_count": errors,
                    "requests_by_endpoint_status": {f"{ep} {s}": n for (ep, s), n in sorted(self.requests.items())},
                    "latency": lat, "model_version": model_version}
