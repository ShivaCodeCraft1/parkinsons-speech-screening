# Monitoring

The API keeps simple in-process operational metrics (`GET /metrics`): request counts per endpoint/status, error count,
p50/p95 latency per endpoint and the loaded model version (implementation: `src/utils/metrics.py`).
Metrics reset when the server restarts. Raw audio and participant information are never logged.

This is operational monitoring only. It does NOT monitor disease progression or model drift on validated outcomes.
