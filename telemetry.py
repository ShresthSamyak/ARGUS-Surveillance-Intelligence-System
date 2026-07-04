"""Metrics + structured logging (§8.5). Imported by every process.

Dependency-light on purpose: falls back to no-op counters if prometheus_client
is absent so a dev box or the ingestion scripts never hard-depend on it.

TODO (Phase 3): wire real prometheus_client Counters/Histograms and a /healthz
endpoint per process (worker: frames flowing + GPU; reasoner: tick age; vlm:
queue draining). Metric names are fixed by the spec — keep them stable:
  argus_detections_total, argus_det_latency_ms, argus_reasoner_tick_ms,
  argus_bags_by_custody, argus_bags_by_presence, argus_vlm_queue_depth,
  argus_false_alarm_rate_hr, argus_e2e_alert_latency_s (SLO p95 < 2 s)
"""

from __future__ import annotations

import json
import sys
import time


def log_event(**fields) -> None:
    """One event = one JSON line (same schema as the events table)."""
    fields.setdefault("ts", time.time())
    sys.stdout.write(json.dumps(fields, default=str) + "\n")


class _NoopMetric:
    def labels(self, *a, **k):  # noqa: D401
        return self

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass


def counter(name: str, doc: str = "", labels: tuple[str, ...] = ()):  # noqa: ARG001
    return _NoopMetric()


histogram = counter
gauge = counter
