"""FastAPI app entrypoint (§8.3). Phase 3.

Endpoints (planned):
  GET  /healthz                       process liveness (§8.5)
  GET  /entities/{id}/timeline        ordered events + keyframes (the product screen)
  GET  /alerts?tier=                  alert queue
  POST /alerts/{id}/{ack|dismiss|escalate}   operator actions (logged as events)
  WS   /live/{cam_id}                 track JSON for client-side overlay
Auth: operator vs admin roles (§8.7).
"""

from __future__ import annotations

# from fastapi import FastAPI
# app = FastAPI(title="ARGUS")


def build_app():
    raise NotImplementedError("Phase 3: build FastAPI app")
