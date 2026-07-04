"""Scene reasoner — single process, single authority for world state (§6.11).

Consumes all track streams from Redis, owns the entity graph, ownership engine,
custody + presence FSMs, behavior rules, risk. Writes events/alerts to the
store and publishes live alerts + per-camera watch-ROIs back to the fast path.

    def tick(batch):                    # 10 Hz
        bind_tracks_to_entities(batch)  # §6.1, §6.9 Re-ID on new tracks
        update_trajectories_and_banks(batch)
        accumulate_ownership_evidence() # §6.3 (+ birth lookback)
        detect_place_pickup_events()    # §6.4
        ingest_vlm_evidence()           # §5.7 verdicts/fingerprints (v0.2)
        step_all_bag_fsms(); recompute_risk()   # custody + presence + risk
        run_behavior_rules()            # §7
        publish_watch_rois()            # §6.7 expectation feedback (v0.2)
        emit_events(); emit_alerts()

Phase 2 (core) + Phase 3 (behaviors, alerts). Snapshot graph every 30 s;
recovery = snapshot + event replay.
"""

from __future__ import annotations


def tick(batch) -> None:  # noqa: ARG001
    raise NotImplementedError("Phase 2")


def main() -> None:
    raise NotImplementedError("Phase 2: subscribe to tracks.*, run tick loop at 10 Hz")


if __name__ == "__main__":
    main()
