"""Custody + presence FSM and ownership-margin tests (§11 "test what matters").

The FSM and ownership logic are pure functions of track streams — the plan is to
drive scripted (t, x, y) trajectories through the reasoner here, with NO GPU and
NO video, and assert the expected event stream + final states. This is how L2 is
developed fast; real video is for validation, not iteration.

Also the home of the golden staged-scenario regression (§8.8): same recorded
input + same config => byte-identical event log (replay determinism).

Phase 2 — skipped until reasoner logic lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Phase 2: implement reasoner FSM/ownership first")


def test_placed_to_abandoned_timeline():
    ...


def test_ownership_margin_rejects_crowd_passerby():
    ...


def test_presence_removed_unseen_on_occluded_removal():
    ...
