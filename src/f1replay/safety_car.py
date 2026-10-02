"""Simulated Safety Car: deploy from pit lane, lead the field, return.

The F1 data feed provides no GPS for the Safety Car itself, so its position is
simulated from real track-status data (code '4') and the race leader's track
position: the SC runs ~500 m ahead of the leader on the reference polyline.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import ReplayDataset

DEPLOY_DURATION = 3.0  # seconds of replay time
RETURN_DURATION = 3.0
LEAD_AHEAD_METRES = 500.0


@dataclass
class SCFrame:
    x: float
    y: float
    phase: str  # 'deploy' | 'track' | 'return'
    label: str
    s: float


def _shortest_delta(a: float, b: float, length: float) -> float:
    """Signed shortest distance along the loop from a to b."""
    return (b - a + 1.5 * length) % length - 0.5 * length


def compute_sc(ds: ReplayDataset, t: float, leader_s: float) -> SCFrame | None:
    track = ds.track
    L = track.length
    for w in ds.sc_windows:
        if t < w.start or t > w.end + RETURN_DURATION:
            continue
        target = (leader_s + LEAD_AHEAD_METRES) % L

        if t < w.start + DEPLOY_DURATION:
            # phase 1: burst out of the pit lane onto the track
            f = (t - w.start) / DEPLOY_DURATION
            start = ds.pit_exit_s
            s = (start + f * _shortest_delta(start, target, L)) % L
            x, y = track.point_at(s)
            if len(ds.pit_lane_pts):
                px, py = ds.pit_lane_pts[-1]
                bx, by = track.point_at(ds.pit_exit_s)
                x += (1.0 - f) * (px - bx)
                y += (1.0 - f) * (py - by)
            return SCFrame(x=x, y=y, phase="deploy", label="SC DEPLOYING", s=s)

        if t <= w.end:
            # phase 2: circulating ahead of the leader
            x, y = track.point_at(target)
            return SCFrame(x=x, y=y, phase="track", label="SC", s=target)

        # phase 3: handing back over and returning to the pit lane
        f = (t - w.end) / RETURN_DURATION
        end_target = (leader_s + LEAD_AHEAD_METRES) % L
        s = (end_target + f * _shortest_delta(end_target, ds.pit_entry_s, L)) % L
        x, y = track.point_at(s)
        if len(ds.pit_lane_pts):
            px, py = ds.pit_lane_pts[0]
            bx, by = track.point_at(ds.pit_entry_s)
            x += f * (px - bx)
            y += f * (py - by)
        return SCFrame(x=x, y=y, phase="return", label="SC IN", s=s)
    return None
