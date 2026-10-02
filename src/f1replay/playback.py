"""Playback clock and per-frame state sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import DriverData, ReplayDataset
from .safety_car import SCFrame, compute_sc

SPEEDS = [0.5, 1.0, 2.0, 4.0]

STATUS_TEXT = {
    "1": "GREEN",
    "2": "YELLOW",
    "4": "SAFETY CAR",
    "5": "RED FLAG",
    "6": "VSC",
    "7": "VSC ENDING",
}


@dataclass
class DriverFrame:
    number: str
    code: str
    name: str
    team: str
    color: str
    x: float
    y: float
    speed: float
    gear: int
    drs: int
    drs_text: str
    lap: int
    compound: str
    rank: int
    out: bool
    race_dist: float


@dataclass
class FrameState:
    t: float
    elapsed: float  # race clock (from lights out)
    drivers: list[DriverFrame]
    by_number: dict[str, DriverFrame]
    leader_number: str
    lap_current: int
    lap_total: int
    status_code: str
    status_text: str
    sc: SCFrame | None
    in_sc_window: bool


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def _sample_scalar(times: np.ndarray, values: np.ndarray, t: float) -> float:
    i = int(np.searchsorted(times, t, side="right"))
    if i <= 0:
        return float(values[0])
    if i >= len(times):
        return float(values[-1])
    t0 = float(times[i - 1])
    t1 = float(times[i])
    if t1 <= t0:
        return float(values[i])
    f = (t - t0) / (t1 - t0)
    return float(_lerp(float(values[i - 1]), float(values[i]), f))


def _sample_hold(times: np.ndarray, values: np.ndarray, t: float) -> float:
    i = int(np.searchsorted(times, t, side="right"))
    if i <= 0:
        return float(values[0])
    return float(values[min(i, len(times)) - 1])


class Playback:
    def __init__(self, ds: ReplayDataset):
        self.ds = ds
        self.t = ds.t0
        self.playing = True
        self.speed_index = 1  # 1.0x

    # ---- clock ----------------------------------------------------------- #
    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_index]

    def cycle_speed(self, direction: int = 1) -> float:
        self.speed_index = (self.speed_index + direction) % len(SPEEDS)
        return self.speed

    def set_speed(self, index: int) -> float:
        if 0 <= index < len(SPEEDS):
            self.speed_index = index
        return self.speed

    def toggle_play(self) -> bool:
        self.playing = not self.playing
        if self.t >= self.ds.t1 and self.playing:
            self.t = self.ds.t0
        return self.playing

    def update(self, dt: float) -> None:
        if not self.playing:
            return
        self.t += dt * self.speed
        if self.t >= self.ds.t1:
            self.t = self.ds.t1
            self.playing = False

    def seek(self, delta: float) -> None:
        self.t = float(np.clip(self.t + delta, self.ds.t0, self.ds.t1))

    def seek_fraction(self, frac: float) -> None:
        self.t = float(np.clip(self.ds.t0 + frac * (self.ds.t1 - self.ds.t0), self.ds.t0, self.ds.t1))

    def restart(self) -> None:
        self.t = self.ds.t0
        self.playing = True

    # ---- frame ------------------------------------------------------------ #
    def frame(self) -> FrameState:
        ds = self.ds
        t = self.t
        frames: list[DriverFrame] = []
        for number, d in ds.drivers.items():
            if len(d.t) == 0:
                continue
            x = _sample_scalar(d.t, d.x, t)
            y = _sample_scalar(d.t, d.y, t)
            speed = _sample_scalar(d.t, d.speed, t)
            gear = int(round(_sample_hold(d.t, d.gear, t)))
            drs = int(round(_sample_hold(d.t, d.drs, t)))
            rd = _sample_scalar(d.t, d.race_dist, t)
            frames.append(
                DriverFrame(
                    number=number,
                    code=d.code,
                    name=d.name,
                    team=d.team,
                    color=d.color,
                    x=x,
                    y=y,
                    speed=speed,
                    gear=gear,
                    drs=drs,
                    drs_text=_drs_text(drs),
                    lap=d.lap_at(t),
                    compound=d.compound_at(t),
                    rank=0,
                    out=d.is_out(t),
                    race_dist=rd,
                )
            )

        if ds.finished_at is not None and t >= ds.finished_at:
            # official classification once the chequered flag is out
            pos = {n: i for i, n in enumerate(ds.order)}
            frames.sort(key=lambda f: pos.get(f.number, 999))
        else:
            frames.sort(key=lambda f: (-f.race_dist, f.number))
        for i, f in enumerate(frames):
            f.rank = i + 1

        by_number = {f.number: f for f in frames}
        leader_number = frames[0].number if frames else ""
        active = [f for f in frames if not f.out and f.lap > 0]
        lap_current = max((f.lap for f in active), default=1)

        code, text = _current_status(ds, t)
        # the SC anchors to the leader's track distance (not race progress)
        leader_s = _leader_track_s(ds, t, leader_number)
        sc = compute_sc(ds, t, leader_s)
        in_sc = any(w.start <= t <= w.end for w in ds.sc_windows)

        return FrameState(
            t=t,
            elapsed=max(0.0, t - (ds.t0 + 20.0)),
            drivers=frames,
            by_number=by_number,
            leader_number=leader_number,
            lap_current=lap_current,
            lap_total=ds.total_laps,
            status_code=code,
            status_text=text,
            sc=sc,
            in_sc_window=in_sc,
        )


def _drs_text(drs: int) -> str:
    if drs <= 1:
        return "OFF"
    if 8 <= drs <= 10:
        return "AVAIL"
    return "ON"


def _current_status(ds: ReplayDataset, t: float) -> tuple[str, str]:
    code = "1"
    for ev in ds.status_events:
        if ev.t <= t:
            code = ev.status
        else:
            break
    return code, STATUS_TEXT.get(code, code)


def _leader_track_s(ds: ReplayDataset, t: float, leader_number: str) -> float:
    d = ds.drivers.get(leader_number)
    if d is None or len(d.t) == 0:
        return 0.0
    return float(_sample_scalar(d.t, d.s, t))
