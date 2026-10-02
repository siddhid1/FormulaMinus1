"""Core data structures for the F1 race replay dataset."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID_SLOT_SPACING = 8.0  # metres between grid slots
PIT_LATERAL_THRESHOLD = 10.0  # metres off the racing line to count as pit lane
PIT_SEAM_WINDOW = 1000.0  # only treat off-line samples this close to S/F as pit lane


@dataclass
class TrackData:
    """Reference polyline of the circuit (closed loop, metres)."""

    pts: np.ndarray  # (N, 2) float64, pts[-1] == pts[0]
    cum: np.ndarray  # (N,) cumulative distance along loop; cum[-1] == length
    length: float

    def __post_init__(self) -> None:
        if len(self.pts) != len(self.cum):
            raise ValueError("pts and cum must have equal length")

    def point_at(self, s: float) -> tuple[float, float]:
        """Interpolate (x, y) at track distance s (metres, wraps)."""
        s = float(s) % self.length
        i = int(np.searchsorted(self.cum, s, side="right"))
        if i <= 0:
            return float(self.pts[0, 0]), float(self.pts[0, 1])
        if i >= len(self.cum):
            return float(self.pts[-1, 0]), float(self.pts[-1, 1])
        s0 = self.cum[i - 1]
        s1 = self.cum[i]
        f = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        p0 = self.pts[i - 1]
        p1 = self.pts[i]
        return float(p0[0] + f * (p1[0] - p0[0])), float(p0[1] + f * (p1[1] - p0[1]))

    def forward_offset(self, s_from: float, dist: float) -> float:
        """s reached by moving `dist` metres forward from `s_from` (wraps)."""
        return (float(s_from) + float(dist)) % self.length


@dataclass
class Stint:
    lap_from: int
    lap_to: int  # inclusive
    compound: str


@dataclass
class SCWindow:
    """A safety car period derived from track status code '4'."""

    start: float  # seconds from session start
    end: float


@dataclass
class StatusEvent:
    t: float
    status: str  # numeric track status code as string
    message: str


@dataclass
class DriverData:
    number: str
    code: str
    name: str
    team: str
    color: str  # '#RRGGBB'
    grid: int  # grid position (1-based)
    final_position: str  # classification, e.g. '1', 'NC', ''
    final_status: str  # results Status column

    # time-indexed arrays (seconds from session start), sorted by t
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    gear: np.ndarray
    drs: np.ndarray
    s: np.ndarray  # distance along track (0..length)
    race_dist: np.ndarray  # unwrapped progress for ranking

    # lap timing (sorted by lap start time)
    lap_starts: np.ndarray  # (K,) lap start times
    lap_numbers: np.ndarray  # (K,) int
    lap_times: np.ndarray  # (K,) lap time seconds (nan if unknown)
    compounds: np.ndarray  # (K,) object: compound string per lap

    stints: list[Stint] = field(default_factory=list)
    out_time: float | None = None  # when replay should mark this driver OUT

    @property
    def last_t(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def is_retired(self) -> bool:
        return self.out_time is not None

    def lap_at(self, t: float) -> int:
        """Current lap number at time t (0 before the first lap starts)."""
        if len(self.lap_starts) == 0 or t < self.lap_starts[0]:
            return 0
        i = int(np.searchsorted(self.lap_starts, t, side="right"))
        return int(self.lap_numbers[min(i, len(self.lap_numbers) - 1)])

    def compound_at(self, t: float) -> str:
        if len(self.lap_starts) == 0:
            return ""
        if t < self.lap_starts[0]:
            return str(self.compounds[0])
        i = int(np.searchsorted(self.lap_starts, t, side="right"))
        return str(self.compounds[min(i, len(self.compounds) - 1)])

    def is_out(self, t: float) -> bool:
        return self.out_time is not None and t >= self.out_time


@dataclass
class ReplayDataset:
    session_title: str
    year: int
    gp: str
    session_id: str
    track: TrackData
    drivers: dict[str, DriverData]  # keyed by driver number
    order: list[str]  # driver numbers in classified order
    t0: float
    t1: float
    total_laps: int
    finished_at: float | None  # chequered flag time (official order after this)
    sc_windows: list[SCWindow]
    status_events: list[StatusEvent]
    drs_zones: list[tuple[float, float]]  # (s_from, s_to)
    pit_entry_s: float
    pit_exit_s: float
    pit_lane_pts: np.ndarray  # (M, 2) pit lane reference points or empty
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # x, y, number

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)
