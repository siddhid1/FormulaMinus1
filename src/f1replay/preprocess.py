"""Build the ReplayDataset from a loaded FastF1 session (with pickle cache)."""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .loader import PREPROCESSED_DIR, cache_key, load_session
from .models import (
    PIT_LATERAL_THRESHOLD,
    PIT_SEAM_WINDOW,
    DriverData,
    ReplayDataset,
    SCWindow,
    Stint,
    StatusEvent,
    TrackData,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

SEAM_THRESHOLD = 8.0  # perp distance under this = on racing line
DRS_ZONE_BIN = 50.0  # metres
DRS_ZONE_MIN_COUNT = 3
DRS_ZONE_MIN_LEN = 80.0
DRS_ZONE_MERGE_GAP = 200.0


def load_dataset(
    year: int,
    gp: str,
    session_id: str,
    progress: ProgressFn | None = None,
    force_rebuild: bool = False,
) -> ReplayDataset:
    progress = progress or (lambda msg: None)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PREPROCESSED_DIR / f"{cache_key(year, gp, session_id)}.pkl"
    if path.exists() and not force_rebuild:
        progress(f"Loading cached replay data ({path.name}) ...")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    progress("Downloading session data from FastF1 ...")
    sess = load_session(year, gp, session_id, telemetry=True)
    progress("Preprocessing race data ...")
    ds = build_dataset(sess, year=year, gp=gp, session_id=session_id)
    progress(f"Saving replay data ({path.name}) ...")
    with open(path, "wb") as fh:
        pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ds


# --------------------------------------------------------------------------- #
# track geometry
# --------------------------------------------------------------------------- #

def _dedupe(pts: np.ndarray, min_dist: float = 0.3) -> np.ndarray:
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > min_dist:
            keep.append(i)
    return pts[keep]


def build_track(pos: pd.DataFrame) -> TrackData:
    """Reference polyline from one lap's position samples (X/Y in decimetres)."""
    pts = pos[["X", "Y"]].to_numpy(dtype=np.float64) / 10.0
    pts = _dedupe(pts)
    if len(pts) < 20:
        raise ValueError("not enough position samples to build track")
    if np.hypot(*(pts[0] - pts[-1])) > 0.3:
        pts = np.vstack([pts, pts[0]])
    seg = np.diff(pts, axis=0)
    slen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(slen)])
    return TrackData(pts=pts, cum=cum, length=float(cum[-1]))


class Projector:
    """Projects world (x, y) onto the track polyline -> (s, perp distance)."""

    def __init__(self, track: TrackData):
        self.verts = track.pts[:-1]  # unique vertices, loop closed by modulo
        self.n = len(self.verts)
        self.cum = track.cum[:-1]
        nxt = np.roll(self.verts, -1, axis=0)
        self.slen = np.hypot(*(nxt - self.verts).T)
        self.length = track.length
        self.tree = cKDTree(self.verts)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(xy)
        _, idx = self.tree.query(xy)
        a = self.verts[idx]
        b = self.verts[(idx + 1) % self.n]
        fwd = self._proj(xy, a, b, self.cum[idx], self.slen[idx])

        ib = (idx - 1) % self.n
        a2 = self.verts[ib]
        b2 = self.verts[idx]
        bwd = self._proj(xy, a2, b2, self.cum[ib], self.slen[ib])

        use_bwd = bwd[1] < fwd[1]
        s = np.where(use_bwd, bwd[0], fwd[0])
        perp = np.where(use_bwd, bwd[1], fwd[1])
        return np.mod(s, self.length), perp

    @staticmethod
    def _proj(
        xy: np.ndarray, a: np.ndarray, b: np.ndarray, s_a: np.ndarray, seg_len: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ab = b - a
        denom = (ab * ab).sum(axis=1)
        denom[denom == 0] = 1e-9
        f = np.clip(((xy - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        closest = a + f[:, None] * ab
        s = s_a + f * seg_len
        perp = np.linalg.norm(xy - closest, axis=1)
        return s, perp


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _td_seconds(v) -> float:
    if pd.isna(v):
        return float("nan")
    return float(pd.Timedelta(v).total_seconds())


def _integrate_progress(s: np.ndarray, length: float) -> np.ndarray:
    """Unwrap track distance into race progress by integrating step deltas.

    Jumps larger than half a lap are projection artefacts or seam crossings
    and get folded into the correct forward direction, so neither pit-lane
    seam crossings nor start/finish jitter can fake extra laps.
    """
    out = np.empty(len(s), dtype=np.float64)
    if len(s) == 0:
        return out
    out[0] = s[0]
    if len(s) == 1:
        return out
    ds = np.diff(s.astype(np.float64))
    ds[ds < -0.5 * length] += length  # forward crossing of the seam
    ds[ds > 0.5 * length] -= length  # backward projection glitch
    out[1:] = s[0] + np.cumsum(ds)
    return out


def _stints_from_laps(lap_numbers: np.ndarray, compounds: np.ndarray) -> list[Stint]:
    stints: list[Stint] = []
    cur: str | None = None
    start = 0
    for i, (lap, comp) in enumerate(zip(lap_numbers, compounds)):
        comp = str(comp)
        if cur is None:
            cur, start = comp, int(lap)
        elif comp != cur:
            stints.append(Stint(lap_from=start, lap_to=int(lap_numbers[i - 1]), compound=cur))
            cur, start = comp, int(lap)
    if cur is not None and len(lap_numbers):
        stints.append(Stint(lap_from=start, lap_to=int(lap_numbers[-1]), compound=cur))
    return stints


def _drs_text(drs: float) -> str:
    d = int(drs)
    if d <= 1:
        return "OFF"
    if 8 <= d <= 10:
        return "AVAIL"
    return "ON"


# --------------------------------------------------------------------------- #
# main build
# --------------------------------------------------------------------------- #

def build_dataset(sess, *, year: int, gp: str, session_id: str) -> ReplayDataset:
    results = sess.results

    # ---- track ------------------------------------------------------------
    track = _build_track_robust(sess)
    proj = Projector(track)
    L = track.length
    logger.info("Track length: %.1f m (%d vertices)", L, len(track.pts))

    # ---- status events / SC windows --------------------------------------
    status_events: list[StatusEvent] = []
    for _, row in sess.track_status.iterrows():
        status_events.append(
            StatusEvent(
                t=_td_seconds(row["Time"]),
                status=str(row["Status"]),
                message=str(row.get("Message", "")),
            )
        )
    status_events.sort(key=lambda e: e.t)
    sc_windows: list[SCWindow] = []
    for i, ev in enumerate(status_events):
        if ev.status == "4":
            end = status_events[i + 1].t if i + 1 < len(status_events) else _session_end(sess)
            sc_windows.append(SCWindow(start=ev.t, end=end))
    logger.info("SC windows: %s", [(round(w.start), round(w.end)) for w in sc_windows])

    # ---- session timing ---------------------------------------------------
    ss = sess.session_status
    started = next((_td_seconds(r["Time"]) for _, r in ss.iterrows() if r["Status"] == "Started"), 0.0)
    finished = next((_td_seconds(r["Time"]) for _, r in ss.iterrows() if r["Status"] == "Finished"), None)
    ends = next((_td_seconds(r["Time"]) for _, r in ss.iterrows() if r["Status"] == "Ends"), None)

    # ---- per driver -------------------------------------------------------
    drivers: dict[str, DriverData] = {}
    drs_hits: list[np.ndarray] = []
    pit_entry_cands: list[float] = []
    pit_exit_cands: list[float] = []
    pit_samples_xy: list[np.ndarray] = []
    all_last_t: list[float] = []
    all_first_t: list[float] = []

    for number in results["DriverNumber"].astype(str):
        if number not in sess.pos_data:
            logger.warning("No position data for driver %s", number)
            continue
        car = sess.car_data[number]
        pos = sess.pos_data[number]
        tel = car.merge_channels(pos, frequency="original")
        tel = tel.dropna(subset=["X", "Y"]).sort_values("Date")
        tel = tel[~tel["Date"].duplicated(keep="last")]
        if len(tel) < 10:
            logger.warning("Driver %s: too few samples", number)
            continue

        t = tel["SessionTime"].dt.total_seconds().to_numpy(dtype=np.float64)
        xy = tel[["X", "Y"]].to_numpy(dtype=np.float64) / 10.0
        speed = tel["Speed"].to_numpy(dtype=np.float64)
        gear = tel["nGear"].to_numpy(dtype=np.float64)
        drs = tel["DRS"].to_numpy(dtype=np.float64)
        on_track_status = tel["Status"].to_numpy()

        s_track, perp = proj.project(xy)

        # pit-lane samples near the S/F seam (for pit lane geometry)
        seam_dist = np.minimum(s_track, L - s_track)
        pit_hold = (perp > PIT_LATERAL_THRESHOLD) & (seam_dist < PIT_SEAM_WINDOW)
        pit_hold |= (on_track_status == "OffTrack") & (seam_dist < PIT_SEAM_WINDOW)

        race_dist = _integrate_progress(s_track, L)

        # DRS zone evidence (on-track samples only)
        zone_mask = (drs >= 8) & (perp < SEAM_THRESHOLD)
        if zone_mask.any():
            drs_hits.append(s_track[zone_mask])

        # pit entry / exit candidates
        near = seam_dist < PIT_SEAM_WINDOW
        if near.any():
            off = perp > PIT_LATERAL_THRESHOLD
            for i in range(1, len(t)):
                if not (near[i] and near[i - 1]):
                    continue
                if s_track[i] > 0.5 * L and not off[i - 1] and off[i]:
                    pit_entry_cands.append(s_track[i])
                if s_track[i] < 0.5 * L and off[i - 1] and not off[i]:
                    pit_exit_cands.append(s_track[i])
            pit_samples_xy.append(xy[pit_hold & (perp > PIT_LATERAL_THRESHOLD)])

        # laps
        laps = sess.laps.pick_drivers(str(number)).sort_values("LapNumber")
        lap_starts = laps["LapStartTime"].map(_td_seconds).to_numpy(dtype=np.float64)
        lap_numbers = laps["LapNumber"].to_numpy(dtype=np.int64)
        lap_times = laps["LapTime"].map(
            lambda v: float("nan") if pd.isna(v) else pd.Timedelta(v).total_seconds()
        ).to_numpy(dtype=np.float64)
        compounds = laps["Compound"].astype(str).to_numpy()

        row = results[results["DriverNumber"].astype(str) == number].iloc[0]
        status = str(row["Status"])
        out_time = _retirement_time(t, xy, status) if status != "Finished" else None

        drv = DriverData(
            number=number,
            code=str(row["Abbreviation"]),
            name=str(row["BroadcastName"]),
            team=str(row["TeamName"]),
            color="#" + str(row["TeamColor"]).lstrip("#"),
            grid=int(row["GridPosition"]) if not pd.isna(row["GridPosition"]) else 99,
            final_position=str(int(row["Position"])) if not pd.isna(row["Position"]) else "",
            final_status=status,
            t=t.astype(np.float32),
            x=xy[:, 0].astype(np.float32),
            y=xy[:, 1].astype(np.float32),
            speed=speed.astype(np.float32),
            gear=gear.astype(np.float32),
            drs=drs.astype(np.float32),
            s=s_track.astype(np.float32),
            race_dist=race_dist.astype(np.float32),
            lap_starts=lap_starts,
            lap_numbers=lap_numbers,
            lap_times=lap_times,
            compounds=compounds,
            stints=_stints_from_laps(lap_numbers, compounds),
            out_time=out_time,
        )
        drivers[number] = drv
        all_first_t.append(float(t[0]))
        all_last_t.append(float(t[-1]))

    # ---- DRS zones --------------------------------------------------------
    drs_zones = _extract_drs_zones(drs_hits, L)

    # ---- pit lane geometry ------------------------------------------------
    pit_entry_s = float(np.median(pit_entry_cands)) if pit_entry_cands else (L - 200.0) % L
    pit_exit_s = float(np.median(pit_exit_cands)) if pit_exit_cands else 200.0
    pit_lane_pts = _pit_lane_polyline(pit_samples_xy, proj, L)

    # ---- corners ----------------------------------------------------------
    corners = _load_corners(sess, track)

    # ---- time range -------------------------------------------------------
    t0 = max(0.0, started - 20.0)
    last_sample = max(all_last_t) if all_last_t else (ends or 0.0)
    if finished is not None:
        t1 = min(last_sample, finished + 60.0)
    else:
        t1 = last_sample

    order = [
        str(n)
        for n in sorted(
            results["DriverNumber"].astype(str),
            key=lambda n: _class_sort_key(results, n),
        )
        if n in drivers
    ]
    total_laps = int(np.nanmax(results["Laps"].to_numpy(dtype=np.float64)))

    title = f"{sess.event['EventName']} {year} — {sess.name}"
    return ReplayDataset(
        session_title=title,
        year=year,
        gp=gp,
        session_id=session_id,
        track=track,
        drivers=drivers,
        order=order,
        t0=t0,
        t1=t1,
        total_laps=total_laps,
        finished_at=finished,
        sc_windows=sc_windows,
        status_events=status_events,
        drs_zones=drs_zones,
        pit_entry_s=pit_entry_s,
        pit_exit_s=pit_exit_s,
        pit_lane_pts=pit_lane_pts,
        corners=corners,
    )


# --------------------------------------------------------------------------- #
# helpers (part 2)
# --------------------------------------------------------------------------- #

def _build_track_robust(sess) -> TrackData:
    """Fastest lap first; fall back to other drivers' laps if it looks broken."""
    candidates = [sess.laps.pick_fastest()]
    best: TrackData | None = None
    for lap in candidates:
        try:
            track = build_track(lap.get_pos_data())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Track build failed (%s)", exc)
            continue
        if best is None or track.length > best.length:
            best = track
    if best is None or not (1000 < best.length < 30000):
        # last resort: longest lap's position data
        for _, lap in sess.laps.iterrows():
            try:
                track = build_track(lap.get_pos_data())
            except Exception:  # noqa: BLE001
                continue
            if best is None or track.length > best.length:
                best = track
            if best and best.length > 3000:
                break
    if best is None:
        raise RuntimeError("Could not build track outline from session data")
    return best


def _class_sort_key(results, number: str):
    row = results[results["DriverNumber"].astype(str) == number].iloc[0]
    pos = row["Position"]
    return int(pos) if not pd.isna(pos) else 999


def _session_end(sess) -> float:
    ss = sess.session_status
    for _, r in ss.iterrows():
        if r["Status"] in ("Ends", "Finalised"):
            return _td_seconds(r["Time"])
    return float("nan")


def _retirement_time(t: np.ndarray, xy: np.ndarray, status: str) -> float | None:
    """Time a driver went out: last moment the car was still moving."""
    if status == "Finished":
        return None
    if len(t) < 2:
        return None
    disp = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    moving = disp > 3.0  # >3 m between consecutive samples
    if moving.any():
        last_move = float(t[1:][moving][-1])
    else:
        last_move = float(t[-1])
    # car parked in the pits long before the session ended -> use last movement
    if float(t[-1]) - last_move > 60.0:
        return last_move + 1.0
    return float(t[-1]) + 1.0


def _extract_drs_zones(hits: list[np.ndarray], L: float) -> list[tuple[float, float]]:
    if not hits:
        return []
    all_s = np.concatenate(hits)
    nb = max(4, int(L // DRS_ZONE_BIN))
    hist, _ = np.histogram(all_s, bins=nb, range=(0.0, L))
    peak = int(hist.max())
    if peak <= DRS_ZONE_MIN_COUNT:
        return []
    # require a meaningful share of the busiest bin: kills scattered noise
    threshold = max(DRS_ZONE_MIN_COUNT, int(np.ceil(0.04 * peak)))
    active = hist >= threshold
    if not active.any():
        return []
    idx = np.nonzero(active)[0]
    runs: list[list[float]] = [[idx[0], idx[0] + 1]]
    for i in idx[1:]:
        if i <= runs[-1][1]:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    zones = [(a * DRS_ZONE_BIN, b * DRS_ZONE_BIN) for a, b in runs]
    # merge zones separated by a small gap; also across the seam
    merged: list[list[float]] = []
    for zs, ze in zones:
        if merged and zs - merged[-1][1] <= DRS_ZONE_MERGE_GAP:
            merged[-1][1] = ze
        else:
            merged.append([zs, ze])
    if len(merged) > 1:
        if (L - merged[-1][1]) + merged[0][0] <= DRS_ZONE_MERGE_GAP:
            first = merged.pop(0)
            merged[-1][1] = first[1] + L
    result = [(a, b) for a, b in merged if (b - a) >= DRS_ZONE_MIN_LEN]
    logger.info("DRS zones: %s", [(round(a), round(b)) for a, b in result])
    return result


def _pit_lane_polyline(
    pit_samples: list[np.ndarray], proj: Projector, L: float
) -> np.ndarray:
    """Average pit-lane samples into a coarse polyline near the seam."""
    if not pit_samples:
        return np.zeros((0, 2))
    xy = np.vstack(pit_samples)
    if len(xy) == 0:
        return np.zeros((0, 2))
    s, _ = proj.project(xy)
    seam_dist = np.minimum(s, L - s)
    mask = seam_dist < PIT_SEAM_WINDOW
    xy, s = xy[mask], s[mask]
    if len(xy) < 10:
        return np.zeros((0, 2))
    # bin by s (10 m) and average positions
    s_adj = np.where(s > 0.5 * L, s - L, s)  # seam-centred coordinate
    bins = np.floor(s_adj / 10.0).astype(int)
    pts = []
    for b in sorted(set(bins.tolist())):
        sel = bins == b
        if sel.sum() >= 3:
            pts.append(xy[sel].mean(axis=0))
    return np.asarray(pts, dtype=np.float64) if len(pts) >= 4 else np.zeros((0, 2))


def _load_corners(sess, track: TrackData) -> np.ndarray:
    """Corner markers (x, y, number) in metres; tolerant of unit systems."""
    try:
        ci = sess.get_circuit_info()
        if ci is None or ci.corners is None or not len(ci.corners):
            return np.zeros((0, 3))
    except Exception:  # noqa: BLE001
        return np.zeros((0, 3))
    corners = ci.corners[["X", "Y", "Number"]].copy()
    xy = corners[["X", "Y"]].to_numpy(dtype=np.float64)

    def score(pts: np.ndarray) -> float:
        from scipy.spatial import cKDTree as _KD

        tree = _KD(track.pts)
        d, _ = tree.query(pts)
        return float(np.median(d))

    cands = {"raw": xy, "dm_to_m": xy / 10.0, "m_to_dm": xy * 10.0}
    best_name, best_score = "dm_to_m", float("inf")
    for name, cand in cands.items():
        sc = score(cand)
        if sc < best_score:
            best_name, best_score = name, sc
    if best_score > 60.0:
        return np.zeros((0, 3))
    xy = cands[best_name]
    out = np.column_stack([xy, corners["Number"].to_numpy(dtype=np.float64)])
    return out.astype(np.float64)
