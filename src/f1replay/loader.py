"""FastF1 session download / cache management."""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache"
F1_CACHE_DIR = CACHE_ROOT / "fastf1"
PREPROCESSED_DIR = CACHE_ROOT / "preprocessed"

_session: "fastf1.core.Session | None" = None


def enable_cache() -> None:
    F1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(F1_CACHE_DIR))


def load_session(year: int, gp: str, session_id: str, *, telemetry: bool = True):
    """Download (and cache) a session. Returns a loaded fastf1 Session."""
    global _session
    enable_cache()
    logger.info("Loading %s %s %s ...", year, gp, session_id)
    sess = fastf1.get_session(year, gp, session_id)
    logger.info("Downloading lap timing data ...")
    sess.load(laps=True, messages=True)
    if telemetry:
        logger.info("Downloading telemetry / position data ...")
        sess.load(telemetry=True)
    _session = sess
    return sess


def cache_key(year: int, gp: str, session_id: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in gp).strip("_").lower()
    return f"{year}_{safe}_{session_id.upper()}"


def list_cache() -> list[dict]:
    """Return per-session cache usage (raw FastF1 cache + preprocessed file)."""
    rows = []
    if PREPROCESSED_DIR.exists():
        for p in sorted(PREPROCESSED_DIR.glob("*.pkl")):
            rows.append(
                {
                    "key": p.stem,
                    "preprocessed_bytes": p.stat().st_size,
                    "raw_bytes": _raw_cache_bytes(),
                }
            )
    return rows


def _raw_cache_bytes() -> int:
    if not F1_CACHE_DIR.exists():
        return 0
    return sum(f.stat().st_size for f in F1_CACHE_DIR.rglob("*") if f.is_file())


def clear_cache(key: str | None = None) -> list[str]:
    """Delete cached data. `key` selects one preprocessed session; raw FastF1
    cache is only removed when clearing everything."""
    removed: list[str] = []
    if key is None:
        if PREPROCESSED_DIR.exists():
            for p in PREPROCESSED_DIR.glob("*.pkl"):
                p.unlink()
                removed.append(str(p))
        if F1_CACHE_DIR.exists():
            for f in F1_CACHE_DIR.rglob("*"):
                if f.is_file():
                    f.unlink()
                    removed.append(str(f))
    else:
        target = PREPROCESSED_DIR / f"{key}.pkl"
        if target.exists():
            target.unlink()
            removed.append(str(target))
    return removed


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"
