#!/usr/bin/env python3
"""F1 Race Replay — desktop app entry point.

Examples:
    python main.py                          # 2023 Las Vegas GP race (default)
    python main.py --year 2024 --gp Bahrain --session R
    python main.py --list-cache
    python main.py --clear-cache            # delete all cached data
    python main.py --clear-cache 2023_las_vegas_R
    python main.py --smoke-test out/        # headless render check
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# --session aliases -> FastF1 session ids
SESSION_ALIASES = {
    "r": "R", "race": "R",
    "q": "Q", "qualifying": "Q",
    "s": "S", "sprint": "S",
    "sq": "SQ", "sprintshootout": "SQ", "shootout": "SQ",
    "fp1": "FP1", "p1": "FP1", "practice1": "FP1",
    "fp2": "FP2", "p2": "FP2", "practice2": "FP2",
    "fp3": "FP3", "p3": "FP3", "practice3": "FP3",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="f1replay",
        description="Animated F1 race telemetry replay (FastF1 + Pygame).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                                    # default race\n"
            "  python main.py --year 2024 --gp Bahrain --session Q\n"
            "  python main.py --list-gps 2024                    # valid --gp names\n"
            "  python main.py --list-cache / --clear-cache\n"
        ),
    )
    ap.add_argument("--year", type=int, default=2023, help="season year (default: 2023)")
    ap.add_argument("--gp", default="Las Vegas", help="grand prix name (default: Las Vegas)")
    ap.add_argument("--session", default="R",
                    help="R | Q | S | SQ | FP1 | FP2 | FP3 (default: R)")
    ap.add_argument("--fullscreen", action="store_true", help="open in fullscreen")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore preprocessed cache and rebuild from raw FastF1 data")
    ap.add_argument("--smoke-test", metavar="DIR", nargs="?", const="smoke", default=None,
                    help="render sample frames headlessly into DIR and exit")
    ap.add_argument("--list-gps", nargs="?", const=0, default=None, metavar="YEAR",
                    help="list a season's events (valid --gp names) and exit "
                         "(default year: --year)")
    ap.add_argument("--list-cache", action="store_true", help="list cached sessions and exit")
    ap.add_argument("--clear-cache", action="store_true", help="delete all cached data and exit")
    return ap.parse_args(argv)


def resolve_session(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "").replace("_", "")
    sid = SESSION_ALIASES.get(key)
    if sid is None:
        sys.exit(f"Unknown session {raw!r}. Use one of: "
                 + ", ".join(sorted(set(SESSION_ALIASES.values()))))
    return sid


# --------------------------------------------------------------------------- #
SESSION_SHORT = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "Q", "Race": "R", "Sprint": "S",
    "Sprint Shootout": "SQ", "Sprint Qualifying": "SQ",
}


def list_gps(year: int) -> int:
    """Print a season's event schedule so users know valid --gp values."""
    try:
        import fastf1
    except ImportError:
        sys.exit("fastf1 is required: pip install -r requirements.txt")
    from f1replay import loader

    loader.enable_cache()
    try:
        sched = fastf1.get_event_schedule(year)
    except Exception as exc:  # network / unknown year
        sys.exit(f"Could not load the {year} schedule: {exc}")

    print(f"\n{year} F1 season — copy an EVENT name into --gp\n")
    print(f"{'RND':>3}  {'EVENT':<32} {'DATE':<11}  SESSIONS")
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd == 0:
            continue  # pre-season testing
        date = ev["EventDate"].strftime("%Y-%m-%d")
        sessions = []
        for i in range(1, 6):
            name = ev.get(f"Session{i}")
            if isinstance(name, str) and name:
                sessions.append(SESSION_SHORT.get(name, name))
        print(f"{rnd:>3}  {ev['EventName']:<32} {date:<11}  {' '.join(sessions)}")
    print(f'\nExample: python main.py --year {year} --gp "Bahrain Grand Prix" --session R')
    return 0


# --------------------------------------------------------------------------- #
def loading_window():
    import pygame

    pygame.font.init()
    surf = pygame.display.set_mode((620, 130))
    pygame.display.set_caption("F1 Replay — loading")
    return surf


def draw_loading(surf, lines: list[str]) -> None:
    import pygame

    from f1replay import theme

    surf.fill((16, 18, 22))
    y = 18
    for i, line in enumerate(lines[-4:]):
        theme.text(surf, line, (18, y), 15 if i == len(lines[-4:]) - 1 else 13,
                   theme.TEXT if i == len(lines[-4:]) - 1 else theme.TEXT_DIM)
        y += 26
    pygame.display.flip()
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            sys.exit(0)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    session_id = resolve_session(args.session)

    from f1replay import loader

    if args.list_gps is not None:
        if args.list_gps == 0:
            year = args.year
        else:
            try:
                year = int(args.list_gps)
            except ValueError:
                sys.exit("--list-gps expects a year, e.g. --list-gps 2024")
        return list_gps(year)

    if args.list_cache:
        rows = loader.list_cache()
        if not rows:
            print("Cache is empty.")
        for r in rows:
            print(f"  {r['key']:32s}  preprocessed {loader.human_size(r['preprocessed_bytes'])}"
                  f"   raw-fastf1 {loader.human_size(r['raw_bytes'])}")
        print(f"\nCache dir: {loader.F1_CACHE_DIR}")
        return 0

    if args.clear_cache:
        removed = loader.clear_cache()
        print(f"Removed {len(removed)} cached file(s).")
        return 0

    if args.smoke_test is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame  # noqa: F401  (display driver selection must happen first)
    from f1replay.app import App
    from f1replay.preprocess import load_dataset

    loader.enable_cache()
    surf = None
    lines: list[str] = []

    def progress(msg: str) -> None:
        nonlocal surf
        if surf is None:
            surf = loading_window()
        lines.append(msg)
        draw_loading(surf, lines)

    try:
        ds = load_dataset(
            args.year, args.gp, session_id,
            progress=progress, force_rebuild=args.rebuild,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    if surf is not None:
        pygame.display.quit()

    app = App(ds, fullscreen=args.fullscreen)

    if args.smoke_test is not None:
        saved = app.smoke(args.smoke_test)
        print("Smoke test frames:")
        print(saved)
        print("SMOKE TEST OK")
        pygame.quit()
        return 0

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
