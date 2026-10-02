"""Headless input/control regression tests for the F1 Replay app.

Runs fully offline against the preprocessed cache (default: 2023 Las Vegas
race) with SDL's dummy video driver — no window is opened.

    python tests/test_controls.py

Covers every control path (keyboard, leaderboard, transport, scrubbing,
insights menu, panel drag/close, track-map picking) plus two crash
regressions that were found on a real display:
  * panel close-click followed by mouse motion (TypeError on None offset)
  * mouse events delivered WITHOUT a `mod` attribute (real SDL shape)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pygame

from f1replay.app import App
from f1replay.preprocess import load_dataset

E = pygame.event.post


def main() -> None:
    pygame.init()
    ds = load_dataset(2023, "Las Vegas", "R")
    app = App(ds)

    def key(k, mod=0):
        E(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod))

    def click(x, y, shift=False):
        # real SDL mouse events have NO `mod` attribute (that was the crash);
        # shift state comes from pygame.key.get_mods(), so set/reset it here
        if shift:
            pygame.key.set_mods(pygame.KMOD_SHIFT)
        try:
            E(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x, y)))
            E(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(x, y)))
            app.handle_events()
        finally:
            pygame.key.set_mods(0)

    def busy_rects():
        """UI rects that swallow clicks (menus, panel headers/close buttons)."""
        rects = [app.menu.rect]
        for p in [app.telemetry, app.chart, app.stints]:
            if p.visible:
                rects.append(p.header_rect().inflate(8, 8))
                rects.append(p.close_rect())
        return rects

    def free_car(exclude=()):
        for n, (x, y) in app.view.car_screen.items():
            if n in exclude:
                continue
            if not any(r.collidepoint(x, y) for r in busy_rects()):
                return n, x, y
        raise AssertionError("no car outside panels")

    # initial draw populates menu_regions / board_regions
    app.draw(app.screen)

    # ---- keyboard (B deliberately excluded; tested later)
    for k in [pygame.K_SPACE, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_UP, pygame.K_DOWN,
              pygame.K_1, pygame.K_3, pygame.K_l, pygame.K_t, pygame.K_g, pygame.K_y,
              pygame.K_r, pygame.K_SPACE]:
        key(k)
    app.handle_events()
    app.draw(app.screen)
    assert app.states["telemetry"] and app.states["chart"] and app.states["stints"]
    assert not app.hud.show_names

    # ---- leaderboard click / shift-click / replace / deselect
    app.hud.layout(1440, 900)
    row0, row1 = app.hud.board_regions[0], app.hud.board_regions[1]
    click(*row0[0].center)
    assert app.selected == {row0[1]}, app.selected
    click(*row1[0].center, shift=True)
    assert len(app.selected) == 2, app.selected
    click(*row0[0].center)
    assert app.selected == {row0[1]}, app.selected
    click(*row0[0].center)
    assert app.selected == set(), "re-click should deselect"

    # ---- transport buttons (play/seek/speed/restart all hit)
    for rect, action in list(app.hud.transport_regions):
        click(*rect.center)
    assert 0 <= app.playback.speed_index < 4

    # ---- progress scrub
    tr = app.hud.progress_track
    click(tr.x + tr.w // 2, tr.centery)
    frac = (app.playback.t - ds.t0) / ds.duration
    assert 0.4 < frac < 0.6, frac

    # ---- insights menu toggles
    n_before = dict(app.states)
    for rect, attr in list(app.menu_regions):
        click(*rect.center)
    assert not app.states["drs"], "menu click should turn DRS off"
    assert app.states["telemetry"] is not n_before["telemetry"]

    # ---- panel drag (re-open telemetry/chart first: menu loop closed them)
    app.states["telemetry"] = app.telemetry.visible = True
    app.states["chart"] = app.chart.visible = True
    app.draw(app.screen)
    hd = app.telemetry.header_rect()
    E(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=hd.center))
    app.handle_events()
    old = app.telemetry.rect.topleft
    E(pygame.event.Event(pygame.MOUSEMOTION, pos=(hd.x + 300, hd.y + 150),
                         rel=(300, 150), buttons=(1, 0, 0)))
    E(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(hd.x + 300, hd.y + 150)))
    app.handle_events()
    assert app.telemetry.rect.topleft != old, "panel should have moved"

    # ---- close a panel via X
    cr = app.chart.close_rect()
    click(*cr.center)
    assert not app.chart.visible

    # ---- car pick on track map (avoid panels)
    app.draw(app.screen)
    n1, x1, y1 = free_car()
    click(x1, y1)
    assert app.selected == {n1}, (n1, app.selected)
    n2, x2, y2 = free_car(exclude={n1})
    click(x2, y2, shift=True)
    assert app.selected == {n1, n2}, app.selected

    # ---- empty viewport corner clears selection
    corner = (app.hud.viewport.right - 6, app.hud.viewport.bottom - 6)
    assert app.hud.viewport.collidepoint(corner)
    click(*corner)
    if app.view.pick(*corner) is None:
        assert app.selected == set(), app.selected

    # ---- run frames end-to-end
    for _ in range(5):
        app.handle_events()
        app.update(1.0 / 60)
        app.draw()
        pygame.display.flip()

    # ---- end-of-race stops playback; restart returns to t0
    app.playback.t = ds.t1 - 1
    app.playback.update(1.0)
    assert not app.playback.playing, "should stop at the end"
    app.transport("restart")
    assert app.playback.t == ds.t0 and app.playback.playing

    # ---- B toggles progress bar + relayout
    key(pygame.K_b)
    app.handle_events()
    app.draw(app.screen)
    assert not app.hud.show_progress and app.hud.progress_region is None
    key(pygame.K_b)
    app.handle_events()
    app.draw(app.screen)
    assert app.hud.show_progress and app.hud.progress_region is not None

    # ---- regression 1: close-click + mouse motion must not crash
    app.states["telemetry"] = app.telemetry.visible = True
    app.draw(app.screen)
    hd = app.telemetry.header_rect()
    E(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=hd.center))
    E(pygame.event.Event(pygame.MOUSEMOTION, pos=(hd.x + 40, hd.y + 40),
                         rel=(40, 40), buttons=(1, 0, 0)))
    E(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(hd.x + 40, hd.y + 40)))
    app.handle_events()
    assert app.dragging is None
    cr = app.telemetry.close_rect()
    click(*cr.center)
    assert not app.telemetry.visible and app.dragging is None
    E(pygame.event.Event(pygame.MOUSEMOTION, pos=(cr.x + 2, cr.y + 2),
                         rel=(2, 0), buttons=(0, 0, 0)))
    app.handle_events()  # would raise TypeError before the fix
    app.draw(app.screen)
    assert app.states["telemetry"] is False, "menu checkbox must sync with X close"

    # ---- regression 2: close-click without any prior drag must also be safe
    app.states["stints"] = app.stints.visible = True
    app.draw(app.screen)
    cr2 = app.stints.close_rect()
    click(*cr2.center)
    E(pygame.event.Event(pygame.MOUSEMOTION, pos=(cr2.x + 1, cr2.y + 1),
                         rel=(1, 0), buttons=(0, 0, 0)))
    app.handle_events()
    app.draw(app.screen)
    assert not app.stints.visible and app.states["stints"] is False
    assert app.dragging is None

    # ---- ESC quits
    key(pygame.K_ESCAPE)
    app.handle_events()
    assert not app.running

    pygame.quit()
    print("INPUT TEST OK")


if __name__ == "__main__":
    main()
