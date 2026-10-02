"""Application shell: window, event loop, input handling, draw orchestration."""
from __future__ import annotations

import sys

import pygame

from .hud import Hud
from .models import ReplayDataset
from .panels import FloatingPanel, InsightsMenu, LapChartPanel, StintsPanel, TelemetryPanel
from .playback import Playback
from .theme import SC_COLOR  # noqa: F401  (re-exported for smoke tests)
from .track_render import TrackView

WIDTH, HEIGHT = 1440, 900
SCRUB_RATE = 45.0  # seconds of replay per second while holding LEFT/RIGHT
SEEK_STEP = 5.0
PANEL_RECTS = {
    "menu": (14, 68, 244, 0),
    "telemetry": (14, 320, 268, 0),
    "chart": (340, 110, 620, 330),
    "stints": (340, 470, 620, 0),
}


class App:
    def __init__(self, ds: ReplayDataset, *, fullscreen: bool = False):
        pygame.display.set_caption(f"F1 Replay — {ds.session_title}")
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        self.clock = pygame.time.Clock()
        self.ds = ds
        self.playback = Playback(ds)
        self.hud = Hud(ds)
        self.view = TrackView(ds)
        self.selected: set[str] = set()
        self.mouse = (-999, -999)
        self.running = True
        self.dragging: FloatingPanel | None = None
        self.scrubbing = False

        self.states = {"telemetry": False, "chart": False, "stints": False, "drs": True}
        self.menu = InsightsMenu(pygame.Rect(*PANEL_RECTS["menu"]))
        self.telemetry = TelemetryPanel(pygame.Rect(*PANEL_RECTS["telemetry"]))
        self.chart = LapChartPanel(pygame.Rect(*PANEL_RECTS["chart"]))
        self.stints = StintsPanel(pygame.Rect(*PANEL_RECTS["stints"]))
        self.chart.ds = ds
        self.stints.ds = ds
        self.panels = [self.telemetry, self.chart, self.stints]  # draw order, menu on top
        self.menu_regions: list[tuple[pygame.Rect, str]] = []

    # ------------------------------------------------------------------ events
    def handle_events(self) -> None:
        for ev in pygame.event.get():
            try:
                self._dispatch(ev)
            except Exception:
                # a bad input event must never kill the window; show it instead
                import traceback

                traceback.print_exc()
                print(f"[f1replay] ignored event {pygame.event.event_name(ev.type)}",
                      file=sys.stderr)

    def _dispatch(self, ev: pygame.event.Event) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.KEYDOWN:
            self.on_key(ev)
        elif ev.type == pygame.MOUSEMOTION:
            self.mouse = ev.pos
            if self.dragging is not None:
                self.dragging.drag_to(ev.pos)
            elif self.scrubbing:
                self._scrub_to(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.mouse = ev.pos
            # mouse events carry no `mod`; read live modifier state instead
            shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
            self.on_click(ev.pos, shift)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.dragging is not None:
                self.dragging.end_drag()
            self.dragging = None
            self.scrubbing = False
        elif ev.type == pygame.WINDOWFOCUSLOST:
            pass

    def on_key(self, ev: pygame.event.Event) -> None:
        pb = self.playback
        key = ev.key
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            pb.toggle_play()
        elif key == pygame.K_RIGHT:
            pb.seek(SEEK_STEP)
        elif key == pygame.K_LEFT:
            pb.seek(-SEEK_STEP)
        elif key == pygame.K_UP:
            pb.cycle_speed(1)
        elif key == pygame.K_DOWN:
            pb.cycle_speed(-1)
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            pb.set_speed(key - pygame.K_1)
        elif key == pygame.K_r:
            pb.restart()
        elif key == pygame.K_d:
            self.toggle("drs")
        elif key == pygame.K_b:
            self.hud.show_progress = not self.hud.show_progress
        elif key == pygame.K_l:
            self.hud.show_names = not self.hud.show_names
            self.view.show_names = self.hud.show_names
        elif key == pygame.K_t:
            self.toggle("telemetry")
        elif key == pygame.K_g:
            self.toggle("chart")
        elif key == pygame.K_y:
            self.toggle("stints")

    def toggle(self, attr: str) -> None:
        if attr == "drs":
            self.states["drs"] = not self.states["drs"]
            self.view.invalidate(self.states["drs"])
            return
        self.states[attr] = not self.states[attr]
        getattr(self, attr).visible = self.states[attr]

    def sync_states(self) -> None:
        """Mirror panel visibility into the menu checkbox states."""
        for key in ("telemetry", "chart", "stints"):
            self.states[key] = getattr(self, key).visible

    def on_click(self, pos: tuple[int, int], shift: bool) -> None:
        # topmost floating panel first (menu drawn last -> checked first)
        ordered = [self.menu] + list(reversed(self.panels))
        for p in ordered:
            if not p.visible:
                continue
            hit = p.begin_drag(pos)
            if hit == "close":
                self.sync_states()
                return
            if hit == "drag":
                self.dragging = p
                return

        # insights menu buttons
        if self.menu.visible:
            for rect, attr in self.menu_regions:
                if rect.collidepoint(pos):
                    self.toggle(attr)
                    return

        # transport buttons
        for rect, action in self.hud.transport_regions:
            if rect.collidepoint(pos):
                self.transport(action)
                return

        # progress scrubbing
        if self.hud.progress_region and self.hud.progress_region.collidepoint(pos):
            self.scrubbing = True
            self._scrub_to(pos)
            return

        # leaderboard rows
        for rect, number in self.hud.board_regions:
            if rect.collidepoint(pos):
                self.pick_number(number, shift)
                return

        # track map (empty viewport click clears selection)
        if self.hud.viewport.collidepoint(pos):
            num = self.view.pick(pos[0], pos[1])
            if num is not None:
                self.pick_number(num, shift)
            elif not shift:
                self.selected.clear()
            return

    def pick_number(self, number: str, shift: bool) -> None:
        if shift:
            if number in self.selected:
                self.selected.discard(number)
            else:
                self.selected.add(number)
        elif self.selected == {number}:
            self.selected.clear()
        else:
            self.selected = {number}

    def transport(self, action: str) -> None:
        pb = self.playback
        if action == "rewind":
            pb.seek(-15.0)
        elif action == "ff":
            pb.seek(15.0)
        elif action == "play":
            pb.toggle_play()
        elif action == "speed":
            pb.cycle_speed(1)
        elif action == "restart":
            pb.restart()

    def _scrub_to(self, pos: tuple[int, int]) -> None:
        tr = self.hud.progress_track
        if tr.w <= 0:
            return
        frac = (pos[0] - tr.x) / tr.w
        self.playback.seek_fraction(min(1.0, max(0.0, frac)))

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        held = 0.0
        if keys[pygame.K_RIGHT]:
            held += SCRUB_RATE * dt
        if keys[pygame.K_LEFT]:
            held -= SCRUB_RATE * dt
        if held:
            self.playback.seek(held)
        if not self.scrubbing:
            self.playback.update(dt)

    # -------------------------------------------------------------------- draw
    def draw(self, surface: pygame.Surface | None = None,
             mouse: tuple[int, int] | None = None) -> None:
        surface = self.screen if surface is None else surface
        mouse = self.mouse if mouse is None else mouse
        w, h = surface.get_size()
        surface.fill((16, 18, 22))
        self.hud.layout(w, h)
        frame = self.playback.frame()

        self.hud.draw_top(surface, frame)
        self.view.show_names = self.hud.show_names
        self.view.draw(surface, self.hud.viewport, frame, self.selected,
                       pygame.time.get_ticks() / 1000.0)
        self.hud.draw_leaderboard(surface, frame, self.selected, mouse)
        self.hud.draw_bottom(surface, self.playback, frame, mouse, self.scrubbing)
        self.hud.draw_legend(surface)

        # floating panels (menu on top of telemetry)
        if self.chart.visible:
            self.chart.draw(surface, self.selected)
        if self.stints.visible:
            self.stints.draw(surface, self.selected)
        if self.telemetry.visible:
            self.telemetry.draw(surface, frame, self.selected)
        if self.menu.visible:
            self.sync_states()
            self.menu_regions = self.menu.draw(surface, mouse, self.states)
        else:
            self.menu_regions = []

    # --------------------------------------------------------------------- run
    def run(self) -> None:
        while self.running:
            dt = min(0.1, self.clock.tick(60) / 1000.0)
            self.handle_events()
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    # --------------------------------------------------------------- smoke test
    def smoke(self, out_dir) -> str:
        """Render a few representative frames headlessly and save PNGs."""
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ds = self.ds
        leader = ds.order[0] if ds.order else next(iter(ds.drivers))
        self.selected = {leader}

        sc = ds.sc_windows[0].start + 30 if ds.sc_windows else ds.t0 + ds.duration * 0.5
        stops = [
            ("01_start", ds.t0 + 30),
            ("02_running", ds.t0 + ds.duration * 0.35),
            ("03_safety_car", sc),
            ("04_mid_race", ds.t0 + ds.duration * 0.65),
            ("05_finish", ds.t1 - 4),
        ]
        real_playing = self.playback.playing
        real_visible = {id(p): p.visible for p in [self.menu, self.telemetry, self.chart, self.stints]}
        saved = []
        for name, t in stops:
            self.playback.playing = False
            self.playback.t = float(min(max(t, ds.t0), ds.t1))
            self.telemetry.visible = True
            self.chart.visible = name != "01_start"
            self.stints.visible = name == "04_mid_race"
            self.draw(self.screen, mouse=(-999, -999))
            path = out / f"{name}.png"
            pygame.image.save(self.screen, str(path))
            saved.append(str(path))
        self.playback.playing = real_playing
        for p in [self.menu, self.telemetry, self.chart, self.stints]:
            p.visible = real_visible[id(p)]
        return "\n".join(saved)
