"""Top bar, leaderboard, transport controls, progress bar and legend."""
from __future__ import annotations

import pygame

from .models import ReplayDataset
from .playback import FrameState, Playback
from . import theme


class Hud:
    def __init__(self, ds: ReplayDataset):
        self.ds = ds
        self.show_progress = True
        self.show_names = True
        self.show_drs = True
        self.top = pygame.Rect(0, 0, 0, 56)
        self.board = pygame.Rect(0, 0, 300, 0)
        self.viewport = pygame.Rect(0, 0, 0, 0)
        self.bottom = pygame.Rect(0, 0, 0, 0)
        self.progress_track = pygame.Rect(0, 0, 0, 10)
        self.board_regions: list[tuple[pygame.Rect, str]] = []
        self.transport_regions: list[tuple[pygame.Rect, str]] = []
        self.progress_region: pygame.Rect | None = None
        self._row_h = 30

    # ---- layout ----------------------------------------------------------- #
    def layout(self, w: int, h: int) -> None:
        strip = 96 if self.show_progress else 56
        self.top = pygame.Rect(0, 0, w, 56)
        self.board = pygame.Rect(w - 300, 56, 300, h - 56 - strip)
        self.viewport = pygame.Rect(0, 56, w - 300, h - 56 - strip)
        self.bottom = pygame.Rect(0, h - strip, w, strip)
        self._row_h = max(24, (self.board.h - 26) // max(1, len(self.ds.drivers)))
        if self.show_progress:
            x0 = 300
            self.progress_track = pygame.Rect(x0, h - 46, max(60, w - 316 - x0), 10)
            self.progress_region = self.progress_track.inflate(24, 24)
        else:
            self.progress_region = None
        self._transport_rects(w, h)

    def _transport_rects(self, w: int, h: int) -> None:
        y = h - 88 if self.show_progress else h - 44
        x = 16
        labels = [("rewind", "<<", 46), ("play", "PAUSE", 78), ("ff", ">>", 46),
                  ("speed", "1x", 58), ("restart", "R", 40)]
        self.transport_regions = []
        for action, label, width in labels:
            self.transport_regions.append((pygame.Rect(x, y, width, 30), action))
            x += width + 8

    def strip_height(self) -> int:
        return self.bottom.h

    # ---- top bar ---------------------------------------------------------- #
    def draw_top(self, surface: pygame.Surface, frame: FrameState) -> None:
        r = self.top
        pygame.draw.rect(surface, theme.PANEL, r)
        pygame.draw.line(surface, theme.PANEL_EDGE, r.bottomleft, r.bottomright, 1)
        cy = r.centery

        theme.text(surface, self.ds.session_title, (16, cy), 15, theme.TEXT_DIM, anchor="midleft")
        lap_txt = f"LAP {max(1, frame.lap_current)}/{frame.lap_total}"
        theme.text(surface, lap_txt, (r.w * 0.34, cy), 21, theme.TEXT, bold=True, anchor="midleft")

        clock = f"{theme.format_clock(frame.elapsed)} / {theme.format_clock(self.ds.duration - 20)}"
        theme.text(surface, clock, (r.w * 0.58, cy), 19, (200, 205, 212), anchor="midleft")

        # track status pill (right)
        status = frame.status_text
        color = theme.STATUS_COLORS.get(status, (90, 95, 105))
        pill = pygame.Rect(0, 0, theme.text_width(status, 14, True) + 34, 28)
        pill.midright = (r.w - 16, cy)
        pygame.draw.rect(surface, color, pill, border_radius=14)
        theme.text(surface, status, pill.center, 14, (18, 16, 10), bold=True, anchor="center")
        if frame.in_sc_window:
            pygame.draw.rect(surface, (255, 210, 120), pill.inflate(8, 8), 2, border_radius=17)

    # ---- leaderboard ------------------------------------------------------ #
    def draw_leaderboard(
        self,
        surface: pygame.Surface,
        frame: FrameState,
        selected: set[str],
        mouse: tuple[int, int],
    ) -> None:
        b = self.board
        pygame.draw.rect(surface, theme.PANEL, b)
        pygame.draw.line(surface, theme.PANEL_EDGE, b.topleft, b.bottomleft, 1)

        hdr_y = b.y + 4
        theme.text(surface, "POS", (b.x + 8, hdr_y + 8), 11, theme.TEXT_DIM)
        theme.text(surface, "DRIVER", (b.x + 46, hdr_y + 8), 11, theme.TEXT_DIM)
        theme.text(surface, "TYRE", (b.x + b.w - 34, hdr_y + 8), 11, theme.TEXT_DIM, anchor="midtop")
        y = b.y + 26
        self.board_regions = []
        row_h = self._row_h

        for f in frame.drivers:
            row = pygame.Rect(b.x, y, b.w, row_h - 2)
            hover = row.collidepoint(mouse) and not f.out
            is_sel = f.number in selected
            if is_sel:
                pygame.draw.rect(surface, theme.PANEL_HL, row)
                pygame.draw.rect(surface, f.color, (b.x, y, 3, row_h - 2))
            elif hover:
                pygame.draw.rect(surface, (33, 36, 43), row)

            cy = row.centery
            theme.text(surface, str(f.rank), (b.x + 10, cy), 15,
                       (255, 255, 255) if f.rank <= 3 else theme.TEXT, bold=True, anchor="midleft")
            chip = pygame.Rect(b.x + 38, cy - 9, 5, 18)
            pygame.draw.rect(surface, theme.hex_color(f.color), chip, border_radius=2)
            theme.text(surface, f.code, (b.x + 50, cy), 15,
                       (255, 255, 255) if is_sel else theme.TEXT, bold=True, anchor="midleft")

            if f.out:
                theme.text(surface, "OUT", (b.x + b.w - 64, cy), 13, (235, 70, 60), bold=True, anchor="midright")
            else:
                parts = f.name.split() if f.name else []
                name = (parts[-1] if parts else f.code)[:12]
                theme.text(surface, name, (b.x + 92, cy), 12, theme.TEXT_DIM, anchor="midleft")

            comp = f.compound
            badge = pygame.Rect(b.x + b.w - 34, cy - 11, 22, 22)
            ccol = theme.TYRE_COLORS.get(comp, (70, 74, 84))
            pygame.draw.rect(surface, ccol, badge, border_radius=5)
            pygame.draw.rect(surface, (14, 14, 16), badge, 1, border_radius=5)
            letter = theme.TYRE_LETTER.get(comp, "-")
            theme.text(surface, letter, badge.center, 13, (14, 14, 16), bold=True, anchor="center")

            self.board_regions.append((row, f.number))
            y += row_h

    # ---- transport + progress --------------------------------------------- #
    def draw_bottom(
        self,
        surface: pygame.Surface,
        playback: Playback,
        frame: FrameState,
        mouse: tuple[int, int],
        scrubbing: bool,
    ) -> None:
        r = self.bottom
        pygame.draw.rect(surface, theme.PANEL, r)
        pygame.draw.line(surface, theme.PANEL_EDGE, r.topleft, r.topright, 1)

        for rect, action in self.transport_regions:
            hovered = rect.collidepoint(mouse)
            if action == "play":
                label = "PAUSE" if playback.playing else "PLAY"
                theme.button(surface, rect, label, active=playback.playing, hovered=hovered, size=13)
            elif action == "speed":
                sp = playback.speed
                label = f"{sp:g}x"
                theme.button(surface, rect, label, active=sp != 1.0, hovered=hovered, size=13)
            elif action == "rewind":
                theme.button(surface, rect, "<<", hovered=hovered, size=15)
            elif action == "ff":
                theme.button(surface, rect, ">>", hovered=hovered, size=15)
            else:
                theme.button(surface, rect, "R", hovered=hovered, size=14)

        if not self.show_progress:
            return

        tr = self.progress_track
        # SC windows as amber spans
        for w in self.ds.sc_windows:
            f0 = (w.start - self.ds.t0) / max(1e-6, self.ds.duration)
            f1 = (w.end - self.ds.t0) / max(1e-6, self.ds.duration)
            a = tr.x + max(0, f0) * tr.w
            b = tr.x + min(1, f1) * tr.w
            if b > a:
                pygame.draw.rect(surface, (120, 84, 20), (int(a), tr.y, max(2, int(b - a)), tr.h), border_radius=4)

        pygame.draw.rect(surface, (48, 52, 60), tr, border_radius=5)
        frac = (frame.t - self.ds.t0) / max(1e-6, self.ds.duration)
        fill_w = int(tr.w * max(0.0, min(1.0, frac)))
        if fill_w > 2:
            pygame.draw.rect(surface, theme.ACCENT, (tr.x, tr.y, fill_w, tr.h), border_radius=5)
        hx = tr.x + fill_w
        pygame.draw.circle(surface, (255, 255, 255), (hx, tr.centery), 8)
        pygame.draw.circle(surface, theme.ACCENT, (hx, tr.centery), 8, 2)

        hovered = self.progress_region and self.progress_region.collidepoint(mouse)
        hint = "REWIND / FORWARD" if (hovered or scrubbing) else ""
        if hint:
            theme.text(surface, hint, (hx, tr.y - 10), 11, theme.TEXT_DIM, anchor="center")

        elapsed = f"{theme.format_clock(frame.t - self.ds.t0)}"
        total = theme.format_clock(self.ds.duration)
        theme.text(surface, f"{elapsed} / {total}", (tr.right, tr.y + tr.h + 14), 12,
                   theme.TEXT_DIM, anchor="midright")

    # ---- legend ------------------------------------------------------------- #
    LEGEND_ROWS = [
        ("SPACE", "Pause / resume"),
        ("LEFT / RIGHT", "Rewind / fast forward"),
        ("UP / DOWN", "Cycle playback speed"),
        ("1 - 4", "Set speed 0.5x / 1x / 2x / 4x"),
        ("R", "Restart replay"),
        ("D", "Show / hide DRS zones"),
        ("B", "Show / hide progress bar"),
        ("L", "Show / hide driver names"),
        ("T / G / Y", "Insights: telemetry / laps / tyres"),
        ("CLICK", "Select driver (SHIFT = multi)"),
        ("ESC", "Quit"),
    ]

    def legend_rect(self) -> pygame.Rect:
        h = 30 + len(self.LEGEND_ROWS) * 17 + 8
        x = 14
        bottom = self.viewport.bottom - 14
        return pygame.Rect(x, bottom - h, 268, h)

    def draw_legend(self, surface: pygame.Surface) -> None:
        rect = self.legend_rect()
        theme.panel(surface, rect, 215)
        theme.text(surface, "CONTROLS", (rect.x + 12, rect.y + 9), 12, theme.ACCENT, bold=True)
        y = rect.y + 30
        for key, desc in self.LEGEND_ROWS:
            theme.text(surface, key, (rect.x + 12, y), 11, (255, 255, 255), bold=True)
            theme.text(surface, desc, (rect.x + 118, y), 11, theme.TEXT_DIM)
            y += 17
