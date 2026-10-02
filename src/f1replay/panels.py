"""Floating insight panels: menu, telemetry, lap chart, tyre stints."""
from __future__ import annotations

import math

import pygame

from .models import ReplayDataset
from .playback import FrameState
from . import theme


class FloatingPanel:
    """Base: draggable title bar + close button."""

    HEADER = 26

    def __init__(self, title: str, rect: pygame.Rect, visible: bool = False):
        self.title = title
        self.rect = rect
        self.visible = visible
        self._drag_off: tuple[int, int] | None = None

    def header_rect(self) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.y, self.rect.w, self.HEADER)

    def close_rect(self) -> pygame.Rect:
        h = self.header_rect()
        return pygame.Rect(h.right - 26, h.y + 3, 20, 20)

    def begin_drag(self, pos: tuple[int, int]) -> str | None:
        """'close' if the X was hit, 'drag' if the title bar was hit, else None."""
        if self.close_rect().collidepoint(pos):
            self.visible = False
            return "close"
        if self.header_rect().collidepoint(pos):
            self._drag_off = (pos[0] - self.rect.x, pos[1] - self.rect.y)
            return "drag"
        return None

    def drag_to(self, pos: tuple[int, int]) -> None:
        if self._drag_off is None:
            return
        self.rect.topleft = (
            max(0, pos[0] - self._drag_off[0]),
            max(56, pos[1] - self._drag_off[1]),
        )

    def end_drag(self) -> None:
        self._drag_off = None

    def draw_frame(self, surface: pygame.Surface) -> pygame.Rect:
        theme.panel(surface, self.rect)
        h = self.header_rect()
        pygame.draw.line(surface, theme.PANEL_EDGE, h.bottomleft, h.bottomright, 1)
        theme.text(surface, self.title, (h.x + 10, h.centery), 12, theme.ACCENT, bold=True, anchor="midleft")
        cr = self.close_rect()
        theme.text(surface, "x", cr.center, 14, theme.TEXT_DIM, anchor="center")
        return h


# --------------------------------------------------------------------------- #
class InsightsMenu(FloatingPanel):
    ITEMS = [
        ("Driver telemetry", "T", "telemetry"),
        ("Lap time chart", "G", "chart"),
        ("Tyre stints", "Y", "stints"),
        ("DRS zones", "D", "drs"),
    ]

    def __init__(self, rect: pygame.Rect):
        super().__init__("INSIGHTS", rect, visible=True)

    def draw(
        self, surface: pygame.Surface, mouse: tuple[int, int], states: dict[str, bool]
    ) -> list[tuple[pygame.Rect, str]]:
        self.rect.h = self.HEADER + len(self.ITEMS) * 34 + 12
        self.draw_frame(surface)
        regions: list[tuple[pygame.Rect, str]] = []
        y = self.rect.y + self.HEADER + 8
        for label, key, attr in self.ITEMS:
            r = pygame.Rect(self.rect.x + 8, y, self.rect.w - 16, 28)
            active = states.get(attr, False)
            theme.button(surface, r, f"[{'x' if active else ' '}] {label}  ({key})",
                         active=active, hovered=r.collidepoint(mouse), size=12)
            regions.append((r, attr))
            y += 34
        return regions


# --------------------------------------------------------------------------- #
class TelemetryPanel(FloatingPanel):
    def __init__(self, rect: pygame.Rect):
        super().__init__("DRIVER TELEMETRY", rect)

    def draw(self, surface: pygame.Surface, frame: FrameState, selected: set[str]) -> None:
        rows = [f for f in frame.drivers if f.number in selected][:6]
        extra = sum(1 for f in frame.drivers if f.number in selected) - len(rows)
        block = 46
        h = self.HEADER + 12 + (block * len(rows) + (18 if extra else 0)) + (34 if not rows else 10)
        self.rect.h = h
        self.draw_frame(surface)
        y = self.rect.y + self.HEADER + 8

        if not rows:
            theme.text(surface, "Select drivers in the", (self.rect.x + 12, y + 4), 12, theme.TEXT_DIM)
            theme.text(surface, "leaderboard (click / shift+click)", (self.rect.x + 12, y + 22), 12, theme.TEXT_DIM)
            return

        for f in rows:
            chip = pygame.Rect(self.rect.x + 10, y + 4, 5, 16)
            pygame.draw.rect(surface, theme.hex_color(f.color), chip, border_radius=2)
            theme.text(surface, f.code, (self.rect.x + 22, y + 4), 14, theme.TEXT, bold=True)
            theme.text(surface, f"L{max(1, f.lap)}", (self.rect.right - 12, y + 5), 13,
                       theme.TEXT_DIM, anchor="topright")
            if f.out:
                theme.text(surface, "OUT", (self.rect.right - 44, y + 5), 13, (235, 70, 60), bold=True)

            yy = y + 24
            spd = f"{int(round(f.speed))}"
            theme.text(surface, spd, (self.rect.x + 22, yy), 16, (255, 255, 255), bold=True)
            x = self.rect.x + 22 + theme.text_width(spd, 16, True) + 3
            theme.text(surface, "km/h", (x, yy + 3), 10, theme.TEXT_DIM)
            x += 30
            theme.text(surface, f"G{max(0, f.gear)}", (x, yy), 14, theme.TEXT, bold=True)
            x += 30
            drs_col = (60, 220, 120) if f.drs_text == "ON" else (
                (240, 200, 40) if f.drs_text == "AVAIL" else theme.TEXT_DIM)
            theme.text(surface, f"DRS {f.drs_text}", (x, yy + 1), 12, drs_col, bold=f.drs_text == "ON")
            x += 74
            comp = f.compound
            badge = pygame.Rect(x, yy + 1, 18, 18)
            ccol = theme.TYRE_COLORS.get(comp, (70, 74, 84))
            pygame.draw.rect(surface, ccol, badge, border_radius=4)
            theme.text(surface, theme.TYRE_LETTER.get(comp, "-"), badge.center, 11, (14, 14, 16),
                       bold=True, anchor="center")
            y += block

        if extra:
            theme.text(surface, f"+{extra} more selected", (self.rect.x + 12, y + 2), 11, theme.TEXT_DIM)


# --------------------------------------------------------------------------- #
class LapChartPanel(FloatingPanel):
    def __init__(self, rect: pygame.Rect):
        super().__init__("LAP TIMES", rect)
        self._sel_key: tuple = ()
        self._series: dict[str, tuple] = {}

    def _update(self, selected: set[str]) -> None:
        key = tuple(sorted(selected))
        if key == self._sel_key:
            return
        self._sel_key = key
        self._series = {}
        for num in selected:
            d = self.ds.drivers.get(num)
            if d is None or not len(d.lap_numbers):
                continue
            times = d.lap_times
            valid = ~((times != times) | (times <= 0))
            if valid.any():
                floor = 1.6 * float(times[valid].min())
                valid &= times < floor
            self._series[num] = (
                d.lap_numbers[valid].astype(float),
                times[valid].astype(float),
                d.code,
                d.color,
            )

    ds: ReplayDataset  # set by app after construction

    def draw(self, surface: pygame.Surface, selected: set[str]) -> None:
        self._update(selected)
        self.rect.h = 330
        self.draw_frame(surface)
        plot = pygame.Rect(self.rect.x + 50, self.rect.y + 34, self.rect.w - 66, self.rect.h - 62)
        pygame.draw.rect(surface, (20, 22, 27), plot)
        pygame.draw.rect(surface, theme.PANEL_EDGE, plot, 1)
        if not self._series:
            theme.text(surface, "Select drivers to compare lap times", plot.center, 13,
                       theme.TEXT_DIM, anchor="center")
            return

        all_t = np_concat([v[1] for v in self._series.values()])
        ymin, ymax = float(all_t.min()), float(all_t.max())
        if ymax - ymin < 2:
            ymax = ymin + 2
        pad = 0.06 * (ymax - ymin)
        ymin -= pad
        ymax += pad
        total = self.ds.total_laps

        def px(lap: float) -> int:
            return int(plot.x + (lap - 1) / max(1, total - 1) * plot.w)

        def py(t: float) -> int:
            return int(plot.bottom - (t - ymin) / (ymax - ymin) * plot.h)

        # grid + labels
        for i in range(5):
            t = ymin + (ymax - ymin) * i / 4
            y = py(t)
            pygame.draw.line(surface, (40, 44, 52), (plot.x, y), (plot.right, y), 1)
            theme.text(surface, theme.format_laptime(t), (plot.x - 6, y), 10, theme.TEXT_DIM,
                       anchor="midright")
        step = max(5, total // 5)
        for lap in range(1, total + 1, step):
            x = px(lap)
            pygame.draw.line(surface, (40, 44, 52), (x, plot.y), (x, plot.bottom), 1)
            theme.text(surface, str(lap), (x, plot.bottom + 8), 10, theme.TEXT_DIM, anchor="midtop")
        theme.text(surface, "LAP", (plot.centerx, plot.bottom + 22), 10, theme.TEXT_DIM, anchor="midtop")

        # series
        lx = plot.right - 6
        for num, (laps, times, code, color) in reversed(list(self._series.items())):
            col = theme.hex_color(color)
            pts = [(px(l), py(t)) for l, t in zip(laps, times)]
            if len(pts) >= 2:
                pygame.draw.lines(surface, col, False, pts, 2)
            for p in pts:
                pygame.draw.circle(surface, col, p, 2)
            img = theme.font(11, True).render(code, True, col)
            lx -= img.get_width() + 10
            surface.blit(img, (lx, self.rect.y + 32))


class StintsPanel(FloatingPanel):
    def __init__(self, rect: pygame.Rect):
        super().__init__("TYRE STINTS", rect)

    ds: ReplayDataset

    def draw(self, surface: pygame.Surface, selected: set[str]) -> None:
        nums = [n for n in self.ds.order if n in selected][:6]
        self.rect.h = self.HEADER + 30 + 34 * max(1, len(nums))
        self.draw_frame(surface)
        plot = pygame.Rect(self.rect.x + 52, self.rect.y + self.HEADER + 8,
                           self.rect.w - 66, 24)
        total = self.ds.total_laps

        if not nums:
            theme.text(surface, "Select drivers to see stint history",
                       (self.rect.x + 12, self.rect.y + self.HEADER + 12), 12, theme.TEXT_DIM)
            return

        def px(lap: float) -> int:
            return int(plot.x + (lap - 1) / max(1, total) * plot.w)

        y = plot.y
        for num in nums:
            d = self.ds.drivers[num]
            theme.text(surface, d.code, (self.rect.x + 10, y + 12), 13, theme.TEXT, bold=True,
                       anchor="midleft")
            for stint in d.stints:
                a = px(stint.lap_from)
                b = px(stint.lap_to + 1)
                rect = pygame.Rect(a, y, max(4, b - a - 2), 24)
                col = theme.TYRE_COLORS.get(stint.compound, (70, 74, 84))
                pygame.draw.rect(surface, col, rect, border_radius=4)
                if rect.w >= 16:
                    theme.text(surface, theme.TYRE_LETTER.get(stint.compound, "?"), rect.center,
                               12, (14, 14, 16), bold=True, anchor="center")
            y += 34

        axis_y = y + 2
        pygame.draw.line(surface, theme.PANEL_EDGE, (plot.x, axis_y), (plot.right, axis_y), 1)
        step = max(5, total // 5)
        for lap in range(1, total + 1, step):
            theme.text(surface, str(lap), (px(lap), axis_y + 4), 10, theme.TEXT_DIM, anchor="midtop")
        theme.text(surface, "LAP", (plot.right, axis_y + 4), 10, theme.TEXT_DIM, anchor="midtop")


def np_concat(arrays):
    import numpy as np

    return np.concatenate([a for a in arrays if len(a)])
