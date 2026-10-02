"""Track map rendering: static circuit layer + dynamic cars + safety car."""
from __future__ import annotations

import math

import numpy as np
import pygame

from .models import ReplayDataset
from .playback import FrameState
from .safety_car import SCFrame
from . import theme


class TrackView:
    def __init__(self, ds: ReplayDataset):
        self.ds = ds
        self.viewport = pygame.Rect(0, 0, 1, 1)
        self.scale = 1.0
        self.offx = 0.0
        self.offy = 0.0
        self.static: pygame.Surface | None = None
        self.show_drs = True
        self.show_names = True
        self.car_screen: dict[str, tuple[int, int]] = {}

    # ---- transform ------------------------------------------------------- #
    def layout(self, viewport: pygame.Rect) -> None:
        if viewport.size == self.viewport.size and self.static is not None:
            return
        self.viewport = viewport.copy()
        pts = self.ds.track.pts
        margin = 34
        bw = max(1.0, pts[:, 0].max() - pts[:, 0].min())
        bh = max(1.0, pts[:, 1].max() - pts[:, 1].min())
        vw = max(10, viewport.w - 2 * margin)
        vh = max(10, viewport.h - 2 * margin)
        self.scale = min(vw / bw, vh / bh)
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
        self.offx = viewport.w / 2 - cx * self.scale
        self.offy = viewport.h / 2 + cy * self.scale
        self._rebuild_static()

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(self.offx + x * self.scale),
            int(self.offy - y * self.scale),
        )

    def _to_screen_arr(self, pts) -> list[tuple[int, int]]:
        out = []
        for x, y in pts:
            out.append((int(self.offx + x * self.scale), int(self.offy - y * self.scale)))
        return out

    # ---- static layer ---------------------------------------------------- #
    def _rebuild_static(self) -> None:
        ds = self.ds
        vp = self.viewport
        surf = pygame.Surface(vp.size)
        surf.fill(theme.BG)
        if vp.w < 5 or vp.h < 5:
            self.static = surf
            return

        track = ds.track
        screen_pts = self._to_screen_arr(track.pts)
        width = max(10, int(round(13 * self.scale)))
        pygame.draw.lines(surf, theme.TRACK_EDGE, False, screen_pts, width + 4)
        pygame.draw.lines(surf, theme.TRACK, False, screen_pts, width)

        # DRS zones: highlight the arcs where DRS may be used
        if self.show_drs and ds.drs_zones:
            for zs, ze in ds.drs_zones:
                zone_pts = self._zone_points(zs, ze)
                if len(zone_pts) >= 2:
                    pygame.draw.lines(surf, theme.DRS_ZONE, False, zone_pts, max(4, width - 6))

        # pit lane
        if len(ds.pit_lane_pts) >= 2:
            pit = self._to_screen_arr(ds.pit_lane_pts)
            pygame.draw.lines(surf, (78, 82, 94), False, pit, max(3, width // 3))

        # start/finish line
        p0 = track.point_at(0.0)
        p1 = track.point_at(6.0)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n, dx / n
        half = width / 2 + 3
        a = self.world_to_screen(p0[0] + nx * half, p0[1] + ny * half)
        b = self.world_to_screen(p0[0] - nx * half, p0[1] - ny * half)
        pygame.draw.line(surf, (240, 240, 245), a, b, 3)

        # corner numbers
        for cx, cy, num in ds.corners:
            x, y = self.world_to_screen(float(cx), float(cy))
            theme.text(surf, str(int(num)), (x, y), 11, (120, 126, 138), anchor="center")

        self.static = surf

    def _zone_points(self, zs: float, ze: float) -> list[tuple[int, int]]:
        track = self.ds.track
        pts = track.pts
        cum = track.cum
        L = track.length
        if ze > L:  # zone wrapping across the seam
            return self._zone_points(zs, L - 1e-6) + self._zone_points(0.0, ze - L)
        i0 = int(max(0, np.searchsorted(cum, zs) - 1))
        i1 = int(min(len(pts) - 1, np.searchsorted(cum, ze) + 1))
        return self._to_screen_arr(pts[i0 : i1 + 1])

    def invalidate(self, show_drs: bool) -> None:
        if show_drs != self.show_drs:
            self.show_drs = show_drs
            self._rebuild_static()

    # ---- dynamic layer ---------------------------------------------------- #
    def draw(
        self,
        surface: pygame.Surface,
        viewport: pygame.Rect,
        frame: FrameState,
        selected: set[str],
        wall_time: float,
    ) -> None:
        self.layout(viewport)
        prev_clip = surface.get_clip()
        surface.set_clip(viewport)
        if self.static is not None:
            surface.blit(self.static, viewport.topleft)
        else:
            surface.fill(theme.BG, viewport)

        # shift static-layer coordinates: static was drawn in viewport-local space
        origin = viewport.topleft
        self.car_screen = {}

        pulse = 0.5 + 0.5 * math.sin(wall_time * 6.0)

        # cars: leaders drawn last so they stay visible in a train
        ordered = sorted(frame.drivers, key=lambda f: -f.rank)
        label_jobs: list[tuple[str, tuple[int, int], str, tuple, bool]] = []
        for f in ordered:
            x, y = self.world_to_screen(f.x, f.y)
            x += origin[0]
            y += origin[1]
            self.car_screen[f.number] = (x, y)
            color = theme.hex_color(f.color)
            is_sel = f.number in selected
            if f.out:
                color = tuple(int(c * 0.35 + 60) for c in color)
            if is_sel:
                theme.blit_glow(surface, (x, y), theme.CAR_RADIUS, (255, 255, 255), 90 + int(60 * pulse))
            pygame.draw.circle(surface, (10, 10, 12), (x, y), theme.CAR_RADIUS + 2)
            pygame.draw.circle(surface, color, (x, y), theme.CAR_RADIUS)
            if f.out:
                pygame.draw.line(surface, (235, 60, 55), (x - 5, y - 5), (x + 5, y + 5), 2)
                pygame.draw.line(surface, (235, 60, 55), (x - 5, y + 5), (x + 5, y - 5), 2)
            else:
                pygame.draw.circle(surface, (12, 12, 14), (x, y), theme.CAR_RADIUS, 1)
            if is_sel:
                pygame.draw.circle(surface, (255, 255, 255), (x, y), theme.CAR_RADIUS + 3, 2)
            if self.show_names and not f.out:
                label_jobs.append(((x, y), f.code, color, is_sel))

        if self.show_names:
            for (x, y), code, color, is_sel in label_jobs:
                pos = (x + theme.CAR_RADIUS + 6, y - 8)
                theme.text(surface, code, pos, 12, (15, 15, 18), bold=True)
                theme.text(surface, code, (pos[0] - 1, pos[1] - 1), 12,
                           (255, 255, 255) if is_sel else (210, 214, 220), bold=True)

        if frame.sc is not None:
            self._draw_sc(surface, frame.sc, origin, pulse)

        surface.set_clip(prev_clip)

    def _draw_sc(
        self, surface: pygame.Surface, sc: SCFrame, origin: tuple[int, int], pulse: float
    ) -> None:
        x, y = self.world_to_screen(sc.x, sc.y)
        x += origin[0]
        y += origin[1]
        if sc.phase == "track":
            alpha, scale = 150, 1.0
        elif sc.phase == "deploy":
            alpha, scale = int(70 + 130 * pulse), 0.85 + 0.3 * pulse
        else:  # return
            alpha, scale = int(140 * (1.0 - 0.6 * pulse)), 1.0 - 0.2 * pulse
        theme.blit_glow(surface, (x, y), theme.SC_RADIUS + 6, theme.SC_COLOR, alpha, scale)
        pygame.draw.circle(surface, (40, 26, 4), (x, y), theme.SC_RADIUS + 3)
        pygame.draw.circle(surface, theme.SC_COLOR, (x, y), theme.SC_RADIUS)
        pygame.draw.circle(surface, theme.SC_DEEP, (x, y), theme.SC_RADIUS + 3, 2)
        label = sc.label
        img = theme.font(12, True).render(label, True, (20, 14, 2))
        bg = pygame.Surface((img.get_width() + 8, img.get_height() + 4), pygame.SRCALPHA)
        bg.fill((*theme.SC_COLOR, 220))
        bg.blit(img, (4, 2))
        surface.blit(bg, bg.get_rect(midbottom=(x, y - theme.SC_RADIUS - 6)))

    def pick(self, x: int, y: int, radius: int = 13) -> str | None:
        best, best_d = None, radius * radius
        for num, (cx, cy) in self.car_screen.items():
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d <= best_d:
                best, best_d = num, d
        return best
