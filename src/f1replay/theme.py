"""Colors, fonts and small drawing helpers."""
from __future__ import annotations

from functools import lru_cache

import pygame

# ---------------------------------------------------------------- palette --- #
BG = (16, 18, 22)
PANEL = (27, 30, 36)
PANEL_HL = (38, 42, 50)
PANEL_EDGE = (55, 60, 70)
TEXT = (235, 238, 242)
TEXT_DIM = (150, 156, 166)
ACCENT = (0, 150, 255)

TRACK = (62, 66, 76)
TRACK_EDGE = (44, 47, 55)
RACING_LINE = (92, 98, 110)
DRS_ZONE = (24, 170, 96)
SC_COLOR = (255, 168, 40)
SC_DEEP = (255, 130, 20)

STATUS_COLORS = {
    "GREEN": (40, 200, 110),
    "YELLOW": (240, 195, 20),
    "SAFETY CAR": (255, 168, 40),
    "VSC": (250, 140, 30),
    "VSC ENDING": (250, 140, 30),
    "RED FLAG": (230, 60, 55),
}

TYRE_COLORS = {
    "SOFT": (230, 70, 55),
    "MEDIUM": (240, 195, 20),
    "HARD": (235, 238, 242),
    "INTERMEDIATE": (45, 190, 100),
    "WET": (60, 140, 220),
}
TYRE_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}

CAR_RADIUS = 6
SC_RADIUS = 8

_font_family = ["dejavusans", "dejavu", "sans-serif"]


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> pygame.font.Font:
    f = pygame.font.SysFont(_font_family, size, bold=bold)
    if f is None:
        f = pygame.font.Font(None, size + 2)
    return f


def hex_color(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) != 6:
        return (200, 200, 200)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (200, 200, 200)


def text(
    surf: pygame.Surface,
    msg: str,
    pos: tuple[int, int],
    size: int = 14,
    color: tuple = TEXT,
    bold: bool = False,
    anchor: str = "topleft",
) -> pygame.Rect:
    img = font(size, bold).render(msg, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def text_width(msg: str, size: int, bold: bool = False) -> int:
    return font(size, bold).size(msg)[0]


def panel(surface: pygame.Surface, rect: pygame.Rect, alpha: int = 235) -> None:
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    s.fill((*PANEL, alpha))
    pygame.draw.rect(s, (*PANEL_EDGE, min(255, alpha)), s.get_rect(), 1, border_radius=8)
    surface.blit(s, rect.topleft)


def button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    *,
    active: bool = False,
    hovered: bool = False,
    size: int = 14,
    accent: tuple = ACCENT,
) -> None:
    if active:
        bg = tuple(int(c * 0.45 + a * 0.55) for c, a in zip(PANEL_HL, accent))
        edge = accent
    elif hovered:
        bg = PANEL_HL
        edge = PANEL_EDGE
    else:
        bg = PANEL
        edge = PANEL_EDGE
    pygame.draw.rect(surface, bg, rect, border_radius=6)
    pygame.draw.rect(surface, edge, rect, 1, border_radius=6)
    text(surface, label, rect.center, size, TEXT if not active else (255, 255, 255), bold=False, anchor="center")


@lru_cache(maxsize=128)
def glow_template(radius: int, color: tuple[int, int, int], pad: int = 16) -> pygame.Surface:
    """Concentric alpha rings used for pulsing glow effects."""
    size = (radius + pad) * 2
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2
    layers = [(radius + pad, 26), (radius + pad * 2 // 3, 46), (radius + 3, 80)]
    for r, a in layers:
        pygame.draw.circle(s, (*color, a), (center, center), r)
    return s


def blit_glow(
    dest: pygame.Surface,
    pos: tuple[float, float],
    radius: int,
    color: tuple[int, int, int],
    alpha: int,
    scale: float = 1.0,
) -> None:
    if alpha <= 0:
        return
    tpl = glow_template(radius, color)
    if scale != 1.0:
        w = max(1, int(tpl.get_width() * scale))
        tpl = pygame.transform.smoothscale(tpl, (w, w))
    tpl = tpl.copy()
    tpl.set_alpha(min(255, alpha))
    dest.blit(tpl, tpl.get_rect(center=(int(pos[0]), int(pos[1]))))


def format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_laptime(sec: float) -> str:
    if sec != sec or sec <= 0:  # NaN
        return "--:--.---"
    m, rest = divmod(sec, 60)
    return f"{int(m)}:{rest:06.3f}"
