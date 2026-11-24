"""Recolour MLflow's compiled UI assets into a Prussian blue and gold palette.

MLflow ships its UI as a pre-built React bundle with colours baked into both
the stylesheets and JavaScript design tokens, so there is no supported
theming hook. This module rewrites those colour literals:

* Blues in the hue band of MLflow's brand blue (198 to 235 degrees, clearly
  saturated) are moved to the Prussian blue hue (205 degrees). Dark shades,
  which are the ones used on light backgrounds, are darkened toward
  Prussian blue (#003153). Lighter shades keep their lightness so that the
  dark theme and light tints stay legible.
* Everything else (greys, reds, greens, chart teals and yellows) is left
  unchanged so status colours and chart series keep their meaning.

Gold accents are layered on top by ``prussian-gold.css``.
"""

from __future__ import annotations

import colorsys
import hashlib
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PRUSSIAN_HUE_DEG = 205.0
HUE_BAND_DEG = (198.0, 235.0)
MIN_SATURATION = 0.35

# Piecewise-linear lightness map: input lightness -> output lightness.
# Shades up to 0.46 (light-theme brand blues) are darkened; shades from 0.50
# upward (dark-theme blues and tints) keep their lightness.
_LIGHTNESS_X = (0.00, 0.30, 0.46, 0.50, 1.00)
_LIGHTNESS_Y = (0.00, 0.12, 0.19, 0.50, 1.00)

_HEX = re.compile(r"#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?(?![0-9a-fA-F])")
_RGB = re.compile(r"(rgba?\(\s*)(\d{1,3})(\s*,\s*)(\d{1,3})(\s*,\s*)(\d{1,3})")

THEME_VERSION = "2"
_ASSET_SUFFIXES = {".css", ".js"}


def _interpolate(value: float) -> float:
    for (x0, y0), (x1, y1) in zip(
        zip(_LIGHTNESS_X, _LIGHTNESS_Y), zip(_LIGHTNESS_X[1:], _LIGHTNESS_Y[1:])
    ):
        if value <= x1:
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return value


def remap_rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Map one colour; returns it unchanged when it is outside the blue band."""
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if not (HUE_BAND_DEG[0] <= hue <= HUE_BAND_DEG[1]) or s < MIN_SATURATION:
        return r, g, b
    new_l = _interpolate(l)
    new_s = min(1.0, s * 1.3) if l < 0.5 else s
    nr, ng, nb = colorsys.hls_to_rgb(PRUSSIAN_HUE_DEG / 360.0, new_l, new_s)
    return round(nr * 255), round(ng * 255), round(nb * 255)


def remap_hex(token: str) -> str:
    """Map a ``#rrggbb`` or ``#rrggbbaa`` literal, preserving case and alpha."""
    body = token.lstrip("#")
    r, g, b = (int(body[i : i + 2], 16) for i in (0, 2, 4))
    out = "#{:02x}{:02x}{:02x}".format(*remap_rgb(r, g, b)) + body[6:]
    return out.upper() if any(c.isupper() for c in body) else out


def remap_text(text: str) -> str:
    """Rewrite every hex and rgb()/rgba() colour literal in a text asset."""

    def _hex(match: re.Match[str]) -> str:
        return remap_hex(match.group(0))

    def _rgb(match: re.Match[str]) -> str:
        channels = [int(match.group(i)) for i in (2, 4, 6)]
        if any(c > 255 for c in channels):
            return match.group(0)
        r, g, b = remap_rgb(*channels)
        return f"{match.group(1)}{r}{match.group(3)}{g}{match.group(5)}{b}"

    return _RGB.sub(_rgb, _HEX.sub(_hex, text))


def _theme_fingerprint(mlflow_version: str, css_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"{THEME_VERSION}|{mlflow_version}".encode())
    digest.update(css_path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:12]


def default_cache_root() -> Path:
    """Where themed copies of the UI are stored."""
    override = os.environ.get("EVMLOPS_THEME_CACHE")
    return Path(override) if override else Path.home() / ".cache" / "evmlops" / "theme"


def _inject_into_index(html: str) -> str:
    link = '<link href="static-files/prussian-gold.css" rel="stylesheet">'
    html = html.replace('<meta name="theme-color" content="#000000"/>', '<meta name="theme-color" content="#003153"/>')
    return html.replace("</head>", f"{link}</head>", 1)


def build_themed_static(source_dir: Path, mlflow_version: str, cache_root: Path | None = None) -> Path:
    """Create (or reuse) a recoloured copy of MLflow's static UI directory.

    The result is keyed by MLflow version and theme contents, so upgrading
    either triggers a rebuild. The build happens in a temporary directory and
    is moved into place atomically, which keeps concurrent server workers safe.
    """
    css_path = Path(__file__).with_name("prussian-gold.css")
    cache_root = cache_root or default_cache_root()
    target = cache_root / f"mlflow-{mlflow_version}-{_theme_fingerprint(mlflow_version, css_path)}"
    if (target / "index.html").is_file():
        return target

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="build-", dir=cache_root))
    try:
        LOGGER.info("Building Prussian blue and gold UI assets (one-off, %s)", target.name)
        for path in source_dir.rglob("*"):
            relative = path.relative_to(source_dir)
            destination = staging / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in _ASSET_SUFFIXES:
                destination.write_text(remap_text(path.read_text(encoding="utf-8")), encoding="utf-8")
            else:
                shutil.copy2(path, destination)

        index = staging / "index.html"
        index.write_text(_inject_into_index(index.read_text(encoding="utf-8")), encoding="utf-8")
        shutil.copy2(css_path, staging / "prussian-gold.css")
        try:
            os.rename(staging, target)
        except OSError:
            # Another worker finished first; its build is equivalent.
            shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
