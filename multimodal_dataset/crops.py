"""Nadir drone-image sampling in the robot's bearing frame.

The release-defining drone-image geometry lives here and only here, fixed by
two constants. ``S`` (4.0 m) is the side of the ground footprint, the square
with the GNSS antenna at its bottom-edge midpoint rotated so the robot's
bearing points up, which ``footprint_wkt``, ``pair_overlap_fraction``,
``tier2_overlap_fraction`` and the split rule all use. ``P`` (0.0130 m) is
the output pixel pitch, the sensor-physics ground sample distance of the four
flights. The drone image is the ``N`` x ``N`` grid of ``N = round(S / P)`` =
308 pixels at exactly ``P`` per pixel, so the grid spans ``EXTENT = N * P`` =
4.004 m ahead of the antenna and 2.002 m to either side, 4 mm past the
footprint's far and side edges (below the 1.4 cm RTK accuracy). Every grid
point is interpolated once, bilinearly, from the orthomosaic at its native
pixel size (1.28 to 1.30 cm across the four flights), with no warped VRT and
no intermediate resampled file. ``coverage`` is the fraction of the grid whose
nearest native pixel lies inside the orthomosaic's alpha mask; the build
requires 1.0 on every row.

Coordinates: WGS84 lat/lon in, NAD83(2011) / Montana (EPSG:6514, meters) for
all geometry, the same frame the orthomosaics are published in.
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer

S = 4.0                 # footprint side, meters (split rule, overlap columns, footprint_wkt)
P = 0.0130              # output pixel pitch, meters per pixel
N = round(S / P)        # drone image side, pixels (308)
EXTENT = N * P          # pixel grid extent, meters (4.004)

# capture event -> that site's n-th flight (the release's ordinal pairing rule)
FLIGHTS = {
    ("site_1", 1): "260616-site-1-am",
    ("site_1", 2): "260618-site-1-pm",
    ("site_2", 1): "260617-site-2-am",
    ("site_2", 2): "260617-site-2-pm",
}
ORTHO_URL = ("https://storage.googleapis.com/mpg-aerial-survey/surveys/"
             "multimodal_survey/aerial/processing/drone_deploy/{flight}-RGB.tif")
CACHE = Path(__file__).resolve().parent / "cache"

_TO_M = Transformer.from_crs("EPSG:4326", "EPSG:6514", always_xy=True)
_TO_WGS = Transformer.from_crs("EPSG:6514", "EPSG:4326", always_xy=True)


def antenna_en(latitude: float, longitude: float) -> tuple[float, float]:
    """WGS84 -> (easting, northing) meters in EPSG:6514."""
    return _TO_M.transform(longitude, latitude)


def _axes(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (forward, left) of the bearing frame, in EN meters."""
    th = math.radians(bearing_deg)
    return (np.array([math.sin(th), math.cos(th)]),
            np.array([-math.cos(th), math.sin(th)]))


def footprint_corners_en(e: float, n: float, bearing_deg: float) -> list[tuple[float, float]]:
    """Footprint corners in EN meters: antenna-left, antenna-right, front-right,
    front-left, the documented ``footprint_wkt`` ring order."""
    fwd, left = _axes(bearing_deg)
    p = np.array([e, n])
    robot_frame = [(0.0, S / 2), (0.0, -S / 2), (S, -S / 2), (S, S / 2)]
    return [tuple(p + a * fwd + b * left) for a, b in robot_frame]


def footprint_wkt(latitude: float, longitude: float, bearing_deg: float) -> str:
    """The footprint as a WKT polygon in WGS84 lon/lat, ring closed."""
    e, n = antenna_en(latitude, longitude)
    corners = footprint_corners_en(e, n, bearing_deg)
    ring = [_TO_WGS.transform(x, y) for x, y in corners]
    ring.append(ring[0])
    pts = ", ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in ring)
    return f"POLYGON (({pts}))"


def fetch_ortho(flight: str) -> Path:
    """Download a flight's RGB orthomosaic into the local cache once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{flight}-RGB.tif"
    if not dst.exists():
        url = ORTHO_URL.format(flight=flight)
        print(f"[crops] fetching {url}")
        tmp = dst.with_suffix(".tif.part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


class Ortho:
    """One flight's orthomosaic, opened for windowed drone-image sampling."""

    def __init__(self, flight: str):
        import rasterio
        self.flight = flight
        self.ds = rasterio.open(fetch_ortho(flight))
        t = self.ds.transform
        if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
            raise ValueError(f"{flight}: rotated ortho transform is unsupported")
        if abs(abs(t.e) - t.a) > 1e-9:
            raise ValueError(f"{flight}: non-square pixels ({t.a}, {t.e}) are unsupported")
        self.x0, self.y0 = t.c, t.f
        self.px = t.a                      # native pixel size, meters (y pitch = -px)
        print(f"[crops] {flight}: native pixel size {self.px:.7f} m "
              f"({self.px * 100:.4f} cm), ratio to output pitch {P} m = {self.px / P:.5f}")

    def sample(self, e: float, n: float, bearing_deg: float) -> tuple[np.ndarray, float]:
        """(N x N x 3 uint8 drone image, coverage) at an antenna EN position."""
        from rasterio.windows import Window
        fwd, left = _axes(bearing_deg)
        # grid of pixel-center ground positions at the fixed pitch P: row 0 is
        # EXTENT - P/2 meters ahead, column 0 is EXTENT/2 - P/2 meters left
        # (matches the published pixel_to_ground)
        a = EXTENT - (np.arange(N) + 0.5) * P
        b = EXTENT / 2 - (np.arange(N) + 0.5) * P
        A, B = np.meshgrid(a, b, indexing="ij")
        E = e + A * fwd[0] + B * left[0]
        Nn = n + A * fwd[1] + B * left[1]

        # fractional native pixel coordinates of every grid point (pixel
        # centers at half-integers), sampled from the raster's own transform
        col = (E - self.x0) / self.px - 0.5
        row = (self.y0 - Nn) / self.px - 0.5
        c0, c1 = math.floor(col.min()) - 2, math.ceil(col.max()) + 3
        r0, r1 = math.floor(row.min()) - 2, math.ceil(row.max()) + 3
        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = self.ds.read(window=win, boundless=True, fill_value=0)
        rgb, alpha = data[:3].astype(np.float32), data[3] if data.shape[0] > 3 else None

        lc, lr = col - c0, row - r0                      # window-local coords
        ci, ri = np.floor(lc).astype(int), np.floor(lr).astype(int)
        dc, dr = lc - ci, lr - ri
        ci = np.clip(ci, 0, rgb.shape[2] - 2)
        ri = np.clip(ri, 0, rgb.shape[1] - 2)
        w00 = (1 - dc) * (1 - dr)
        w10 = dc * (1 - dr)
        w01 = (1 - dc) * dr
        w11 = dc * dr
        out = (rgb[:, ri, ci] * w00 + rgb[:, ri, ci + 1] * w10 +
               rgb[:, ri + 1, ci] * w01 + rgb[:, ri + 1, ci + 1] * w11)
        crop = np.clip(np.rint(out), 0, 255).astype(np.uint8).transpose(1, 2, 0)

        if alpha is None:
            coverage = 1.0
        else:
            ni = np.clip(np.rint(lc).astype(int), 0, alpha.shape[1] - 1)
            nj = np.clip(np.rint(lr).astype(int), 0, alpha.shape[0] - 1)
            coverage = float((alpha[nj, ni] > 127).mean())
        return crop, coverage

    def close(self):
        self.ds.close()
