"""Nadir drone-crop sampling in the robot's bearing frame.

The release-defining crop geometry lives here and only here: a ``S`` x ``S``
meter window with the GNSS antenna at the bottom-edge midpoint, rotated so the
robot's bearing points up the image, pinned to ``N`` x ``N`` pixels regardless
of each flight's native GSD. Bilinear resampling for the RGB bands; the crop's
``drone_coverage`` is the fraction of its grid inside the flight footprint
(nearest-neighbor alpha at native resolution).

Coordinates: WGS84 lat/lon in, NAD83(2011) / Montana (EPSG:6514, meters) for
all geometry — the same frame the orthomosaics are published in.
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer

S = 4.0     # crop side, meters
N = 312     # crop side, pixels

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
    """Crop corners in EN meters: antenna-left, antenna-right, front-right,
    front-left — the documented ``footprint_wkt`` ring order."""
    fwd, left = _axes(bearing_deg)
    p = np.array([e, n])
    robot_frame = [(0.0, S / 2), (0.0, -S / 2), (S, -S / 2), (S, S / 2)]
    return [tuple(p + a * fwd + b * left) for a, b in robot_frame]


def footprint_wkt(latitude: float, longitude: float, bearing_deg: float) -> str:
    """The crop outline as a WKT polygon in WGS84 lon/lat, ring closed."""
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
    """One flight's orthomosaic, opened for windowed crop sampling."""

    def __init__(self, flight: str):
        import rasterio
        self.flight = flight
        self.ds = rasterio.open(fetch_ortho(flight))
        t = self.ds.transform
        if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
            raise ValueError(f"{flight}: rotated ortho transform is unsupported")
        self.x0, self.y0 = t.c, t.f
        self.px = t.a                      # pixel size, meters (y pitch = -px)

    def sample(self, e: float, n: float, bearing_deg: float) -> tuple[np.ndarray, float]:
        """(N x N x 3 uint8 crop, drone_coverage) at an antenna EN position."""
        from rasterio.windows import Window
        fwd, left = _axes(bearing_deg)
        # grid of pixel-center ground positions: row 0 is S meters ahead,
        # column 0 is S/2 meters left (matches the published pixel_to_ground)
        a = S - (np.arange(N) + 0.5) * (S / N)
        b = S / 2 - (np.arange(N) + 0.5) * (S / N)
        A, B = np.meshgrid(a, b, indexing="ij")
        E = e + A * fwd[0] + B * left[0]
        Nn = n + A * fwd[1] + B * left[1]

        col = (E - self.x0) / self.px
        row = (self.y0 - Nn) / self.px
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
