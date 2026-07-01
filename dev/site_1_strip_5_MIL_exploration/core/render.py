"""Overlay rendering: draw the per-species ALPHA-SHAPE (concave region) of the patches that fire
(score >= the species' frozen threshold), with a centroid marker and a light-up legend. cv2/NumPy/PIL.

The firing DECISION is still the frozen one-threshold-per-species rule (patch fires iff score >=
threshold; no Otsu/hysteresis/recalibration). On top of that, this overlay is a PRESENTATION layer:
the fired patches are morphologically closed (small gaps bridged) and traced as a concave contour
(the "alpha-shape") that hugs the region. EVERY fired region is drawn — down to a single patch (no deletions). This
does NOT change the any-patch metric (computed separately in infer_mil over all 9 species); it only
changes how the fired region is drawn. Each region's centroid is a candidate map-marker position.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# per-class (interior fill, outline) pastel colors, keyed by species name (RGB).
COLORS = {
    "Lupinus sericeus":       (np.array([222, 212, 245], np.float32), np.array([150, 120, 205], np.float32)),  # lavender
    "Gaillardia aristata":    (np.array([250, 214, 206], np.float32), np.array([232, 120, 105], np.float32)),  # coral
    "Balsamorhiza sagittata": (np.array([250, 236, 200], np.float32), np.array([228, 180,  70], np.float32)),  # gold
    "Thinopyrum intermedium": (np.array([216, 238, 212], np.float32), np.array([120, 190, 120], np.float32)),  # sage
    "Tragopogon dubius":      (np.array([252, 228, 200], np.float32), np.array([240, 160,  80], np.float32)),  # peach
    "Bromus tectorum":        (np.array([248, 216, 232], np.float32), np.array([224, 130, 180], np.float32)),  # rose
    "Achillea millefolium":   (np.array([208, 240, 238], np.float32), np.array([ 90, 195, 190], np.float32)),  # aqua
    "Poa bulbosa":            (np.array([212, 226, 250], np.float32), np.array([110, 160, 225], np.float32)),  # sky blue
}
# Species drawn in overlays + legend. Set to None to draw every class with a color.
DRAW_CLASSES = ["Lupinus sericeus", "Gaillardia aristata", "Balsamorhiza sagittata"]
FILL_ALPHA = 0.26                 # translucency of the region fill
CLOSE_PATCH = 1.6                 # alpha tightness: morphological-close kernel, in patch-widths
PAD_PATCH = 0.45                  # small outward margin over the fired cells, in patch-widths
MIN_PATCHES = 1                   # draw EVERY fired region (no deletion), down to a single patch;
                                  # the ~1-patch floor only guards against degenerate sub-patch contours
HALO_RGB = (15, 15, 15)


def _drawable(name):
    return name in COLORS and (DRAW_CLASSES is None or name in DRAW_CLASSES)


def alpha_regions(grid_hwc, thresholds, classes, size):
    """Per drawn species, the concave alpha-shape(s) of its fired patches at image `size` (W,H).
    -> {name: [{"contour": np(int) Nx2, "centroid": (x,y), "area": px}]}. Pure geometry, no drawing."""
    W, H = size
    GH, GW = grid_hwc.shape[:2]
    pp = W / GW                                                   # px per patch
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(CLOSE_PATCH * pp)) | 1),) * 2)
    pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, int(round(PAD_PATCH * pp))),) * 2)
    min_area = MIN_PATCHES * pp * pp
    out = {}
    for c, name in enumerate(classes):
        if not _drawable(name):
            continue
        fire = grid_hwc[..., c] >= thresholds[c]
        if not fire.any():
            continue
        px = cv2.resize((fire.astype(np.uint8) * 255), (W, H), interpolation=cv2.INTER_NEAREST)
        px = cv2.dilate(cv2.morphologyEx(px, cv2.MORPH_CLOSE, close), pad)
        cnts, _ = cv2.findContours(px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regs = []
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if a < min_area:
                continue
            M = cv2.moments(cnt); m00 = max(M["m00"], 1.0)
            regs.append({"contour": cnt.reshape(-1, 2), "centroid": (int(M["m10"] / m00), int(M["m01"] / m00)),
                         "area": int(a)})
        if regs:
            out[name] = regs
    return out


def overlay(orig, grid_hwc, thresholds, classes):
    """orig PIL frame, grid_hwc (H,W,ncls) per-patch scores, per-class thresholds.
    Draws each drawn species' alpha-shape region(s) + centroid; legend lights up species drawn here.
    Returns (overlaid PIL image, regions dict from alpha_regions)."""
    base = np.asarray(orig.convert("RGB")).astype(np.uint8)
    H, W = base.shape[:2]
    pp = W / grid_hwc.shape[1]
    regions = alpha_regions(grid_hwc, thresholds, classes, (W, H))

    fill = base.copy()
    for name, regs in regions.items():
        irgb = tuple(int(v) for v in COLORS[name][0])
        cv2.fillPoly(fill, [r["contour"] for r in regs], irgb)
    out = cv2.addWeighted(fill, FILL_ALPHA, base, 1 - FILL_ALPHA, 0)

    halo_w = max(2, int(round(0.275 * pp))); line_w = max(1, int(round(0.14 * pp))); dot = max(1, int(round(0.105 * pp)))
    for name, regs in regions.items():                                   # dark halo under every outline
        for r in regs:
            cv2.polylines(out, [r["contour"]], True, HALO_RGB, halo_w, cv2.LINE_AA)
    for name, regs in regions.items():                                   # colored outline + centroid
        orgb = tuple(int(v) for v in COLORS[name][1])
        for r in regs:
            cv2.polylines(out, [r["contour"]], True, orgb, line_w, cv2.LINE_AA)
            cv2.circle(out, r["centroid"], dot, orgb, -1, cv2.LINE_AA)
            cv2.circle(out, r["centroid"], dot, (255, 255, 255), max(1, line_w // 2), cv2.LINE_AA)

    img = Image.fromarray(out)
    _draw_legend(img, [n for n in classes if _drawable(n)], set(regions))
    return img, regions


# --------------------------------------------------------------------------- #
# Legend                                                                       #
# --------------------------------------------------------------------------- #
def _load_font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_legend(img, names, present=(), font_size=15):
    if not names:
        return
    dr = ImageDraw.Draw(img, "RGBA"); font = _load_font(font_size)
    sw, pad, gap = 15, 8, 7
    lh = max(sw, font_size) + gap
    tw = max(dr.textlength(n, font=font) for n in names)
    boxw = pad + sw + 8 + int(tw) + pad
    W = img.size[0]; x1, y1 = W - boxw - 10, 10
    dr.rectangle([x1, y1, x1 + boxw, y1 + pad + lh * len(names)], fill=(0, 0, 0, 170))
    for i, n in enumerate(names):
        on = n in present
        col = tuple(int(v) for v in COLORS[n][1])
        sx, sy = x1 + pad, y1 + pad + i * lh
        if on:
            dr.rectangle([x1 + 3, sy - 3, x1 + boxw - 3, sy + sw + 3], fill=(255, 255, 255, 32))
            dr.rectangle([sx, sy, sx + sw, sy + sw], fill=col + (255,), outline=(255, 255, 255, 240))
            dr.text((sx + sw + 8, sy), n, font=font, fill=(245, 245, 245, 255))
        else:
            dr.rectangle([sx, sy, sx + sw, sy + sw], fill=col + (55,), outline=(130, 130, 130, 130))
            dr.text((sx + sw + 8, sy), n, font=font, fill=(150, 150, 150, 165))
