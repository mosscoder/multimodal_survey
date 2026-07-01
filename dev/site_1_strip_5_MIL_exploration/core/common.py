"""Shared config + frozen DINOv3 backbone + patch-feature extraction.

Multi-target one-vs-all demo: a frozen DINOv3 ViT-B/16 produces per-patch tokens,
and a small multiple-instance-learning (MIL) head with an OPTIONAL hidden layer decides,
per patch, which target species it looks like. The backbone is never trained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

# --- backbone -----------------------------------------------------------
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"   # swap-in point for any DINOv3/v2 ViT
PATCH = 16
EMBED_DIM = 768
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# --- paths --------------------------------------------------------------
# this package's own directory (…/dev/site_1_strip_5_MIL_exploration), derived from
# __file__ so renaming the directory can't break the path again
DEMO_DIR = str(Path(__file__).resolve().parents[1])
INAT_ROOT = "/Users/kdoherty/multimodal_survey/inat_dataset"   # plants under plants/, birds under birds/
INAT_IMAGES = f"{INAT_ROOT}/plants/out/images"
CAPTURES = ("/Users/kdoherty/multimodal_survey/missions/site_1/strip_5/runs/"
            "strip_5_2026-06-16_12-36-44/captures")
VIZ_DIR = f"{DEMO_DIR}/inference/site_1_strip_5/visualizations"
# The target species + their iNaturalist dataset dirs are the one source of truth in
# data.TARGETS (all 9). Overlay colors/legend live in render.COLORS.

TILE_SIZE = 224                       # training tile / robot tile side (a multiple of PATCH)


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_backbone(device: torch.device):
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------- #
# Crop views (CLAUDE.md crop spec) — two outputs per training image            #
#                                                                              #
#   ("square",)    View A -- center-crop to the SHORT side, resize to 224.     #
#                  The whole centered subject, downsized into one tile.        #
#                  e.g. 1024x768 -> crop 768x768 -> resize 224.                #
#   ("native", z)  View B -- resize the whole image by z, then take a NATIVE   #
#                  224 center window (no resize of the window itself). z=1.0 is #
#                  a native 224 center crop (tightest, smallest slice of the   #
#                  scene); z<1 downsizes first so the same window covers MORE   #
#                  of the scene and the subject looks smaller. e.g. z=0.5 on    #
#                  1024x768 -> resize 512x384 -> center 224.                    #
#   ("corner", c)  View S (sky) -- a NATIVE 224 window from corner c ("ul"/     #
#                  "ur"). For a centered subject the UPPER corners are sky;     #
#                  used ONLY for the auxiliary 'Sky' hard-negative class.       #
# --------------------------------------------------------------------------- #
def _to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """RGB PIL (already exactly TILE_SIZE square) -> normalized (1,3,224,224)."""
    x = torch.from_numpy(np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0)
    return (x.permute(2, 0, 1).unsqueeze(0) - MEAN) / STD


def view_square(pil: Image.Image, size: int = TILE_SIZE) -> torch.Tensor:
    """View A: center-crop to the short side, resize to `size`."""
    pil = pil.convert("RGB")
    s = min(pil.size)
    l, t = (pil.size[0] - s) // 2, (pil.size[1] - s) // 2
    pil = pil.crop((l, t, l + s, t + s)).resize((size, size), Image.BICUBIC)
    return _to_tensor(pil)


def view_native(pil: Image.Image, z: float = 1.0, size: int = TILE_SIZE) -> torch.Tensor:
    """View B: resize the whole image by z, then take a native `size` center window.
    z=1.0 -> no resize (a native center crop). If a side falls below `size` after the
    resize it is reflection-padded (centered) up to `size`, never upscaled."""
    pil = pil.convert("RGB")
    if z != 1.0:
        W0, H0 = pil.size
        pil = pil.resize((max(1, round(W0 * z)), max(1, round(H0 * z))), Image.BICUBIC)
    w, h = pil.size
    if w < size or h < size:                                 # too small to crop a full tile:
        tw, th = max(w, size), max(h, size)                  # reflect-pad (centered) up to size
        pl, pt = (tw - w) // 2, (th - h) // 2
        arr = np.pad(np.asarray(pil), ((pt, th - h - pt), (pl, tw - w - pl), (0, 0)), mode="reflect")
        pil = Image.fromarray(arr)
        w, h = tw, th
    l, t = (w - size) // 2, (h - size) // 2
    return _to_tensor(pil.crop((l, t, l + size, t + size)))


def crop_corner(pil: Image.Image, corner: str = "ul", size: int = TILE_SIZE) -> Image.Image:
    """Raw PIL `size`x`size` window from an image CORNER (no resize). corner in {"ul","ur"}
    (upper-left / upper-right; top = 0). A side below `size` is reflection-padded (centered) up to
    `size`, never upscaled. Single source of truth for the corner region (DINOv3 view + sky gate)."""
    pil = pil.convert("RGB")
    w, h = pil.size
    if w < size or h < size:                                 # too small for a full tile: reflect-pad
        tw, th = max(w, size), max(h, size)
        pl, pt = (tw - w) // 2, (th - h) // 2
        arr = np.pad(np.asarray(pil), ((pt, th - h - pt), (pl, tw - w - pl), (0, 0)), mode="reflect")
        pil = Image.fromarray(arr); w, h = tw, th
    left = 0 if corner == "ul" else w - size                 # upper-left or upper-right; top = 0
    return pil.crop((left, 0, left + size, size))


def view_corner(pil: Image.Image, corner: str = "ul", size: int = TILE_SIZE) -> torch.Tensor:
    """Sky/cloud view: normalized (1,3,size,size) tensor of the upper-corner crop (see crop_corner).
    For a centered subject (a bird in flight) the UPPER corners are open sky/cloud — sky-dominated
    tiles for the auxiliary 'Sky' hard-negative class."""
    return _to_tensor(crop_corner(pil, corner, size))


def make_view(pil: Image.Image, view, size: int = TILE_SIZE) -> torch.Tensor:
    """Build one crop view from a spec: ("square",), ("native", z) or ("corner","ul"/"ur")
    -> (1,3,size,size)."""
    if view[0] == "square":
        return view_square(pil, size)
    if view[0] == "native":
        return view_native(pil, float(view[1]), size)
    if view[0] == "corner":
        return view_corner(pil, str(view[1]), size)
    raise ValueError(f"unknown crop view {view!r}")


@torch.no_grad()
def patch_features(model, x: torch.Tensor, device: torch.device):
    """(B,3,H,W) -> (feats (B, gh*gw, EMBED_DIM), gh, gw).

    Returns the per-patch tokens only. The CLS + register tokens (which precede the
    patch tokens in the sequence) are dropped.
    """
    gh, gw = x.shape[2] // PATCH, x.shape[3] // PATCH
    out = model(pixel_values=x.to(device)).last_hidden_state   # (B, special+gh*gw, D)
    n_special = out.shape[1] - gh * gw                          # CLS + registers
    return out[:, n_special:, :], gh, gw


class MILHead(torch.nn.Module):
    """Per-patch multi-label scorer (one task per target species) with an OPTIONAL hidden
    layer, then log-sum-exp (smooth-max) bag pooling.

    Each input feature is a DINOv3 patch token (EMBED_DIM wide). With hidden>0 each patch
    passes LayerNorm -> Linear(hidden) -> GELU -> Linear(n_classes); with hidden=0 it is a
    linear probe (LayerNorm -> Linear(n_classes), no hidden layer). A bag scores high for
    class c if ANY patch looks like species c.

    forward(feats) where feats is (B, N, EMBED_DIM):
      returns (bag_logits (B, C), patch_logits (B, N, C))
    """
    def __init__(self, d: int = EMBED_DIM, n_classes: int = 2, hidden: int = 512, r: float = 8.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        if hidden and hidden > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, n_classes),
            )
        else:                                            # hidden=0 -> linear probe, no hidden layer
            self.mlp = torch.nn.Linear(d, n_classes)
        self.r = r

    def forward(self, feats: torch.Tensor):
        patch_logits = self.mlp(self.norm(feats))                    # (B, N, C)
        bag_logits = torch.logsumexp(self.r * patch_logits, dim=1) / self.r   # (B, C)
        return bag_logits, patch_logits
