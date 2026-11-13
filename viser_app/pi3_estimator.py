"""π³ (Pi3X) front-end: estimate camera poses + intrinsics from unposed images.

Pi3X (yyfz233/Pi3X, ICLR 2026) is a feed-forward Permutation-Equivariant Visual
Geometry Learning model. Given a batch of images, it returns:
  - camera_poses   : (B, N, 4, 4)   camera-to-world in OpenCV convention
  - points         : (B, N, H, W, 3) global point cloud
  - local_points   : (B, N, H, W, 3) per-view points in CAMERA frame
  - conf           : (B, N, H, W, 1) confidence logits

Pi3X does NOT output camera intrinsics. We estimate them from ``local_points``:
for a pinhole camera with principal point at the image center,
    X / Z = (u - cx) / fx     →     fx = (u - cx) * Z / X
We solve the same for fy and take a confidence-weighted median across all valid
pixels of all views. Far from perfect, but good enough as an init for DVSM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from viser_app.scene_loaders import pack_estimator_scene_data


def _list_images(d: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"}
    return sorted([p for p in d.iterdir() if p.suffix in exts])


# ---------------------------------------------------------------------------
# Pi3X loading + inference
# ---------------------------------------------------------------------------

def load_pi3_model(flavor: str = "pi3x", device: str = "cuda"):
    """Load Pi3 or Pi3X via the GitHub package (huggingface checkpoint)."""
    if flavor == "pi3x":
        from pi3.models.pi3x import Pi3X
        model = Pi3X.from_pretrained("yyfz233/Pi3X")
    elif flavor == "pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3")
    else:
        raise ValueError(f"unknown flavor {flavor!r}; expected 'pi3' or 'pi3x'")
    return model.to(device).eval()


def _resize_for_pi3(img_pil: Image.Image, image_size: int = 518, patch: int = 14) -> Image.Image:
    """Aspect-preserving resize so longest edge ≤ image_size, both dims multiple of `patch`."""
    w, h = img_pil.size
    if max(w, h) > image_size:
        s = image_size / max(w, h)
        w = int(round(w * s))
        h = int(round(h * s))
    w = max(patch, (w // patch) * patch)
    h = max(patch, (h // patch) * patch)
    return img_pil.resize((w, h), Image.LANCZOS)


def run_pi3(model, image_paths: list[Path], device: str = "cuda",
            image_size: int = 518) -> dict:
    """Returns:
        raw_images  : list of native HxWx3 uint8 ndarrays
        c2w         : (N, 4, 4) float32 numpy   camera-to-world (OpenCV)
        intr        : (fx, fy, cx, cy) pixels, valid for image_size
        image_size  : (h, w) of the (uniform) image batch fed to Pi3X
    """
    raw_imgs: list[np.ndarray] = []
    resized_imgs: list[np.ndarray] = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        raw_imgs.append(np.asarray(im))
        resized_imgs.append(np.asarray(_resize_for_pi3(im, image_size=image_size)))

    # Batched Pi3 input must be uniform-size. Use the smallest dims (snapped to 14).
    h_b = min(im.shape[0] for im in resized_imgs)
    w_b = min(im.shape[1] for im in resized_imgs)
    h_b = (h_b // 14) * 14
    w_b = (w_b // 14) * 14
    batch = np.stack([
        np.asarray(Image.fromarray(im).resize((w_b, h_b), Image.LANCZOS))
        for im in resized_imgs
    ])
    print(f"[pi3] feeding {len(batch)} images at {h_b}x{w_b}")

    imgs_t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    imgs_t = imgs_t.to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs_t[None])  # (1, N, ...)

    c2w = out["camera_poses"][0].float().cpu().numpy()         # (N, 4, 4)
    local_pts = out["local_points"][0].float().cpu().numpy()    # (N, H, W, 3)
    conf = out["conf"][0].float().cpu().numpy() if "conf" in out else None

    fx, fy, cx, cy = _estimate_intrinsics_from_local_points(local_pts, conf)
    print(f"[pi3] estimated intr (px @ {h_b}x{w_b}): fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}")

    return dict(
        raw_images=raw_imgs,
        c2w=c2w,
        intr=(fx, fy, cx, cy),
        image_size=(h_b, w_b),
    )


def _estimate_intrinsics_from_local_points(local_pts: np.ndarray,
                                            conf: Optional[np.ndarray] = None,
                                            ) -> tuple[float, float, float, float]:
    """Assume centered principal point; solve fx, fy by robust median.

    local_pts: (N, H, W, 3) — camera-frame XYZ for each pixel
    conf:      (N, H, W, 1) sigmoid logits (optional)
    """
    N, H, W, _ = local_pts.shape
    cx = W / 2.0
    cy = H / 2.0
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    ux = (u - cx).astype(np.float64)[None]     # (1, H, W)
    vy = (v - cy).astype(np.float64)[None]
    X = local_pts[..., 0].astype(np.float64)
    Y = local_pts[..., 1].astype(np.float64)
    Z = local_pts[..., 2].astype(np.float64)

    # X/Z = (u-cx)/fx  →  fx = (u-cx) Z / X
    eps = 1e-3
    valid_fx = (Z > 0.05) & (np.abs(X) > eps) & (np.abs(ux) > 0.5)
    valid_fy = (Z > 0.05) & (np.abs(Y) > eps) & (np.abs(vy) > 0.5)
    if conf is not None:
        c = 1.0 / (1.0 + np.exp(-conf.squeeze(-1)))   # sigmoid
        valid_fx &= (c > 0.5)
        valid_fy &= (c > 0.5)

    fx_pix = ux * Z / X
    fy_pix = vy * Z / Y
    fx = float(np.median(fx_pix[valid_fx])) if valid_fx.any() else max(W, H) * 1.0
    fy = float(np.median(fy_pix[valid_fy])) if valid_fy.any() else max(W, H) * 1.0
    # fx/fy can come out negative if Pi3's convention has flipped sign on X; take abs.
    return abs(fx), abs(fy), cx, cy


# ---------------------------------------------------------------------------
# Pi3 result → scene_data (same format scene_loaders returns)
# ---------------------------------------------------------------------------

def pi3_to_scene_data(pi3_result: dict, target_h: int, target_w: int) -> dict:
    return pack_estimator_scene_data(
        pi3_result, target_h, target_w,
        scene_name="pi3_scene", source="pi3_estimated",
    )
