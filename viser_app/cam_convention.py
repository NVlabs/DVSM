"""Camera convention conversion between viser and DVSM.

Both use OpenCV camera-local convention (+X right, +Y down, +Z forward) per
viser's API docs ("Like all cameras in the viser Python API, frustums follow
the OpenCV [+Z forward, +X right, +Y down] convention"). So the wxyz/position
mapping is direct -- no camera-local axis flip.

The viewer applies an additional WORLD rotation outside of this module to
align DVSM's normalized world (with -Y up) to viser/three.js native (+Y up).
"""
from __future__ import annotations

import numpy as np
import torch


def wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = wxyz
    n = w * w + x * x + y * y + z * z
    s = 0.0 if n < 1e-12 else 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy        ],
        [xy + wz,         1.0 - (xx + zz), yz - wx        ],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float32)


def matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w, x, y, z). Stable branch selection."""
    R = R.astype(np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def fov_to_normalized_intrinsics(fov_y: float, aspect: float) -> np.ndarray:
    """Vertical FoV (rad) and aspect (W/H) -> DVSM-normalized intrinsics [fx, fy, cx, cy].

    DVSM normalization: fx*W = focal_x_pixels, fy*H = focal_y_pixels.
    Pinhole with square pixels: tan(fov_x/2) = aspect * tan(fov_y/2).
    """
    tan_half = np.tan(0.5 * fov_y)
    fy = 0.5 / tan_half
    fx = fy / aspect
    return np.array([fx, fy, 0.5, 0.5], dtype=np.float32)


def normalized_intrinsics_to_fov(intr: np.ndarray) -> tuple[float, float]:
    """DVSM intrinsics [fx, fy, cx, cy] -> (fov_y in radians, aspect W/H)."""
    fx, fy = float(intr[0]), float(intr[1])
    fov_y = 2.0 * np.arctan(0.5 / fy)
    aspect = fy / fx
    return fov_y, aspect


def viser_cam_to_dvsm(
    position: np.ndarray,
    wxyz: np.ndarray,
    fov_y: float,
    aspect: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert viser ClientHandle.camera state to DVSM render inputs.

    Returns:
        intr:  [1, 1, 4]   normalized [fx, fy, cx, cy]
        c2w_r: [1, 1, 3, 3]
        c2w_t: [1, 1, 3]
    """
    R_cv = wxyz_to_matrix(np.asarray(wxyz, dtype=np.float32))
    t = np.asarray(position, dtype=np.float32)
    intr_np = fov_to_normalized_intrinsics(fov_y, aspect)
    intr = torch.from_numpy(intr_np).to(device=device, dtype=dtype).view(1, 1, 4)
    c2w_r = torch.from_numpy(R_cv).to(device=device, dtype=dtype).view(1, 1, 3, 3)
    c2w_t = torch.from_numpy(t).to(device=device, dtype=dtype).view(1, 1, 3)
    return intr, c2w_r, c2w_t


def dvsm_cam_to_viser(
    intr: torch.Tensor | np.ndarray,
    c2w_r: torch.Tensor | np.ndarray,
    c2w_t: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Inverse of viser_cam_to_dvsm. Accepts unbatched or [1,1,...] shapes.

    Returns:
        position: (3,) numpy
        wxyz:     (4,) numpy, w-first
        fov_y:    float, radians
        aspect:   float, W/H
    """
    def _to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)
    intr_np = _to_np(intr).reshape(-1)[:4]
    R_cv = _to_np(c2w_r).reshape(3, 3)
    t = _to_np(c2w_t).reshape(3).astype(np.float32)
    wxyz = matrix_to_wxyz(R_cv)
    fov_y, aspect = normalized_intrinsics_to_fov(intr_np.astype(np.float32))
    return t, wxyz, fov_y, aspect
