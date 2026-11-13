"""External-scene loaders for the viser viewer.

Supports two input formats (auto-detected):

1. **COLMAP text**: a directory containing ``sparse/0/cameras.txt`` and
   ``sparse/0/images.txt`` plus an ``images/`` folder. Standard COLMAP output.
   Convention: ``qvec``+``tvec`` are world-to-camera in OpenCV convention.

2. **nerfstudio transforms.json**: a JSON file with a ``frames`` array, each
   frame having ``file_path`` and ``transform_matrix``. Camera intrinsics may
   be at the JSON root or per-frame. Convention: ``transform_matrix`` is
   camera-to-world in **OpenGL** convention (Y up, Z back); we flip Y and Z
   columns to convert to DVSM's OpenCV convention.

The loaders return tensors shaped exactly like ``PresampledTestingDataset``
emits, so ``recon_scene.py`` can reuse the same downstream code.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# COLMAP binary model IDs and their (num_params, [(SIMPLE_)PINHOLE-equivalent param indices])
# https://colmap.github.io/format.html#camera-models
_COLMAP_MODEL_BY_ID = {
    0: ("SIMPLE_PINHOLE",         3),   # f, cx, cy
    1: ("PINHOLE",                4),   # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL",          4),   # f, cx, cy, k
    3: ("RADIAL",                 5),   # f, cx, cy, k1, k2
    4: ("OPENCV",                 8),   # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE",         8),
    6: ("FULL_OPENCV",           12),
    7: ("FOV",                    5),
    8: ("SIMPLE_RADIAL_FISHEYE",  4),
    9: ("RADIAL_FISHEYE",         5),
   10: ("THIN_PRISM_FISHEYE",    12),
}


# ---------------------------------------------------------------------------
# COLMAP text parser
# ---------------------------------------------------------------------------

def _parse_colmap_cameras_txt(path: Path) -> dict[int, dict]:
    """Returns {camera_id: {model, width, height, fx, fy, cx, cy}}.

    Supports SIMPLE_PINHOLE / PINHOLE / SIMPLE_RADIAL / RADIAL / OPENCV.
    Distortion coefficients are ignored.
    """
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            fx = fy = f
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[:4]
        elif model in ("SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            raise ValueError(
                f"COLMAP camera model {model!r} not supported (camera_id={cam_id})"
            )
        cameras[cam_id] = dict(
            model=model, width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
    return cameras


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP-convention quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qw * qx)],
        [    2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _parse_colmap_images_txt(path: Path) -> list[dict]:
    """Returns list of {image_id, name, camera_id, c2w_r [3,3], c2w_t [3]}.

    Each image in images.txt occupies TWO lines; we only care about the first.
    qvec+tvec are w2c (OpenCV); we convert to c2w.
    """
    entries = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        # First line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]])  # w, x, y, z
        tvec = np.array([float(x) for x in parts[5:8]])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        # Second line: 2D feature observations — skip
        i += 2

        w2c_r = _qvec_to_rotmat(qvec)
        w2c_t = tvec
        c2w_r = w2c_r.T
        c2w_t = -c2w_r @ w2c_t
        entries.append(dict(
            image_id=image_id, name=name, camera_id=cam_id,
            c2w_r=c2w_r.astype(np.float32),
            c2w_t=c2w_t.astype(np.float32),
        ))
    return entries


def _parse_colmap_cameras_bin(path: Path) -> dict[int, dict]:
    """Binary cameras.bin parser (Mip-NeRF 360 et al. ship binary)."""
    out = {}
    with open(path, "rb") as f:
        (n_cams,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_cams):
            cam_id, model_id, w, h = struct.unpack("<IiQQ", f.read(24))
            if model_id not in _COLMAP_MODEL_BY_ID:
                raise ValueError(f"unknown COLMAP model id {model_id}")
            model_name, n_params = _COLMAP_MODEL_BY_ID[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            if model_name == "SIMPLE_PINHOLE":
                fc, cx, cy = params[:3]
                fx = fy = fc
            elif model_name in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE",
                                "THIN_PRISM_FISHEYE"):
                fx, fy, cx, cy = params[:4]
            elif model_name in ("SIMPLE_RADIAL", "RADIAL",
                                "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
                fc, cx, cy = params[:3]
                fx = fy = fc
            elif model_name == "FOV":
                fc, cx, cy = params[:3]
                fx = fy = fc
            else:
                raise ValueError(f"unhandled model {model_name}")
            out[cam_id] = dict(
                model=model_name, width=int(w), height=int(h),
                fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            )
    return out


def _parse_colmap_images_bin(path: Path) -> list[dict]:
    """Binary images.bin parser."""
    entries = []
    with open(path, "rb") as f:
        (n_imgs,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_imgs):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack(
                "<IdddddddI", f.read(4 + 8 * 7 + 4),
            )
            # Null-terminated name
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if not c or c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            # Skip the 2D points block: uint64 num_points then 24 bytes per point.
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)

            w2c_r = _qvec_to_rotmat(np.array([qw, qx, qy, qz]))
            w2c_t = np.array([tx, ty, tz])
            c2w_r = w2c_r.T
            c2w_t = -c2w_r @ w2c_t
            entries.append(dict(
                image_id=image_id, name=name, camera_id=cam_id,
                c2w_r=c2w_r.astype(np.float32),
                c2w_t=c2w_t.astype(np.float32),
            ))
    return entries


def _find_colmap_dir(scene_dir: Path) -> tuple[Path, Path, str]:
    """Locate cameras + images files (binary or text) inside a scene dir.

    Returns (cams_path, imgs_path, format) where format ∈ {"bin", "txt"}.
    """
    candidates = [
        scene_dir / "sparse" / "0",
        scene_dir / "sparse",
        scene_dir / "colmap" / "sparse" / "0",
        scene_dir,
    ]
    for d in candidates:
        for ext, fmt in (("bin", "bin"), ("txt", "txt")):
            cams = d / f"cameras.{ext}"
            imgs = d / f"images.{ext}"
            if cams.exists() and imgs.exists():
                return cams, imgs, fmt
    raise FileNotFoundError(
        f"could not find cameras.{{bin,txt}} + images.{{bin,txt}} under {scene_dir}"
    )


def _find_images_dir(scene_dir: Path, subdir: Optional[str] = None) -> Path:
    if subdir is not None:
        d = scene_dir / subdir
        if d.is_dir():
            return d
        raise FileNotFoundError(f"{d} does not exist")
    for sub in ("images", "image", "imgs"):
        d = scene_dir / sub
        if d.is_dir():
            return d
    raise FileNotFoundError(f"no images/ directory under {scene_dir}")


# ---------------------------------------------------------------------------
# Image loading + resize/crop with intrinsics adjustment
# ---------------------------------------------------------------------------

def _resize_and_crop(
    img: np.ndarray, intr_px: dict, target_h: int, target_w: int,
) -> tuple[np.ndarray, dict]:
    """Aspect-preserving resize then center-crop to (target_h, target_w).
    Returns the cropped image and the adjusted pixel intrinsics."""
    from PIL import Image

    src_h, src_w = img.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    img_pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    crop_l = (new_w - target_w) // 2
    crop_t = (new_h - target_h) // 2
    img_pil = img_pil.crop((crop_l, crop_t, crop_l + target_w, crop_t + target_h))

    intr_new = dict(
        fx=intr_px["fx"] * scale,
        fy=intr_px["fy"] * scale,
        cx=intr_px["cx"] * scale - crop_l,
        cy=intr_px["cy"] * scale - crop_t,
    )
    return np.asarray(img_pil), intr_new


def pack_estimator_scene_data(est_result: dict, target_h: int, target_w: int,
                              scene_name: str, source: str) -> dict:
    """Generic packer: take a pose-estimator result dict and produce the
    scene_data shape consumed by build_renderer.

    `est_result` must contain:
        raw_images  : list[HxWx3 uint8 ndarray] in native resolution
        c2w         : (V, 4, 4) camera-to-world (OpenCV)
        intr        : (fx, fy, cx, cy) in pixels, valid at `image_size`
        image_size  : (H, W) of the input fed to the estimator
    """
    raw_imgs = est_result["raw_images"]
    c2w = est_result["c2w"]
    fx, fy, cx, cy = est_result["intr"]
    h_src, w_src = est_result["image_size"]
    V = len(raw_imgs)

    img_arr = np.zeros((V, target_h, target_w, 3), dtype=np.uint8)
    intr_arr = np.zeros((V, 4), dtype=np.float32)
    c2w_r_arr = np.zeros((V, 3, 3), dtype=np.float32)
    c2w_t_arr = np.zeros((V, 3), dtype=np.float32)

    for i, raw in enumerate(raw_imgs):
        # Estimator's intrinsics live at h_src × w_src. The native image is
        # bigger (or differently sized), so scale to native size first, then
        # let _resize_and_crop scale them again to the DVSM target.
        native_h, native_w = raw.shape[:2]
        sx = native_w / w_src
        sy = native_h / h_src
        intr_native = dict(
            fx=fx * sx, fy=fy * sy,
            cx=cx * sx, cy=cy * sy,
        )
        img_resized, intr_new = _resize_and_crop(raw, intr_native, target_h, target_w)
        img_arr[i] = img_resized
        intr_arr[i] = (
            intr_new["fx"] / target_w, intr_new["fy"] / target_h,
            intr_new["cx"] / target_w, intr_new["cy"] / target_h,
        )
        c2w_r_arr[i] = c2w[i, :3, :3]
        c2w_t_arr[i] = c2w[i, :3, 3]

    image_t = torch.from_numpy(img_arr).permute(0, 3, 1, 2).float() / 255.0
    return dict(
        image=image_t,
        intr=torch.from_numpy(intr_arr),
        c2w_r=torch.from_numpy(c2w_r_arr),
        c2w_t=torch.from_numpy(c2w_t_arr),
        H=target_h, W=target_w,
        scene_name=scene_name,
        source=source,
        n_input_views=V,
    )


# ---------------------------------------------------------------------------
# COLMAP scene loader
# ---------------------------------------------------------------------------

def load_colmap_scene(
    scene_dir: Path,
    target_h: int,
    target_w: int,
    max_views: Optional[int] = None,
    stride: int = 1,
    sort_by_name: bool = True,
    images_subdir: Optional[str] = None,
) -> dict:
    """Load a COLMAP scene.

    Args:
        images_subdir: e.g. "images_4" to use Mip-NeRF 360 downscaled images.
            cameras.bin intrinsics are pre-scaled to match the actual image
            resolution we end up reading from disk.
    """
    import imageio.v3 as iio

    cams_path, imgs_path, fmt = _find_colmap_dir(scene_dir)
    images_dir = _find_images_dir(scene_dir, subdir=images_subdir)

    print(f"[colmap] format: {fmt}")
    print(f"[colmap] cameras: {cams_path}")
    print(f"[colmap] images:  {imgs_path}")
    print(f"[colmap] images dir: {images_dir}")

    if fmt == "bin":
        cameras = _parse_colmap_cameras_bin(cams_path)
        image_entries = _parse_colmap_images_bin(imgs_path)
    else:
        cameras = _parse_colmap_cameras_txt(cams_path)
        image_entries = _parse_colmap_images_txt(imgs_path)
    if sort_by_name:
        image_entries.sort(key=lambda e: _natural_key(e["name"]))
    image_entries = image_entries[::stride]
    if max_views is not None:
        # Evenly subsample to N views.
        if len(image_entries) > max_views:
            idxs = np.linspace(0, len(image_entries) - 1, max_views).round().astype(int)
            image_entries = [image_entries[i] for i in idxs]
    V = len(image_entries)
    print(f"[colmap] loading {V} views at {target_h}x{target_w} ...")

    img_arr = np.zeros((V, target_h, target_w, 3), dtype=np.uint8)
    intr_arr = np.zeros((V, 4), dtype=np.float32)
    c2w_r_arr = np.zeros((V, 3, 3), dtype=np.float32)
    c2w_t_arr = np.zeros((V, 3), dtype=np.float32)

    for i, entry in enumerate(image_entries):
        cam = cameras[entry["camera_id"]]
        img_path = images_dir / entry["name"]
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        img = iio.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        # Pre-scale intrinsics if we're reading a downscaled image variant
        # (e.g. Mip-NeRF 360's images_2/_4/_8 vs the native cameras.bin width/height).
        img_h, img_w = img.shape[:2]
        if img_w != cam["width"] or img_h != cam["height"]:
            sx = img_w / cam["width"]
            sy = img_h / cam["height"]
            intr_px = dict(fx=cam["fx"] * sx, fy=cam["fy"] * sy,
                           cx=cam["cx"] * sx, cy=cam["cy"] * sy)
        else:
            intr_px = dict(fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"])
        img_resized, intr_new = _resize_and_crop(img, intr_px, target_h, target_w)
        img_arr[i] = img_resized
        intr_arr[i] = (
            intr_new["fx"] / target_w,
            intr_new["fy"] / target_h,
            intr_new["cx"] / target_w,
            intr_new["cy"] / target_h,
        )
        c2w_r_arr[i] = entry["c2w_r"]
        c2w_t_arr[i] = entry["c2w_t"]

    image_t = torch.from_numpy(img_arr).permute(0, 3, 1, 2).float() / 255.0
    return dict(
        image=image_t,
        intr=torch.from_numpy(intr_arr),
        c2w_r=torch.from_numpy(c2w_r_arr),
        c2w_t=torch.from_numpy(c2w_t_arr),
        H=target_h, W=target_w,
        scene_name=scene_dir.name,
        source="colmap_text",
        n_input_views=V,
    )


# ---------------------------------------------------------------------------
# nerfstudio transforms.json loader
# ---------------------------------------------------------------------------

# OpenGL → OpenCV: flip Y and Z columns of the camera-to-world rotation.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def load_transforms_json(
    json_path: Path,
    target_h: int,
    target_w: int,
    max_views: Optional[int] = None,
    stride: int = 1,
    convention: str = "opengl",
) -> dict:
    """Load a nerfstudio-style transforms.json.

    Args:
        convention: "opengl" (default — nerfstudio's convention; we flip to
            OpenCV) or "opencv" (already OpenCV — no flip).
    """
    import imageio.v3 as iio

    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)
    base_dir = json_path.parent

    frames = data["frames"][::stride]
    if max_views is not None and len(frames) > max_views:
        idxs = np.linspace(0, len(frames) - 1, max_views).round().astype(int)
        frames = [frames[i] for i in idxs]
    V = len(frames)
    print(f"[transforms_json] {V} frames; convention={convention}; "
          f"target {target_h}x{target_w}")

    def _intr_for_frame(frame):
        fx = frame.get("fl_x", data.get("fl_x"))
        fy = frame.get("fl_y", data.get("fl_y", fx))
        cx = frame.get("cx", data.get("cx"))
        cy = frame.get("cy", data.get("cy"))
        w = frame.get("w", data.get("w"))
        h = frame.get("h", data.get("h"))
        return fx, fy, cx, cy, w, h

    img_arr = np.zeros((V, target_h, target_w, 3), dtype=np.uint8)
    intr_arr = np.zeros((V, 4), dtype=np.float32)
    c2w_r_arr = np.zeros((V, 3, 3), dtype=np.float32)
    c2w_t_arr = np.zeros((V, 3), dtype=np.float32)

    for i, fr in enumerate(frames):
        img_path = base_dir / fr["file_path"]
        if not img_path.exists():
            # nerfstudio sometimes drops the file extension; try a few.
            for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
                p2 = img_path.with_suffix(ext)
                if p2.exists():
                    img_path = p2
                    break
            else:
                raise FileNotFoundError(img_path)
        img = iio.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        fx, fy, cx, cy, w, h = _intr_for_frame(fr)
        if w is None: w = img.shape[1]
        if h is None: h = img.shape[0]
        if cx is None: cx = w / 2.0
        if cy is None: cy = h / 2.0
        img_resized, intr_new = _resize_and_crop(
            img, dict(fx=fx, fy=fy, cx=cx, cy=cy), target_h, target_w,
        )
        img_arr[i] = img_resized
        intr_arr[i] = (
            intr_new["fx"] / target_w, intr_new["fy"] / target_h,
            intr_new["cx"] / target_w, intr_new["cy"] / target_h,
        )

        c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        c2w_r = c2w[:3, :3]
        c2w_t = c2w[:3, 3]
        if convention == "opengl":
            c2w_r = c2w_r @ _GL_TO_CV   # flip camera-local Y, Z
        elif convention != "opencv":
            raise ValueError(f"unknown convention: {convention}")
        c2w_r_arr[i] = c2w_r
        c2w_t_arr[i] = c2w_t

    image_t = torch.from_numpy(img_arr).permute(0, 3, 1, 2).float() / 255.0
    return dict(
        image=image_t,
        intr=torch.from_numpy(intr_arr),
        c2w_r=torch.from_numpy(c2w_r_arr),
        c2w_t=torch.from_numpy(c2w_t_arr),
        H=target_h, W=target_w,
        scene_name=json_path.parent.name,
        source="transforms_json",
        n_input_views=V,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_external_scene(
    path: Path,
    target_h: int,
    target_w: int,
    max_views: Optional[int] = None,
    stride: int = 1,
    convention: str = "opengl",
    images_subdir: Optional[str] = None,
) -> dict:
    """Auto-detect format and load.

    Acceptable ``path``:
      - a directory with ``transforms.json`` inside → nerfstudio
      - a directory with ``sparse/.../cameras.{bin,txt}`` inside → COLMAP
      - a ``.json`` file → nerfstudio transforms.json
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        return load_transforms_json(
            path, target_h, target_w, max_views=max_views, stride=stride,
            convention=convention,
        )
    if path.is_dir():
        if (path / "transforms.json").exists():
            return load_transforms_json(
                path / "transforms.json", target_h, target_w,
                max_views=max_views, stride=stride, convention=convention,
            )
        # try COLMAP
        try:
            return load_colmap_scene(
                path, target_h, target_w,
                max_views=max_views, stride=stride,
                images_subdir=images_subdir,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{path} has neither transforms.json nor COLMAP sparse/*/cameras.{{bin,txt}}: {e}"
            )
    raise FileNotFoundError(path)


# ---------------------------------------------------------------------------

def _natural_key(s: str):
    """Sort by embedded integers naturally (e.g. img2 < img10)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------------
# Input file kinds + video frame extraction (shared by recon_scene.py / gradio)
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_video_frames(video_path: Path, out_dir: Path, n_frames: int) -> int:
    """Sample ``n_frames`` evenly-spaced frames from ``video_path`` and write
    them as JPEGs into ``out_dir``. Returns the number of frames actually
    written. Tries cv2 first, falls back to imageio.

    Frames keep the video's original aspect; the recon pipeline will resize.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    # Try OpenCV (much faster on long videos due to keyframe seeking).
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError("cv2 saw 0 frames")
        n_take = min(n_frames, total)
        idxs = np.linspace(0, total - 1, n_take).round().astype(int)
        for k, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            cv2.imwrite(str(out_dir / f"frame_{k:04d}.jpg"),
                        frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            n_written += 1
        cap.release()
        return n_written
    except Exception as e:
        print(f"[video] cv2 path failed ({e}); falling back to imageio")

    # imageio fallback (works for most containers via PyAV).
    import imageio.v3 as iio
    frames = list(iio.imiter(str(video_path), plugin="pyav"))
    total = len(frames)
    if total == 0:
        raise RuntimeError(f"imageio saw 0 frames in {video_path}")
    n_take = min(n_frames, total)
    idxs = np.linspace(0, total - 1, n_take).round().astype(int)
    for k, idx in enumerate(idxs):
        iio.imwrite(out_dir / f"frame_{k:04d}.jpg", frames[idx])
        n_written += 1
    return n_written
