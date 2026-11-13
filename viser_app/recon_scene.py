"""Stage A: reconstruct one external scene -> self-contained .recon.pt artifact.

Loads a trained DVSM/TTB-DVSM checkpoint, builds the per-block fast weights
(kv_lst) + per-scene PosePreprocessor state for one scene, and saves a
self-contained artifact for viser_app/viewer.py (no input images needed after).

Pose source (--pose):
  - auto (default): use COLMAP / transforms.json found under --input, otherwise
    estimate poses with π³ (a flat image folder needs no poses).
  - colmap: read COLMAP ``sparse/*/cameras.{bin,txt}`` or nerfstudio
    ``transforms.json`` from --input (a directory or a .json file).
  - pi3:    estimate camera-to-world poses + intrinsics with π³ from a
            flat folder of images OR a video file (no cameras.bin, no transforms.json).
            A video is decoded to --max_views evenly-spaced frames first.

Recon size defaults to: keep the input's aspect ratio, resize the long edge to
--long_edge (960), snapped to patch_size. Override with --target_h/--target_w.

Usage:
    # auto: COLMAP/transforms if present, else π³ (image folder / video)
    python -m viser_app.recon_scene \\
        --ckpt_root outputs/dl3dv/960p_ttb_dvsm_minus \\
        --input /path/to/scene_or_video --max_views 32 \\
        --out outputs/recon/scene.recon.pt

    # force π³ on a flat image folder, long edge 720
    python -m viser_app.recon_scene --pose pi3 --long_edge 720 \\
        --ckpt_root outputs/dl3dv/960p_ttb_dvsm_minus \\
        --input /path/to/images --max_views 32 \\
        --out outputs/recon/scene.recon.pt
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dvsm.config import cfg, update_config_from_ckpt
from dvsm.models.DVSM import DecoderViewSynthesisModel, load_checkpoint, resolve_checkpoint_path
from viser_app.scene_loaders import (
    VIDEO_EXTS, _find_colmap_dir, extract_video_frames, load_external_scene,
)


# ---------------------------------------------------------------------------
# Pose source
# ---------------------------------------------------------------------------
def detect_pose_source(input_path: Path) -> str:
    """auto: 'colmap' if --input has COLMAP sparse / transforms.json, else 'pi3'.

    A video file or a flat image folder has no poses -> 'pi3'.
    """
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return "pi3"
    if p.is_file() and p.suffix == ".json":
        return "colmap"
    if p.is_dir():
        if (p / "transforms.json").exists():
            return "colmap"
        try:
            _find_colmap_dir(p)
            return "colmap"
        except Exception:
            pass
    return "pi3"


def _snap(v: float, patch: int) -> int:
    return max(patch, (int(round(v)) // patch) * patch)


def derive_target_hw(sample_img: Path, long_edge: int, patch: int) -> tuple:
    """Keep the input's aspect ratio, resize the LONG edge to ``long_edge``, and
    snap both axes to multiples of ``patch`` (mirrors the gradio demo's logic)."""
    from PIL import Image
    with Image.open(sample_img) as im:
        iw, ih = im.size
    aspect = iw / ih  # >1 landscape, <1 portrait
    le = _snap(long_edge, patch)
    if aspect >= 1.0:
        w, h = le, _snap(le / aspect, patch)
    else:
        h, w = le, _snap(le * aspect, patch)
    return h, w


def _peek_image(input_path: Path) -> Path:
    """First image under --input (a dir) or its parent (a .json), to infer aspect."""
    from viser_app.scene_loaders import IMAGE_EXTS
    p = Path(input_path)
    root = p.parent if p.is_file() else p
    cands = sorted(f for f in root.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    if not cands:
        raise FileNotFoundError(f"no image under {root} to infer aspect ratio")
    return cands[0]


def _resolve_target(args, sample_img: Path, patch: int) -> tuple:
    """--target_h/--target_w if both given, else derive from aspect + --long_edge."""
    if args.target_h and args.target_w:
        return args.target_h, args.target_w
    th, tw = derive_target_hw(sample_img, args.long_edge, patch)
    print(f"[recon] target derived from aspect: {tw}x{th} "
          f"(long_edge={args.long_edge}, patch={patch})", flush=True)
    return th, tw


def load_scene_data(args, pose: str) -> dict:
    """Build scene_data {image, intr, c2w_r, c2w_t, scene_name, ...} for recon."""
    # Load the checkpoint config early to know patch_size for target snapping.
    update_config_from_ckpt(args.ckpt_root)
    patch = cfg.model.patch_size
    if pose == "colmap":
        target_h, target_w = _resolve_target(args, _peek_image(args.input), patch)
        print(f"[recon] pose=colmap  loading {args.input}", flush=True)
        scene_data = load_external_scene(
            args.input, target_h=target_h, target_w=target_w,
            max_views=args.max_views, stride=args.stride,
            convention=args.convention, images_subdir=args.images_subdir,
        )
    elif pose == "pi3":
        from viser_app.pi3_estimator import (
            _list_images, load_pi3_model, pi3_to_scene_data, run_pi3,
        )
        # A video file -> extract evenly-spaced frames to a temp folder first.
        image_root = Path(args.input)
        if image_root.is_file() and image_root.suffix.lower() in VIDEO_EXTS:
            n_take = args.max_views or 64
            tmp = Path(tempfile.mkdtemp(prefix="recon_video_"))
            n = extract_video_frames(image_root, tmp, n_take)
            print(f"[recon] extracted {n} frames from video {image_root.name} -> {tmp}", flush=True)
            image_root = tmp
        image_paths = _list_images(image_root)[::args.stride]
        if args.max_views and len(image_paths) > args.max_views:
            idxs = np.linspace(0, len(image_paths) - 1, args.max_views).round().astype(int)
            image_paths = [image_paths[i] for i in idxs]
        target_h, target_w = _resolve_target(args, image_paths[0], patch)
        print(f"[recon] pose=pi3  {len(image_paths)} images from {args.input}", flush=True)
        pi3 = load_pi3_model(flavor=args.pi3_flavor)
        est = run_pi3(pi3, image_paths, image_size=args.pi3_image_size)
        del pi3; torch.cuda.empty_cache()
        scene_data = pi3_to_scene_data(est, target_h, target_w)
    else:
        raise ValueError(f"unknown pose source: {pose}")
    print(f"[recon] scene packed: V={scene_data['n_input_views']} "
          f"{scene_data['H']}x{scene_data['W']}  source={scene_data.get('source', pose)}", flush=True)
    return scene_data


# ---------------------------------------------------------------------------
# Reconstruction core (shared by every pose source)
# ---------------------------------------------------------------------------
def reconstruct(ckpt_root: Path, ckpt_name: str, scene_data: dict, cmd_lst: list[str]):
    update_config_from_ckpt(ckpt_root, cmd_lst)
    cfg.model.use_compile = False  # avoid first-call compile cost during recon
    ckpt_path = resolve_checkpoint_path(ckpt_root, ckpt_name)
    print(f"[recon] block_type={cfg.model.block_type} patch={cfg.model.patch_size}")

    model = DecoderViewSynthesisModel(**cfg.model).cuda().eval()
    load_checkpoint(model, ckpt_path)

    image = scene_data["image"].cuda().unsqueeze(0)
    intr = scene_data["intr"].cuda().unsqueeze(0)
    c2w_r = scene_data["c2w_r"].cuda().unsqueeze(0)
    c2w_t = scene_data["c2w_t"].cuda().unsqueeze(0)

    H, W = image.shape[-2:]
    assert H % cfg.model.patch_size == 0 and W % cfg.model.patch_size == 0, (
        f"target {H}x{W} not divisible by patch_size {cfg.model.patch_size}")

    print(f"[recon] input image shape {tuple(image.shape)}")
    print(f"[recon] running reconstruction...")
    with torch.no_grad():
        renderer = model(image, intr, c2w_r, c2w_t)
    print(f"[recon] kv_lst length {len(renderer.kv_lst)}")

    batch = {"ds_names": [scene_data["scene_name"]]}
    return renderer, batch, image.shape, c2w_r, c2w_t, intr


def _move_kv_to_cpu(kv):
    if kv is None:
        return None
    if isinstance(kv, dict):
        return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kv.items()}
    if isinstance(kv, (tuple, list)):
        return type(kv)(_move_kv_to_cpu(v) for v in kv)
    if isinstance(kv, torch.Tensor):
        return kv.cpu()
    return kv


def save_artifact(renderer, batch, image_shape, c2w_r, c2w_t, intr,
                  ckpt_root: Path, ckpt_name: str, out_path: Path):
    pp = renderer.pose_preprocessor
    artifact = {
        "kv_lst": [_move_kv_to_cpu(kv) for kv in renderer.kv_lst],
        "pose_preproc": {"r": pp.r.cpu(), "t": pp.t.cpu(), "s": pp.s.cpu()},
        "model_cfg": dict(cfg.model),
        "ckpt_root": str(ckpt_root),
        "ckpt_name": ckpt_name,
        "scene_meta": {
            "ds_name": batch.get("ds_names", ["unknown"])[0],
            "n_input_views": image_shape[1],
            "input_res": (int(image_shape[-2]), int(image_shape[-1])),
            "input_c2w_r": c2w_r[0].cpu(),
            "input_c2w_t": c2w_t[0].cpu(),
            "input_intr": intr[0].cpu(),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    sz = out_path.stat().st_size / (1024 ** 2)
    print(f"[recon] saved {out_path}  ({sz:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", type=Path, required=True,
                   help="Checkpoint directory: a Hugging Face release (config.json + "
                        "model.safetensors) or a train.py output (config.yaml + latest.pt).")
    p.add_argument("--ckpt_name", default="latest.pt",
                   help="Weight file inside a train.py output directory "
                        "(a Hugging Face release always uses model.safetensors).")
    p.add_argument("--input", type=Path, required=True,
                   help="Scene dir / transforms.json (COLMAP/nerfstudio), a flat image folder, "
                        "or a video file (image folder / video -> π³).")
    p.add_argument("--pose", default="auto", choices=("auto", "colmap", "pi3"),
                   help="Pose source. auto = COLMAP/transforms if present, else π³.")
    p.add_argument("--long_edge", type=int, default=960,
                   help="Default recon size: keep input aspect, resize long edge to this "
                        "(snapped to patch_size). Ignored if --target_h/--target_w are both set.")
    p.add_argument("--target_h", type=int, default=None,
                   help="Override recon height (else derived from aspect + --long_edge).")
    p.add_argument("--target_w", type=int, default=None,
                   help="Override recon width (else derived from aspect + --long_edge).")
    p.add_argument("--max_views", type=int, default=None,
                   help="Evenly sub-sample this many views (after --stride).")
    p.add_argument("--stride", type=int, default=1,
                   help="Sample every Nth frame BEFORE max_views subsampling.")
    p.add_argument("--out", type=Path, required=True)
    # COLMAP / transforms.json options
    p.add_argument("--convention", default="opengl", choices=("opengl", "opencv"),
                   help="transforms.json camera-local convention (ignored for COLMAP).")
    p.add_argument("--images_subdir", default=None,
                   help="COLMAP: image sub-folder (e.g. 'images_4'); intrinsics are pre-scaled.")
    # π³ options
    p.add_argument("--pi3_flavor", default="pi3x", choices=("pi3", "pi3x"))
    p.add_argument("--pi3_image_size", type=int, default=518)
    args, cmd_lst = p.parse_known_args()

    pose = detect_pose_source(args.input) if args.pose == "auto" else args.pose
    if args.pose == "auto":
        print(f"[recon] auto-detected pose source: {pose}", flush=True)

    scene_data = load_scene_data(args, pose)
    renderer, batch, image_shape, c2w_r, c2w_t, intr = reconstruct(
        args.ckpt_root, args.ckpt_name, scene_data, cmd_lst)
    save_artifact(renderer, batch, image_shape, c2w_r, c2w_t, intr,
                  args.ckpt_root, args.ckpt_name, args.out)


if __name__ == "__main__":
    main()
