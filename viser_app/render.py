"""Pure-DVSM render utilities (no viser dependency).

Splits the model-loading and NeuralRenderer reconstruction code out of
viewer.py so that headless tests and offline rendering scripts can use it
without pulling in viser.
"""
from __future__ import annotations

from pathlib import Path

import torch

from dvsm.config import cfg, update_config_from_ckpt
from dvsm.models.DVSM import (
    DecoderViewSynthesisModel, NeuralRenderer, load_checkpoint, resolve_checkpoint_path,
)
from dvsm.models.pose_preprocessor import PosePreprocessor


# Rotates DVSM normalized world (X right, Y down, Z forward; OpenCV) into
# viser/three.js native world (X right, Y up, Z toward viewer; OpenGL). Self-
# inverse since it is diag(1, -1, -1) (an involution).
WORLD_DVSM_TO_VISER = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32,
)


def apply_pp_single(pp: PosePreprocessor, c2w_r, c2w_t, apply_scale: bool = False):
    """DVSM raw pose -> viser world pose: rotate + translate via pp, then Rw.

    By default we OMIT pp.s. Reason: pp.s shrinks the scene to a unit sphere,
    which makes viser's orbit/pan/zoom feel too fast (controls are tuned for
    a "natural" metric scale). Keeping the original metric for viser and
    applying pp.s only inside the renderer gives natural control feel.

        c2w_r_n = pp.r @ c2w_r
        c2w_t_n = pp.r @ c2w_t + pp.t           (pp.s applied iff apply_scale)
        c2w_r_viser = Rw @ c2w_r_n
        c2w_t_viser = Rw @ c2w_t_n
    """
    r = pp.r[0]              # [3, 3]
    t = pp.t[0, 0]           # [3]
    r_n = r @ c2w_r
    t_n = r @ c2w_t + t
    if apply_scale:
        s = float(pp.s.reshape(-1)[0])
        t_n = t_n * s
    Rw = WORLD_DVSM_TO_VISER.to(r_n.dtype)
    r_v = Rw @ r_n
    t_v = Rw @ t_n
    return r_v, t_v


def load_model(ckpt_root: Path, ckpt_name: str = "latest.pt",
               cmd_lst: list[str] | None = None) -> DecoderViewSynthesisModel:
    """Build the model from a checkpoint directory: a train.py output
    (config.yaml + latest.pt) or a Hugging Face release (config.json + model.safetensors)."""
    update_config_from_ckpt(ckpt_root, cmd_lst or [])
    cfg.model.use_compile = True  # try compile + CUDA Graph together
    ckpt_path = resolve_checkpoint_path(ckpt_root, ckpt_name)
    model = DecoderViewSynthesisModel(**cfg.model).cuda().eval()
    load_checkpoint(model, ckpt_path)
    return model


def _kv_to_cuda(kv):
    if kv is None:
        return None
    if isinstance(kv, dict):
        return {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in kv.items()}
    if isinstance(kv, (tuple, list)):
        return type(kv)(_kv_to_cuda(v) for v in kv)
    if isinstance(kv, torch.Tensor):
        return kv.cuda()
    return kv


def build_renderer(model: DecoderViewSynthesisModel, artifact: dict) -> NeuralRenderer:
    """Re-materialize a NeuralRenderer from an artifact saved by recon_scene.py."""
    pp = PosePreprocessor.__new__(PosePreprocessor)
    pp.r = artifact["pose_preproc"]["r"].cuda()
    pp.t = artifact["pose_preproc"]["t"].cuda()
    pp.s = artifact["pose_preproc"]["s"].cuda()
    kv_lst = [_kv_to_cuda(kv) for kv in artifact["kv_lst"]]
    # NeuralRenderer no longer accepts chunk_centroids / chunk_K (k-means
    # routing is now baked into the kv_lst itself). We silently drop them from
    # older artifacts.
    return NeuralRenderer(
        pose_preprocessor=pp,
        model=model,
        kv_lst=kv_lst,
    )
