# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
from pathlib import Path
from yacs.config import CfgNode


cfg = CfgNode()

cfg.ckpt = CfgNode(dict(
    exp_dir = "outputs/exps/",
    exp_name = "dev",
    period = 1000,

    pretrained = "",
))

cfg.dataset = CfgNode(dict(
    # Training setups
    train_path = "configs/dl3dv_train.json",
    min_img_per_seq = 2,
    max_img_per_seq = 32,
    num_input_per_gpu = 32,
    num_target_per_gpu = 32,
    divisible_only = True,     # Only sample #views divisible by num_input_per_gpu
    fixed_img_per_seq = False, # All gpu load the same #views
    num_workers = 8,

    num_pixels = 256 * 256,

    ap_ratio_lst = "16:9",     # Enumerate aspect ratio as 16:9,1:1,...
    crop_pad_ratio = 0.0,
    skip_divisible_resize = False,

    randomize_static_time = False,

    # Validation setupts
    valid_path = "configs/dl3dv_test.json",
    valid_ap_ratio = "16:9",
    valid_crop_pad_ratio = 0.,
))

cfg.model = CfgNode(dict(
    patch_size=8,
    latent_dim=768,
    pose_enc='plucker',
    pose_avg_strategy='down',
    render_up_scale=0.,
    recon_up_scale=0.,
    auto_divisible_pad=False,

    backbone='',
    freeze_backbone=False,

    num_register=0,

    block_type='IMCM',
    num_layers=12,
    head_dim=64,
    expand_ratio=4,
    use_bias=False,
    use_qk_norm=True,
    use_gating=False,
    use_swiglumlp=False,
    use_sandwich=False,

    ttb_feedback=True,
    ttb_use_gating=True,
    ttb_head_dim=0,
    ttb_inter_multi=4.0,
    ttb_base_lr=0.01,
    ttb_muon_steps=5,
    ttb_num_steps=3,   # Number of inner-loop update steps on same data (>1 defers weight norm to last step)
    ttb_target_residual=True,  # True: TTB target = norm(x_img) - norm(x_cam); False: TTB target = norm(x_img)
    ttb_pre_cm_target=True,  # Use x_img BEFORE cross-view attn+MLP as TTB target (only affects CMTB/IMCMTB)
    ttb_use_mse_loss=True,  # Use MSE error-correction in TTB inner loop (False = Hebbian)
    ttb_per_step_norm=False,  # Weight norm every update step (LaCT-style) vs deferred to last step (default)
    ttb_share_qk_proj=True,  # Share Q/K projection in TTB (ensures read/write address space alignment)
    inner_grad_checkpoint=True,  # Gradient checkpoint each TTB inner loop step (save memory, recompute activations)

    ttb_per_cluster_K=0,
    ttb_per_cluster_boundary_ratio=0.7,

    use_amp=True,
    use_compile=True,
    grad_checkpoint=True,
))

cfg.optim = CfgNode(dict(
    iter = 100_000,
    lr = 4e-4,
    lr_scaler = 1.0,
    beta1 = 0.9,
    beta2 = 0.95,
    wd = 0.,

    grad_max_norm = 1.0,

    sche_warpup_iter = 2500,
    sche_anneal_strategy = "cos",
    sche_div_factor = 100.,

    lambda_percep = 0.2,
    lambda_ssim = 0.,
    lambda_lpips = 0.,
))

for i_cfg in cfg.values():
    i_cfg.set_new_allowed(True)

# Pristine copy of the model defaults (see update_config_from_ckpt)
_MODEL_DEFAULTS = cfg.model.clone()


def everytype2bool(v):
    if v.isnumeric():
        return bool(int(v))
    v = v.lower()
    if v in ['n', 'no', 'none', 'false']:
        return False
    return True


def update_argparser(parser):
    for name in cfg.keys():
        group = parser.add_argument_group(name)
        for key, value in getattr(cfg, name).items():
            full_key = f"{name}.{key}"
            t = type(value)

            if t == bool:
                group.add_argument(f"--{full_key}", default=value, type=everytype2bool)
            elif t == list:
                group.add_argument(f"--{full_key}", default=value, type=type(value[0]), nargs="*")
            elif t == tuple:
                group.add_argument(f"--{full_key}", default=value, type=type(value[0]), nargs=len(value))
            else:
                group.add_argument(f"--{full_key}", default=value, type=t)


def update_config(cfg_files, cmd_lst=[]):
    # Update from config files
    if isinstance(cfg_files, str):
        cfg_files = [cfg_files]
    for cfg_path in cfg_files:
        cfg.merge_from_file(cfg_path)

    if len(cmd_lst) == 0:
        return

    # Parse the arguments from command line
    internal_parser = argparse.ArgumentParser()
    update_argparser(internal_parser)
    internal_args = internal_parser.parse_args(cmd_lst)

    # Update from command line args
    for name in cfg.keys():
        cfg_subgroup = getattr(cfg, name)
        for key in cfg_subgroup.keys():
            full_key = f"{name}.{key}"
            arg_val = getattr(internal_args, full_key)
            # Check if the default values is updated
            if internal_parser.get_default(full_key) != arg_val:
                cfg_subgroup[key] = arg_val


def update_config_from_ckpt(ckpt_root, cmd_lst=[]):
    """Update cfg from the config stored in a checkpoint directory.

    Two checkpoint layouts are supported:
      - train.py output:      <ckpt_root>/config.yaml (full config)   + latest.pt
      - Hugging Face release: <ckpt_root>/config.json (model options) + model.safetensors
    """
    ckpt_root = Path(ckpt_root)
    yaml_path = ckpt_root / 'config.yaml'
    json_path = ckpt_root / 'config.json'

    if yaml_path.exists():
        update_config([yaml_path], cmd_lst)
        return

    if not json_path.exists():
        raise FileNotFoundError(
            f"No config.yaml or config.json found in checkpoint directory: {ckpt_root}")

    # config.json only holds the model constructor arguments. Start from the
    # defaults so that options of a previously loaded checkpoint do not carry over.
    with open(json_path) as f:
        model_kwargs = json.load(f)
    for key in list(cfg.model.keys()):
        if key not in _MODEL_DEFAULTS:
            del cfg.model[key]
    for key, value in _MODEL_DEFAULTS.items():
        cfg.model[key] = value
    for key, value in model_kwargs.items():
        # JSON may store a whole-number float as int (e.g. "recon_up_scale": 2)
        if isinstance(cfg.model.get(key), float) and type(value) == int:
            value = float(value)
        cfg.model[key] = value

    update_config([], cmd_lst)
