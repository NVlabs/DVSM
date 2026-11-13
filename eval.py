import os
import sys
import torch
from datetime import timedelta

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = '1000'

global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl', timeout=timedelta(seconds=3600))
torch._dynamo.config.optimize_ddp = False

PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

import json
import tqdm
import time
import imageio
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline

from dvsm.config import cfg, update_config_from_ckpt
from dvsm.dataset import PresampledTestingDataset
from dvsm.loss import LossComputer
from dvsm.models.DVSM import DecoderViewSynthesisModel, load_checkpoint, resolve_checkpoint_path



def _fix_longer_side(image, fix_size, patch_size):
    """Resize image so longer side == fix_size; shorter side snapped to patch_size multiple.
    Returns (image, oH, oW)."""
    iH, iW = image.shape[-2:]
    if iW > iH:
        oW = fix_size
        oH = round(iH * oW / iW / patch_size) * patch_size
    else:
        oH = fix_size
        oW = round(iW * oH / iH / patch_size) * patch_size
    if (iH, iW) != (oH, oW):
        image = torch.nn.functional.interpolate(
            image.squeeze(0), size=(oH, oW), mode='bilinear').unsqueeze(0)
    return image, oH, oW


@torch.no_grad()
def viz_test(target_render, target_image, vizdir):
    assert target_render.shape == target_image.shape
    vizdir.mkdir(exist_ok=True)
    target_render = target_render.flatten(0,1).moveaxis(1,-1).float().mul(255).cpu().numpy().astype(np.uint8)
    target_image = target_image.flatten(0,1).moveaxis(1,-1).float().mul(255).cpu().numpy().astype(np.uint8)
    for i, (p, g) in enumerate(zip(target_render, target_image)):
        imageio.imwrite(vizdir / f"{i:03d}_rend.png", p)
        imageio.imwrite(vizdir / f"{i:03d}_gt.png", g)

def interpolate_poses(c2w_r, c2w_t, n_frame, periodic=False):

    assert len(c2w_r) > 1
    assert len(c2w_r) == len(c2w_t)

    c2w_r = list(c2w_r)
    c2w_t = list(c2w_t)
    bc_type = None

    if periodic:
        c2w_r.append(c2w_r[0])
        c2w_t.append(c2w_t[0])
        bc_type = "periodic"

    pos_lst = np.stack(c2w_t)
    lookat_lst = np.stack([R[:3, 2] for R in c2w_r])
    right_lst = np.stack([R[:3, 0] for R in c2w_r])

    ts = np.linspace(0, 1, len(c2w_r))
    pos_interp_f = make_interp_spline(ts, pos_lst, bc_type=bc_type)
    lookat_interp_f = make_interp_spline(ts, lookat_lst, bc_type=bc_type)
    right_interp_f = make_interp_spline(ts, right_lst, bc_type=bc_type)

    samps = np.linspace(0, 1, n_frame+1)[:n_frame]
    pos_video = pos_interp_f(samps)
    lookat_video = lookat_interp_f(samps)
    right_video = right_interp_f(samps)
    interp_poses = []
    for i in range(n_frame):
        pos = pos_video[i]
        lookat = lookat_video[i] / np.linalg.norm(lookat_video[i])
        right_ = right_video[i] / np.linalg.norm(right_video[i])
        down = np.cross(lookat, right_)
        right = np.cross(down, lookat)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = lookat
        c2w[:3, 3] = pos
        interp_poses.append(c2w)

    return np.stack(interp_poses)

@torch.no_grad()
def viz_video(renderer, target_intr, target_c2w_r, target_c2w_t, target_h, target_w, vizpath, n_frame=300):
    # Interpolate poses
    video_intr = target_intr[0,0]
    video_pose = interpolate_poses(
        target_c2w_r[0].cpu().numpy(),
        target_c2w_t[0].cpu().numpy(),
        n_frame=n_frame)
    video_c2w_r = torch.tensor(video_pose[:, :3, :3], dtype=torch.float32, device="cuda")
    video_c2w_t = torch.tensor(video_pose[:, :3, 3], dtype=torch.float32, device="cuda")

    # Render video
    video = []
    for i in range(n_frame):
        frame = renderer(
            video_intr.view(1,1,4),
            video_c2w_r[i].view(1,1,3,3),
            video_c2w_t[i].view(1,1,3),
            target_h, target_w)
        video.append(frame[0,0].moveaxis(0,-1).float().mul(255).cpu().numpy().astype(np.uint8))

    imageio.mimwrite(vizpath, video, fps=30)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_root', type=Path)
    parser.add_argument('--ckpt_name', default='latest.pt')
    parser.add_argument('--ap_ratio')
    parser.add_argument('--eval_src_res', action='store_true')
    parser.add_argument('--force_rerun', action='store_true')
    parser.add_argument('--rend_bs', default=0, type=int)
    parser.add_argument('--viz_test', action='store_true')
    parser.add_argument('--viz_video', action='store_true')
    parser.add_argument('--fix_size', type=int, default=None)
    parser.add_argument('--chunk_K', type=int, default=None)
    parser.add_argument('--chunk_boundary_ratio', type=float, default=None)
    args, cmd_lst = parser.parse_known_args()

    if global_rank == 0:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Mem: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # A Hugging Face release (config.json) stores the model options only
    if not (args.ckpt_root / 'config.yaml').exists():
        has_num_pixels = any(v.startswith('--dataset.num_pixels') for v in cmd_lst)
        has_ap_ratio = args.ap_ratio or any(v.startswith('--dataset.valid_ap_ratio') for v in cmd_lst)
        if not (has_num_pixels and has_ap_ratio):
            parser.error("released checkpoints store no dataset settings: pass --dataset.num_pixels "
                         "and --dataset.valid_ap_ratio (or --ap_ratio), see docs/eval.md")

    # Update config from the checkpoint directory and command line
    update_config_from_ckpt(args.ckpt_root, cmd_lst)

    if args.chunk_K is not None:
        cfg.model.ttb_per_cluster_K = args.chunk_K
    if args.chunk_boundary_ratio is not None:
        cfg.model.ttb_per_cluster_boundary_ratio = args.chunk_boundary_ratio
    if args.ap_ratio:
        print(f"Overwriting ap: {cfg.dataset.valid_ap_ratio} => {args.ap_ratio}")
        cfg.dataset.valid_ap_ratio = args.ap_ratio

    print(f"Working #pixels: {cfg.dataset.num_pixels}")

    # Prepare checkpoint path
    ckpt_path = resolve_checkpoint_path(args.ckpt_root, args.ckpt_name)

    # Prepare model
    model = DecoderViewSynthesisModel(**cfg.model)
    model = model.cuda()
    numel = float(sum(p.numel() for p in model.parameters()) / 1024**2)

    if global_rank == 0:
        print(f'#trainable params {numel:9.2f}M')

    # Check if we need to resume
    _, _, done_iter = load_checkpoint(model, ckpt_path)
    done_iter = 'release' if done_iter is None else str(done_iter)

    if global_rank == 0:
        print(f"resumed from: {ckpt_path} at iter={done_iter}")

    # Prepare for doing multigpu training
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Prepare data loader
    dataset_valid = PresampledTestingDataset(
        datasets_json_path=cfg.dataset.valid_path,
        num_pixels=cfg.dataset.num_pixels,
        patch_size=1 if cfg.dataset.skip_divisible_resize else cfg.model.patch_size,
        ap_ratio=cfg.dataset.valid_ap_ratio,
        cp_ratio=0.,
        keep_unscaled_image=True)

    # Valid scenes are offload to each GPU to evaluate
    sampler_valid = torch.utils.data.distributed.DistributedSampler(
        dataset_valid,
        shuffle=False)

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        num_workers=cfg.dataset.num_workers,
        sampler=sampler_valid,
        collate_fn=lambda v: v[0])

    if global_rank == 0:
        print('=' * 50)
        print('Valid set:')
        print(dataset_valid)
        print('=' * 50)

    # Prepare metrics
    outpath = args.ckpt_root / f"metrics.json"
    if outpath.exists():
        with open(outpath) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    if args.eval_src_res:
        view_setup = f"src-res ap{cfg.dataset.valid_ap_ratio}"
    else:
        view_setup = f"pix{cfg.dataset.num_pixels} ap{cfg.dataset.valid_ap_ratio}"
    if args.fix_size:
        view_setup = f"{view_setup} fix{args.fix_size}"
    # Chunk inference tag (eval-only; K=0 leaves forward bit-identical to
    # baseline so this tag is suppressed).
    chunk_K = getattr(cfg.model, 'ttb_per_cluster_K', 0)
    if chunk_K > 0:
        br = getattr(cfg.model, 'ttb_per_cluster_boundary_ratio', 0.0)
        view_setup = f"{view_setup} chunk{chunk_K} br{br:.2f}"
    for k in range(len(dataset_valid.ds_name_lst)):
        ds_name = dataset_valid.ds_name_lst[k]
        if ds_name not in metrics:
            metrics[ds_name] = {}
        if done_iter not in metrics[ds_name]:
            metrics[ds_name][done_iter] = {}
        if view_setup not in metrics[ds_name][done_iter]:
            metrics[ds_name][done_iter][view_setup] = None

    # Prepare visualization folder
    if global_rank == 0 and (args.viz_test or args.viz_video):
        for k in range(len(dataset_valid.ds_name_lst)):
            ds_name = dataset_valid.ds_name_lst[k]
            (args.ckpt_root / "viz" / view_setup / ds_name).mkdir(parents=True, exist_ok=True)

    # Prepare some training stuff
    compute_losses = LossComputer(
        lambda_percep=1.,
        lambda_lpips=1.,
        lambda_ssim=0.,
        report_skimage_ssim=True).cuda()

    # Start evaluation
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    valid_eps = time.perf_counter()

    model.eval()

    accum = {
        'n': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.int32, device="cuda"),
        'inp_psnr': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.float32, device="cuda"),
        'psnr': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.float32, device="cuda"),
        'percep': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.float32, device="cuda"),
        'lpips': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.float32, device="cuda"),
        'ssim': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.float32, device="cuda"),
    }

    for batch in tqdm.tqdm(dataloader_valid):
        assert len(batch['ds_names']) == 1, "Only support batch size of 1"
        ds_idx = dataset_valid.ds_name_idx[batch['ds_names'][0]]

        # Skip if already done and not needed to rerun
        ds_name = batch['ds_names'][0]
        if not args.force_rerun and metrics[ds_name][done_iter][view_setup] is not None:
            continue

        # Move everything to GPU
        image = batch['input_views']['image']
        intr = batch['input_views']['intr'].cuda()
        c2w_r = batch['input_views']['c2w_r'].cuda()
        c2w_t = batch['input_views']['c2w_t'].cuda()

        target_intr = batch['target_views']['intr'].cuda()
        target_c2w_r = batch['target_views']['c2w_r'].cuda()
        target_c2w_t = batch['target_views']['c2w_t'].cuda()
        target_h, target_w = batch['target_views']['image'].shape[-2:]

        if args.eval_src_res:
            target_image = batch['target_views']['src_image']
        else:
            target_image = batch['target_views']['image']

        if args.fix_size:
            image, target_h, target_w = _fix_longer_side(
                batch['input_views']['src_image'], args.fix_size, cfg.model.patch_size)

        # Eval metric
        with torch.no_grad():
            # Reconstruction
            renderer = model(image.cuda(), intr, c2w_r, c2w_t)

            # Eval testing views
            if args.rend_bs == 0:
                target_render = renderer(target_intr, target_c2w_r, target_c2w_t, target_h, target_w)
                if target_render.shape[-2:] != target_image.shape[-2:]:
                    target_render = torch.nn.functional.interpolate(
                        target_render.flatten(0,1),
                        size=target_image.shape[-2:],
                        mode='bilinear',
                        antialias=True).unflatten(0,target_render.shape[:2])
                loss_metrics = compute_losses(target_render, target_image.cuda())

            else:
                loss_metrics = {}
                target_render = []
                for i in range(0, target_intr.shape[1], args.rend_bs):
                    target_render.append(renderer(
                        target_intr[:, i:i+args.rend_bs],
                        target_c2w_r[:, i:i+args.rend_bs],
                        target_c2w_t[:, i:i+args.rend_bs],
                        target_h, target_w))
                    if target_render[-1].shape[-2:] != target_image.shape[-2:]:
                        target_render[-1] = torch.nn.functional.interpolate(
                            target_render[-1].flatten(0,1),
                            size=target_image.shape[-2:],
                            mode='bilinear',
                            antialias=True).unflatten(0,target_render[-1].shape[:2])

                    chunk_metrics = compute_losses(target_render[-1], target_image[:, i:i+args.rend_bs].cuda())
                    for metric, value in chunk_metrics.items():
                        if metric not in loss_metrics:
                            loss_metrics[metric] = 0.
                        loss_metrics[metric] += value * target_render[-1].shape[1]
                    del chunk_metrics

                target_render = torch.cat(target_render, dim=1)
                for metric in loss_metrics.keys():
                    loss_metrics[metric] /= target_render.shape[1]

            # Visualization
            if args.viz_test:
                vizdir = args.ckpt_root / "viz" / view_setup / ds_name / batch['scene_names'][0]
                viz_test(target_render, target_image, vizdir)
            if args.viz_video:
                vizpath = args.ckpt_root / "viz" / view_setup / ds_name / f"{batch['scene_names'][0]}.mp4"
                viz_video(renderer, target_intr, target_c2w_r, target_c2w_t, target_h, target_w, vizpath)

            del target_image, target_render

            # Eval input context views
            if args.rend_bs == 0:
                input_render = renderer(intr, c2w_r, c2w_t, image.shape[-2], image.shape[-1])
                input_mse = (input_render - image.cuda()).square().mean((2,3,4))
            else:
                input_mse = []
                for i in range(0, intr.shape[1], args.rend_bs):
                    input_render = renderer(
                        intr[:, i:i+args.rend_bs],
                        c2w_r[:, i:i+args.rend_bs],
                        c2w_t[:, i:i+args.rend_bs],
                        image.shape[-2], image.shape[-1])
                    input_mse.append((input_render - image[:, i:i+args.rend_bs].cuda()).square().mean((2,3,4)))
                input_mse = torch.cat(input_mse, dim=1)
            # del image, input_render

            inp_psnr = -10.0 * torch.log10(input_mse.float()).mean()
            # inp_psnr = torch.zeros_like(loss_metrics['psnr'])

            del renderer, image
            torch.cuda.empty_cache()

        # Accumulate
        accum['n'][ds_idx] += 1
        accum['inp_psnr'][ds_idx] += inp_psnr
        accum['psnr'][ds_idx] += loss_metrics['psnr']
        accum['percep'][ds_idx] += loss_metrics['percep']
        accum['lpips'][ds_idx] += loss_metrics['lpips']
        accum['ssim'][ds_idx] += loss_metrics['skimage_ssim']

    valid_eps = (time.perf_counter() - valid_eps)
    print(f"rank={global_rank}: finish eval in {valid_eps:.0f} sec", flush=True)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        for k in ['n', 'inp_psnr', 'psnr', 'percep', 'lpips', 'ssim']:
            torch.distributed.all_reduce(accum[k])

    # Log & save
    if global_rank == 0:
        inp_psnr = (accum['inp_psnr'] / accum['n']).tolist()
        psnr = (accum['psnr'] / accum['n']).tolist()
        percep = (accum['percep'] / accum['n']).tolist()
        lpips = (accum['lpips'] / accum['n']).tolist()
        ssim = (accum['ssim'] / accum['n']).tolist()
        num = accum['n']
        for k in range(len(dataset_valid.ds_name_lst)):
            if num[k] == 0:
                continue

            ds_name = dataset_valid.ds_name_lst[k]
            print(f"ds_name={ds_name}")
            print(f"    psnr      : {psnr[k]:.2f}")
            print(f"    percep    : {percep[k]:.2f}")
            print(f"    lpips     : {lpips[k]:.2f}")
            print(f"    ssim      : {ssim[k]:.2f}")
            print(f"    inp_psnr  : {inp_psnr[k]:.2f}")

            # Update metrics report
            metrics[ds_name][done_iter][view_setup] = {
                'psnr': float(psnr[k]),
                'percep': float(percep[k]),
                'lpips': float(lpips[k]),
                'ssim': float(ssim[k]),
                'inp_psnr': float(inp_psnr[k]),
                'size': float(numel),
            }

            with open(outpath, 'w') as f:
                json.dump(metrics, f, indent=2)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
