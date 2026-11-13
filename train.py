#############################################################
# Multi-node multi-gpu training initialization
#############################################################
import os
import sys
import time
import torch
from datetime import timedelta

process_start_time = time.time()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = '1000'

global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl', timeout=timedelta(seconds=1000))
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.recompile_limit = 128

sys.path.insert(0, os.path.abspath('.'))

#############################################################
# Import modules
#############################################################
import tqdm
import numpy as np
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from dvsm.config import cfg, update_config, update_config_from_ckpt
from dvsm.dataset import MixedTrainingDataset, PresampledTestingDataset, MixedBatchRandomSampler
from dvsm.loss import LossComputer
from dvsm.models.pose_preprocessor import PosePreprocessor
from dvsm.models.DVSM import DecoderViewSynthesisModel, load_checkpoint



def save_ckpt(model, optim, scheduler, ith_iter, path):
    path = Path(path)
    st = model.module.state_dict() if torch.distributed.is_initialized() else model.state_dict()
    torch.save({
        'state_dict': st,
        'optim_state': optim.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'iter': ith_iter,
    }, path)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_files', default=[], nargs='*')
    parser.add_argument('--force_rerun', action='store_true')
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--no_optim_resume', action='store_true')
    parser.add_argument('--job_id', type=str)
    parser.add_argument('--timeout', type=int, default=None)  # Minutes; None = no timeout (default for non-SLURM use)
    args, cmd_lst = parser.parse_known_args()

    need_to_stop = False

    if global_rank == 0:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Mem: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Update config from files and command line
    update_config(args.cfg_files, cmd_lst)

    if cfg.ckpt.pretrained:
        pretrained_path = Path(cfg.ckpt.pretrained)
        assert pretrained_path.exists()
        # config.yaml (train.py output) or config.json (Hugging Face release) next to the weights
        update_config_from_ckpt(pretrained_path.parent, cmd_lst)

    # Prepare checkpoint path
    exp_root = os.path.join(cfg.ckpt.exp_dir, cfg.ckpt.exp_name)
    latest_path = os.path.join(exp_root, 'latest.pt')
    if global_rank == 0:
        os.makedirs(exp_root, exist_ok=True)
        with open(os.path.join(exp_root, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())

    # Prepare model
    model = DecoderViewSynthesisModel(**cfg.model)
    model = model.train().cuda()

    if global_rank == 0:
        print(f'#trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000**2:9.2f}M')

    # Prepare optimizer & scheduler
    for m in model.modules():
        # Some parameters we always skip applying weight deacy
        if isinstance(m, torch.nn.LayerNorm) and m.weight is not None:
            m.weight.no_wd = True
        if isinstance(m, torch.nn.RMSNorm) and m.weight is not None:
            m.weight.no_wd = True
    for name, param in model.named_parameters():
        if name.endswith('gateproj'):
            param.no_wd = True
    if hasattr(model, 'rgb_head'):
        for param in model.rgb_head.parameters():
            param.no_wd = True

    optim = torch.optim.AdamW(
        [
            {
                'params': p,
                'lr': cfg.optim.lr * (cfg.optim.lr_scaler if getattr(p, 'use_lr_scaler', False) else 1),
                'weight_decay': 0. if getattr(p, 'no_wd', False) else cfg.optim.wd
            }
            for p in model.parameters() if p.requires_grad
        ],
        betas=(cfg.optim.beta1, cfg.optim.beta2),
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        total_steps=cfg.optim.iter,
        max_lr=[pg['lr'] for pg in optim.param_groups],
        pct_start=cfg.optim.sche_warpup_iter / cfg.optim.iter,
        anneal_strategy=cfg.optim.sche_anneal_strategy,
        div_factor=cfg.optim.sche_div_factor,
        final_div_factor=1,
    )

    # Check if we need to resume
    done_iter = 0
    if os.path.exists(latest_path) and not args.force_rerun:
        ckpt = torch.load(latest_path, weights_only=False, map_location='cpu')
        done_iter = ckpt.get('iter', done_iter)
        model.load_state_dict(ckpt['state_dict'])

        if global_rank == 0:
            print(f"resumed latest: {latest_path} at iter={done_iter}")

        if 'optim_state' in ckpt and not args.no_optim_resume:
            optim.load_state_dict(ckpt['optim_state'])
        elif global_rank == 0:
            if args.no_optim_resume and 'optim_state' in ckpt:
                print("--no_optim_resume: skipping optimizer state load (fresh m,v)")
            else:
                print("no optimizer to resume")

        if 'scheduler_state' in ckpt:
            ckpt['scheduler_state']['total_steps'] = cfg.optim.iter
            scheduler.load_state_dict(ckpt['scheduler_state'])
        elif global_rank == 0:
            print("no scheduler to resume")

        del ckpt

    elif cfg.ckpt.pretrained:
        missing_keys, unexpected_keys, pretrained_iter = load_checkpoint(model, pretrained_path, strict=False)
        if global_rank == 0:
            print(f"resumed pretrain: {pretrained_path} at iter={done_iter if pretrained_iter is None else pretrained_iter}")
        if missing_keys and global_rank == 0:
            print('missing    keys', {'.'.join(k.split('.')[:3]) for k in missing_keys})
        if unexpected_keys and global_rank == 0:
            print('unexpected keys', {'.'.join(k.split('.')[:3]) for k in unexpected_keys})

    torch.cuda.empty_cache()

    # Prepare for doing multigpu training
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Prepare tensorboard
    if global_rank == 0:
        writer = SummaryWriter(log_dir=exp_root)

    # Prepare data loader
    dataset = MixedTrainingDataset(
        datasets_json_path=cfg.dataset.train_path,
        randomize_static_time=cfg.dataset.randomize_static_time)

    batch_sampler = MixedBatchRandomSampler(
        mixed_dataset=dataset,
        min_img_per_seq=cfg.dataset.min_img_per_seq,
        max_img_per_seq=cfg.dataset.max_img_per_seq,
        num_target_per_gpu=cfg.dataset.num_target_per_gpu,
        num_input_per_gpu=cfg.dataset.num_input_per_gpu,
        num_pixels=cfg.dataset.num_pixels,
        patch_size=1 if cfg.dataset.skip_divisible_resize else cfg.model.patch_size,
        divisible_only=cfg.dataset.divisible_only,
        fixed_img_per_seq=cfg.dataset.fixed_img_per_seq,
        remain_iter=cfg.optim.iter - done_iter,
        ap_ratio_lst=cfg.dataset.ap_ratio_lst,
        crop_pad_ratio=cfg.dataset.crop_pad_ratio)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.dataset.num_workers,
        collate_fn=batch_sampler.batchify_sequences)

    dataset_valid = PresampledTestingDataset(
        datasets_json_path=cfg.dataset.valid_path,
        num_pixels=cfg.dataset.num_pixels,
        patch_size=1 if cfg.dataset.skip_divisible_resize else cfg.model.patch_size,
        ap_ratio=cfg.dataset.valid_ap_ratio,
        cp_ratio=cfg.dataset.valid_crop_pad_ratio)

    # Valid scenes are offload to each GPU to evaluate
    sampler_valid = torch.utils.data.distributed.DistributedSampler(
        dataset_valid,
        shuffle=False)

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        sampler=sampler_valid,
        num_workers=4,
        collate_fn=lambda v: v[0])

    if global_rank == 0:
        print('=' * 50)
        print('Train set:')
        print(dataset)
        print('Valid set:')
        print(dataset_valid)
        print('=' * 50)

    # Prepare some training stuff
    compute_losses = LossComputer(
        lambda_percep=cfg.optim.lambda_percep,
        lambda_ssim=cfg.optim.lambda_ssim,
        lambda_lpips=cfg.optim.lambda_lpips).cuda()
    eps = time.perf_counter()

    # Start training
    for ith_iter, batch in enumerate(tqdm.tqdm(dataloader), done_iter + 1):

        # Move everything to GPU
        image = batch['input_views']['image'].cuda()
        intr = batch['input_views']['intr'].cuda()
        c2w_r = batch['input_views']['c2w_r'].cuda()
        c2w_t = batch['input_views']['c2w_t'].cuda()

        target_image = batch['target_views']['image'].cuda()
        target_intr = batch['target_views']['intr'].cuda()
        target_c2w_r = batch['target_views']['c2w_r'].cuda()
        target_c2w_t = batch['target_views']['c2w_t'].cuda()
        target_h, target_w = target_image.shape[-2:]

        # We do pose normalization outside the model inference pipeline for potential augmentation
        pose_preprocessor = PosePreprocessor(
            c2w_r, c2w_t, pose_avg_strategy=cfg.model.pose_avg_strategy)
        c2w_r, c2w_t = pose_preprocessor(c2w_r, c2w_t)
        target_c2w_r, target_c2w_t = pose_preprocessor(target_c2w_r, target_c2w_t)

        # Feed forward
        renderer = model(image, intr, c2w_r, c2w_t, pose_preproc_done=True)
        target_render = renderer(target_intr, target_c2w_r, target_c2w_t, target_h, target_w)

        # Compute rendering losses
        loss_metrics = compute_losses(target_render, target_image)

        # Backward
        loss_metrics['loss'].backward()

        # Trainer step
        global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).item()
        grad_norm_lst = []
        for param in model.parameters():
            if param.grad is not None:
                param.grad.nan_to_num_(nan=0, posinf=0, neginf=0)
                grad_norm = torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
                grad_norm_lst.append(grad_norm.item())
        optim.step()
        optim.zero_grad(set_to_none=True)
        scheduler.step()

        # Log training
        if global_rank == 0:
            writer.add_scalar('train/psnr', loss_metrics['psnr'].item(), global_step=ith_iter)
            writer.add_scalar('train/percep', loss_metrics['percep'].item(), global_step=ith_iter)
            writer.add_scalar('grad/global', global_grad_norm, global_step=ith_iter)
            writer.add_scalar('grad/mean', np.mean(grad_norm_lst), global_step=ith_iter)
            writer.add_scalar('grad/max', np.max(grad_norm_lst), global_step=ith_iter)

        # Checkpointing
        if ith_iter % cfg.ckpt.period == 0 and args.no_eval:
            if global_rank == 0:
                save_ckpt(model, optim, scheduler, ith_iter, latest_path)
                print(f'rank={global_rank}: save to {latest_path} at iter={ith_iter}')
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        elif ith_iter % cfg.ckpt.period == 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            valid_eps = time.perf_counter()

            # Show training stat
            eps = (time.perf_counter() - eps) / 60
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            unq_lr = np.unique([scheduler.get_last_lr() if scheduler else cfg.optim.lr]).tolist()
            print(f"rank={global_rank} iter={ith_iter} lr={unq_lr} peakmem={peak_mem:.2f} eps={eps:.0f}")

            # Run evaluation on valid set
            model.eval()

            accum = {
                'n': torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.int32, device="cuda"),
            }
            for k in ['inp_psnr', 'psnr', 'percep', 'cam_abs_t', 'cam_abs_rot', 'cam_aff_t', 'cam_aff_rot', 'cam_focal', 'cam_cxcy']:
                accum[k] = torch.zeros([len(dataset_valid.ds_name_lst)], dtype=torch.float32, device="cuda")

            for batch in dataloader_valid:
                assert len(batch['ds_names']) == 1, "Only support batch size of 1"
                ds_idx = dataset_valid.ds_name_idx[batch['ds_names'][0]]

                # Move everything to GPU
                image = batch['input_views']['image'].cuda()
                intr = batch['input_views']['intr'].cuda()
                c2w_r = batch['input_views']['c2w_r'].cuda()
                c2w_t = batch['input_views']['c2w_t'].cuda()

                target_image = batch['target_views']['image'].cuda()
                target_intr = batch['target_views']['intr'].cuda()
                target_c2w_r = batch['target_views']['c2w_r'].cuda()
                target_c2w_t = batch['target_views']['c2w_t'].cuda()
                target_h, target_w = target_image.shape[-2:]

                # Eval metric
                with torch.no_grad():
                    # Known input cameras
                    renderer = model(image, intr, c2w_r, c2w_t)
                    target_render = renderer(target_intr, target_c2w_r, target_c2w_t, target_h, target_w)

                    # Compute rendering losses
                    loss_metrics = compute_losses(target_render, target_image)

                    input_render = renderer(intr, c2w_r, c2w_t, image.shape[-2], image.shape[-1])
                    inp_mse = (input_render - image).square().mean((2,3,4))
                    inp_psnr = -10.0 * torch.log10(inp_mse.float()).mean()

                # Accumulate
                accum['n'][ds_idx] += 1
                accum['inp_psnr'][ds_idx] += inp_psnr
                accum['psnr'][ds_idx] += loss_metrics['psnr']
                accum['percep'][ds_idx] += loss_metrics['percep']

            valid_eps = (time.perf_counter() - valid_eps)
            print(f"rank={global_rank}: finish eval in {valid_eps:.0f} sec", flush=True)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.all_reduce(accum['n'])
                torch.distributed.all_reduce(accum['inp_psnr'])
                torch.distributed.all_reduce(accum['psnr'])
                torch.distributed.all_reduce(accum['percep'])

            # Log & save
            if global_rank == 0:
                # Log validation metric
                inp_psnr = (accum['inp_psnr'] / accum['n']).tolist()
                psnr = (accum['psnr'] / accum['n']).tolist()
                percep = (accum['percep'] / accum['n']).tolist()
                for k in range(len(dataset_valid.ds_name_lst)):
                    ds_name = dataset_valid.ds_name_lst[k]
                    writer.add_scalar(f'valid/{ds_name}/inp_psnr', inp_psnr[k], global_step=ith_iter)
                    writer.add_scalar(f'valid/{ds_name}/psnr', psnr[k], global_step=ith_iter)
                    writer.add_scalar(f'valid/{ds_name}/percep', percep[k], global_step=ith_iter)
                    print(f"{ds_name:20s}: psnr={psnr[k]:.2f} / percep={percep[k]:.3f} / inp_psnr={inp_psnr[k]:.2f}")
                writer.add_scalar(f'spec/eps', eps, global_step=ith_iter)
                writer.add_scalar(f'spec/peak_mem', peak_mem, global_step=ith_iter)
                writer.flush()

                # Save model
                save_ckpt(model, optim, scheduler, ith_iter, latest_path)
                print(f'rank={global_rank}: save to {latest_path} at iter={ith_iter}')

            # Back to train mode
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            model.train()
            eps = time.perf_counter()

        # Periodically check if we are requested to stop
        if ith_iter % 10 == 0:
            early_timeout = args.timeout is not None and (time.time() - process_start_time) / 60 > args.timeout
            if early_timeout:
                print(f'rank={global_rank}: need to stop', flush=True)
                need_to_stop = True

                # Mark this job as needing resuming
                if global_rank == 0 and args.job_id:
                    if not os.path.exists("slurm_outputs/wip_jobs"):
                        os.makedirs("slurm_outputs/wip_jobs", exist_ok=True)
                    with open(f"slurm_outputs/wip_jobs/{args.job_id}", "w") as f:
                        f.write("\n")

            # Check if any process need to stop
            any_need_to_stop = torch.tensor([need_to_stop], dtype=torch.float32, device="cuda")
            torch.distributed.all_reduce(any_need_to_stop)
            any_need_to_stop = (any_need_to_stop.item() > 0)

            if any_need_to_stop:
                need_to_stop = True
                if global_rank == 0:
                    save_ckpt(model, optim, scheduler, ith_iter, latest_path)
                    print(f'rank={global_rank}: save to {latest_path} at iter={ith_iter}')
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                print(f"rank={global_rank}: exit training loop")
                break

    # Check if the job is all done
    if not need_to_stop and args.job_id and global_rank == 0:
        os.makedirs("slurm_outputs/done_jobs", exist_ok=True)
        with open(f"slurm_outputs/done_jobs/{args.job_id}", "w") as f:
            f.write("\n")
        if os.path.exists(f"slurm_outputs/wip_jobs/{args.job_id}"):
            os.remove(f"slurm_outputs/wip_jobs/{args.job_id}")
            print(f"Trining completed: job_id={args.job_id}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
