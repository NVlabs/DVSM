# Training

Please check [data.md](./data.md) for training dataset preparation.

## DL3DV

Progressive resolution training: 340p → 480p → 720p → 960p. Original training used 64 GPUs (8 nodes × 8 GPUs).

```bash
dl3dv="--ckpt.exp_dir outputs/dl3dv --dataset.ap_ratio_lst 16:9 --dataset.valid_ap_ratio 16:9 --dataset.fixed_img_per_seq true --optim.wd 5e-2 --dataset.min_img_per_seq 16"
dl3dv480p="$dl3dv --dataset.num_pixels 129600 --optim.lr 1e-5 --optim.iter 10000"
dl3dv720p="$dl3dv --dataset.num_pixels 291600 --optim.lr 1e-5 --optim.iter 10000"
dl3dv960p="$dl3dv --dataset.num_pixels 518400 --optim.lr 1e-5 --optim.iter 10000"
smallbs="--dataset.max_img_per_seq 8 --dataset.num_input_per_gpu 8 --dataset.min_img_per_seq 8"

# DVSM
torchrun --nproc-per-node=8 train.py $dl3dv     --ckpt.exp_name 340p_dvsm_p8       --model.patch_size 8
torchrun --nproc-per-node=8 train.py $dl3dv     --ckpt.exp_name 340p_dvsm_p16_dino --model.patch_size 16 --model.backbone dino
torchrun --nproc-per-node=8 train.py $dl3dv480p --ckpt.exp_name 480p_dvsm_p8       --ckpt.pretrained outputs/dl3dv/340p_dvsm_p8/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv480p --ckpt.exp_name 480p_dvsm_p16_dino --ckpt.pretrained outputs/dl3dv/340p_dvsm_p16_dino/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv720p --ckpt.exp_name 720p_dvsm_p8       --ckpt.pretrained outputs/dl3dv/480p_dvsm_p8/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv720p --ckpt.exp_name 720p_dvsm_p16_dino --ckpt.pretrained outputs/dl3dv/480p_dvsm_p16_dino/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv960p --ckpt.exp_name 960p_dvsm_p8       --ckpt.pretrained outputs/dl3dv/720p_dvsm_p8/latest.pt $smallbs
torchrun --nproc-per-node=8 train.py $dl3dv960p --ckpt.exp_name 960p_dvsm_p16_dino --ckpt.pretrained outputs/dl3dv/720p_dvsm_p16_dino/latest.pt

# TTB (initialized from DVSM 340p)
torchrun --nproc-per-node=8 train.py $dl3dv     --ckpt.exp_name 340p_ttb_dvsm_minus --model.patch_size 8 --model.block_type IMTB   --ckpt.pretrained outputs/dl3dv/340p_dvsm_p8/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv     --ckpt.exp_name 340p_ttb_dvsm       --model.patch_size 8 --model.block_type IMCMTB --ckpt.pretrained outputs/dl3dv/340p_dvsm_p8/latest.pt

torchrun --nproc-per-node=8 train.py $dl3dv480p --ckpt.exp_name 480p_ttb_dvsm_minus --ckpt.pretrained outputs/dl3dv/340p_ttb_dvsm_minus/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv480p --ckpt.exp_name 480p_ttb_dvsm       --ckpt.pretrained outputs/dl3dv/340p_ttb_dvsm/latest.pt

torchrun --nproc-per-node=8 train.py $dl3dv720p --ckpt.exp_name 720p_ttb_dvsm_minus --ckpt.pretrained outputs/dl3dv/480p_ttb_dvsm_minus/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv720p --ckpt.exp_name 720p_ttb_dvsm       --ckpt.pretrained outputs/dl3dv/480p_ttb_dvsm/latest.pt

torchrun --nproc-per-node=8 train.py $dl3dv960p --ckpt.exp_name 960p_ttb_dvsm_minus --ckpt.pretrained outputs/dl3dv/720p_ttb_dvsm_minus/latest.pt
torchrun --nproc-per-node=8 train.py $dl3dv960p --ckpt.exp_name 960p_ttb_dvsm       --ckpt.pretrained outputs/dl3dv/720p_ttb_dvsm/latest.pt --ckpt.period 100
```

## RealEstate10K

```bash
re10k="--ckpt.exp_dir outputs/re10k --dataset.train_path configs/re10k_train.json --dataset.valid_path configs/re10k_test.json --dataset.ap_ratio_lst 1:1 --dataset.valid_ap_ratio 1:1 --dataset.min_img_per_seq 2 --dataset.max_img_per_seq 2 --dataset.num_input_per_gpu 32 --dataset.num_target_per_gpu 48 --optim.wd 5e-2"
re10k512="$re10k --dataset.num_pixels 262144 --optim.lr 1e-4 --optim.iter 20000"

# DVSM
torchrun --nproc-per-node=8 train.py $re10k    --ckpt.exp_name 256_dvsm_p16_dino --model.patch_size 16 --model.backbone dino
torchrun --nproc-per-node=8 train.py $re10k    --ckpt.exp_name 256_dvsm_p8_dino  --model.patch_size 16 --model.backbone dino --model.render_up_scale 2 --model.recon_up_scale 2
torchrun --nproc-per-node=8 train.py $re10k512 --ckpt.exp_name 512_dvsm_p16_dino --ckpt.pretrained outputs/re10k/256_dvsm_p16_dino/latest.pt
torchrun --nproc-per-node=8 train.py $re10k512 --ckpt.exp_name 512_dvsm_p8_dino  --ckpt.pretrained outputs/re10k/256_dvsm_p8_dino/latest.pt

# TTB
torchrun --nproc-per-node=8 train.py $re10k    --ckpt.exp_name 256_ttb_dvsm_minus_multi1 --model.patch_size 16 --model.backbone dino --model.render_up_scale 2 --model.recon_up_scale 2 --model.block_type IMTB   --model.ttb_inter_multi 1.0
torchrun --nproc-per-node=8 train.py $re10k    --ckpt.exp_name 256_ttb_dvsm_multi1       --model.patch_size 16 --model.backbone dino --model.render_up_scale 2 --model.recon_up_scale 2 --model.block_type IMCMTB --model.ttb_inter_multi 1.0

torchrun --nproc-per-node=8 train.py $re10k512 --ckpt.exp_name 512_ttb_dvsm_minus_multi1 --ckpt.pretrained outputs/re10k/256_ttb_dvsm_minus_multi1/latest.pt --model.ttb_inter_multi 1.0
torchrun --nproc-per-node=8 train.py $re10k512 --ckpt.exp_name 512_ttb_dvsm_multi1       --ckpt.pretrained outputs/re10k/256_ttb_dvsm_multi1/latest.pt       --model.ttb_inter_multi 1.0
```

## Ablation (single node, 8 GPUs)

```bash
abla="--ckpt.exp_dir outputs/abla --dataset.ap_ratio_lst 16:9 --dataset.valid_ap_ratio 16:9 --optim.wd 5e-2 --dataset.fixed_img_per_seq true --optim.lr 1e-4 --model.patch_size 16 --dataset.min_img_per_seq 32"
abla480p="$abla --dataset.num_pixels 129600 --optim.lr 4e-6 --optim.iter 10000"
abla720p="$abla --dataset.num_pixels 291600 --optim.lr 4e-6 --optim.iter 10000"
pre="--ckpt.pretrained outputs/abla/base/latest.pt"

# Base pretrain (DVSM at 340p) — used by all architecture ablations below
torchrun --nproc-per-node=8 train.py $abla --ckpt.exp_name base

# 340p: 3 architectures from `base` pretrain
torchrun --nproc-per-node=8 train.py $abla --ckpt.exp_name dvsm           --model.block_type IMCM   $pre
torchrun --nproc-per-node=8 train.py $abla --ckpt.exp_name ttb_dvsm       --model.block_type IMCMTB $pre
torchrun --nproc-per-node=8 train.py $abla --ckpt.exp_name ttb_dvsm_minus --model.block_type IMTB   $pre

# Progressive 480p (block_type inherits from pretrained config.yaml)
torchrun --nproc-per-node=8 train.py $abla480p --ckpt.exp_name 480p_dvsm           --ckpt.pretrained outputs/abla/dvsm/latest.pt
torchrun --nproc-per-node=8 train.py $abla480p --ckpt.exp_name 480p_ttb_dvsm       --ckpt.pretrained outputs/abla/ttb_dvsm/latest.pt
torchrun --nproc-per-node=8 train.py $abla480p --ckpt.exp_name 480p_ttb_dvsm_minus --ckpt.pretrained outputs/abla/ttb_dvsm_minus/latest.pt

# Progressive 720p
torchrun --nproc-per-node=8 train.py $abla720p --ckpt.exp_name 720p_dvsm           --ckpt.pretrained outputs/abla/480p_dvsm/latest.pt
torchrun --nproc-per-node=8 train.py $abla720p --ckpt.exp_name 720p_ttb_dvsm       --ckpt.pretrained outputs/abla/480p_ttb_dvsm/latest.pt
torchrun --nproc-per-node=8 train.py $abla720p --ckpt.exp_name 720p_ttb_dvsm_minus --ckpt.pretrained outputs/abla/480p_ttb_dvsm_minus/latest.pt
```

## Fine-tuning a released checkpoint

To start from a checkpoint downloaded from Hugging Face, point `--ckpt.pretrained` at its weight file, e.g. `--ckpt.pretrained outputs/DVSM-TTB-p8-DL3DV-960p/model.safetensors`. The model options are read from the `config.json` next to it. Dataset and optimizer options are not stored in a released checkpoint and must be given on the command line.

## Multi-node training

For runs spanning multiple nodes, replace `torchrun --nproc-per-node=8` with:

```bash
torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py [args]
```

