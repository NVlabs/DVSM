# Evaluation

Please check [data.md](./data.md) for testing dataset preparation.

## Benchmark Evaluation

Distributed evaluation computing PSNR, SSIM, LPIPS per scene.

```bash
# Single node, one process per GPU (use --nproc-per-node=1 for a single GPU)
common="--standalone --nproc-per-node=8"

# Multiple nodes on SLURM (MASTER_ADDR: hostname of the first node, MASTER_PORT: a free port)
# common="--nnodes=${SLURM_NNODES} --nproc-per-node=${SLURM_GPUS_PER_NODE} --rdzv-id=${SLURM_JOB_ID} --rdzv-backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
```

The checkpoint directory passed to `eval.py` can be a `train.py` output (as in the commands below) or a released checkpoint downloaded from Hugging Face, e.g. `outputs/DVSM-TTB-p8-DL3DV-960p`. A released checkpoint only stores the model options, so pass the evaluation resolution explicitly:

| Released checkpoints | Extra flags |
|---|---|
| DL3DV 480p | `--dataset.num_pixels 129600 --dataset.valid_ap_ratio 16:9` |
| DL3DV 720p | `--dataset.num_pixels 291600 --dataset.valid_ap_ratio 16:9` |
| DL3DV 960p | `--dataset.num_pixels 518400 --dataset.valid_ap_ratio 16:9` |
| RealEstate10K 512p | `--dataset.num_pixels 262144 --dataset.valid_ap_ratio 1:1` |

Key flags:
- `--rend_bs 1` — always use 1 for evaluation
- `--viz_test` — save rendered images
- `--viz_video` — save interpolated video
- `--ap_ratio 0:1 --eval_src_res` — zero-shot evaluation
- `--chunk_K N` — local-duplicate fast weights with N clusters (TTB only)

### DVSM

**RealEstate10K:**
```bash
torchrun $common eval.py outputs/re10k/512_dvsm_p8_dino  --dataset.valid_path configs/re10k_test.json --rend_bs 1
torchrun $common eval.py outputs/re10k/512_dvsm_p16_dino --dataset.valid_path configs/re10k_test.json --rend_bs 1
```

**DL3DV:**
```bash
torchrun $common eval.py outputs/dl3dv/960p_dvsm_p8 --dataset.valid_path configs/dl3dv_test.json --rend_bs 1
```

**DL3DV → zero-shot generalization:**
```bash
torchrun $common eval.py outputs/dl3dv/960p_dvsm_p8 --dataset.valid_path configs/zeroshot.json --rend_bs 1 --ap_ratio 0:1 --eval_src_res
```

### TTB-DVSM

**RealEstate10K:**
```bash
torchrun $common eval.py outputs/re10k/512_ttb_dvsm_multi1       --dataset.valid_path configs/re10k_test.json --rend_bs 1
torchrun $common eval.py outputs/re10k/512_ttb_dvsm_minus_multi1 --dataset.valid_path configs/re10k_test.json --rend_bs 1
```

**DL3DV:**
```bash
torchrun $common eval.py outputs/dl3dv/960p_ttb_dvsm       --dataset.valid_path configs/dl3dv_test.json --rend_bs 1
torchrun $common eval.py outputs/dl3dv/960p_ttb_dvsm_minus --dataset.valid_path configs/dl3dv_test.json --rend_bs 1
```

**DL3DV + local-duplicate fast weights:**
```bash
torchrun $common eval.py outputs/dl3dv/960p_ttb_dvsm --dataset.valid_path configs/dl3dv_test.json --rend_bs 1 --chunk_K 4
```

**DL3DV → zero-shot generalization:**
```bash
torchrun $common eval.py outputs/dl3dv/960p_ttb_dvsm       --dataset.valid_path configs/zeroshot.json --rend_bs 1 --ap_ratio 0:1 --eval_src_res
torchrun $common eval.py outputs/dl3dv/960p_ttb_dvsm_minus --dataset.valid_path configs/zeroshot.json --rend_bs 1 --ap_ratio 0:1 --eval_src_res
```

**DL3DV → zero-shot + local-duplicate:**
```bash
torchrun $common eval.py outputs/dl3dv/960p_ttb_dvsm --dataset.valid_path configs/zeroshot.json --rend_bs 1 --ap_ratio 0:1 --eval_src_res --chunk_K 4
```

## Measure Runtime

Dry-run random inputs to benchmark efficiency (single GPU, no distributed setup required).

Positional args: `<V views> <H> <W> <save_name>`. Append any `--model.*` flags to select architecture.

**Inference (reconstruction + single novel-view render):**
```bash
# DVSM
python eval_runtime_infer.py 2  256 256 dvsm_p8       --model.patch_size 8
python eval_runtime_infer.py 2  256 256 dvsm_p16_dino --model.patch_size 16 --model.backbone dino
python eval_runtime_infer.py 32 544 960 dvsm_p8       --model.patch_size 8
python eval_runtime_infer.py 32 544 960 dvsm_p16_dino --model.patch_size 16 --model.backbone dino

# TTB-DVSM
python eval_runtime_infer.py 2  256 256 ttb_dvsm_p8       --model.patch_size 8 --model.block_type IMCMTB
python eval_runtime_infer.py 32 544 960 ttb_dvsm_p8       --model.patch_size 8 --model.block_type IMCMTB

# TTB-DVSM-minus
python eval_runtime_infer.py 2  256 256 ttb_dvsm_minus_p8 --model.patch_size 8 --model.block_type IMTB
python eval_runtime_infer.py 32 544 960 ttb_dvsm_minus_p8 --model.patch_size 8 --model.block_type IMTB
```

**Training throughput:**
```bash
# DVSM
python eval_runtime_train.py 2 256 256 dvsm_p8       --model.patch_size 8
python eval_runtime_train.py 2 256 256 dvsm_p16_dino --model.patch_size 16 --model.backbone dino
python eval_runtime_train.py 8 544 960 dvsm_p8       --model.patch_size 8
python eval_runtime_train.py 8 544 960 dvsm_p16_dino --model.patch_size 16 --model.backbone dino

# TTB-DVSM
python eval_runtime_train.py 2 256 256 ttb_dvsm_p8       --model.patch_size 8 --model.block_type IMCMTB
python eval_runtime_train.py 8 544 960 ttb_dvsm_p8       --model.patch_size 8 --model.block_type IMCMTB

# TTB-DVSM-minus
python eval_runtime_train.py 2 256 256 ttb_dvsm_minus_p8 --model.patch_size 8 --model.block_type IMTB
python eval_runtime_train.py 8 544 960 ttb_dvsm_minus_p8 --model.patch_size 8 --model.block_type IMTB
```
