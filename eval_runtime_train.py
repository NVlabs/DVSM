import os
import sys
import torch

PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import numpy as np
from pathlib import Path

from dvsm.config import cfg, update_config
from dvsm.models.DVSM import DecoderViewSynthesisModel


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('v', type=int)
    parser.add_argument('h', type=int)
    parser.add_argument('w', type=int)
    parser.add_argument('name')
    parser.add_argument('--nv', default=32, type=int)
    parser.add_argument('--ntimes', default=3, type=int)
    args, cmd_lst = parser.parse_known_args()

    # Update config from files and command line
    update_config([], cmd_lst)
    cfg.model.use_compile = False

    # Prepare model
    model = DecoderViewSynthesisModel(**cfg.model)
    model = model.cuda()
    numel = sum(p.numel() for p in model.parameters()) / 1024**2

    optim = torch.optim.AdamW(model.parameters(), lr=1e-10)

    # Start evaluation
    model.train()

    # Move everything to GPU
    assert args.nv % args.v == 0
    BS = args.nv // args.v
    image = torch.rand([BS, args.v, 3, args.h, args.w], dtype=torch.float32, device="cuda")
    intr = torch.tensor([[[1, 1, 0.5, 0.5]] * args.v] * BS, dtype=torch.float32, device="cuda")
    c2w_r = torch.tensor([[[[1,0,0],[0,1,0],[0,0,1]]] * args.v] * BS, dtype=torch.float32, device="cuda")
    c2w_t = torch.rand([BS, args.v, 3], dtype=torch.float32, device="cuda")

    # Clean statistic
    torch.cuda.empty_cache()
    torch.cuda.memory.reset_peak_memory_stats()

    # Warmup
    renderer = model(image, intr, c2w_r, c2w_t)
    target_render = renderer(intr, c2w_r, c2w_t, args.h, args.w)
    (target_render - image).square().mean().backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
    del renderer

    # Measure runtime
    ts_per_it = []
    for k in range(args.ntimes):
        torch.cuda.synchronize(); eps = time.perf_counter()

        renderer = model(image, intr, c2w_r, c2w_t)
        target_render = renderer(intr, c2w_r, c2w_t, args.h, args.w)
        (target_render - image).square().sum().backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        torch.cuda.synchronize(); eps = time.perf_counter() - eps
        ts_per_it.append(eps)

        del renderer

    outdir = Path(f"runtime_report/train/{args.v}_{args.h}_{args.w}")
    outdir.mkdir(parents=True, exist_ok=True)
    spec = {
        'size': float(numel),
        'ts_per_it': float(np.median(ts_per_it)),
        'peak_mem': float(torch.cuda.max_memory_allocated() / 1024**3),
    }
    for metric, val in spec.items():
        print(f"{metric:20s}: {val}")
    with open(outdir / f"{args.name}.json", "w") as f:
        json.dump(spec, f, indent=4)
    print(f'saved to {outdir / f"{args.name}.json"}')
