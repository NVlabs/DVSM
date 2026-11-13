import os
import json
import glob
import h5py
import time
import tqdm
import torch
import einops
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
import imageio.v3 as iio
from pathlib import Path
import concurrent.futures
from functools import partial
import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def parse_meta(sn, path):
    item = {}
    with h5py.File(path, 'r') as f:
        N = int(f[sn].attrs['n_frames'])
        R = f[sn]['image'].shape[1:3]
    return (sn, N, R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="data/hdf5/*/*.hdf5")
    args = parser.parse_args()

    for path in glob.glob(args.glob):
        path = Path(path)
        outpath = path.with_suffix('.json')
        if outpath.exists():
            logging.info(f"Skip: {outpath} existed")
            continue

        # Process each scene
        with h5py.File(path, 'r') as f:
            scene_names = sorted(list(f.keys()))
        logging.info(f"Processing: {len(scene_names)} scenes in {path}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=min(32, len(scene_names))) as executor:
            worker_function = partial(parse_meta, path=path)
            meta = list(tqdm.tqdm(
                executor.map(worker_function, scene_names),
                total=len(scene_names),
                miniters=1,
            ))

        with open(outpath, 'w') as f:
            json.dump({
                'scene_names': [v[0] for v in meta],
                'num_frames': [v[1] for v in meta],
                'resolutions': [v[2] for v in meta],
            }, f)
        
        logging.info(f"Saved: {outpath}")
