import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import cv2
import glob
import time
import json
import h5py
import tqdm
import shutil
import natsort
import pycolmap
import numpy as np
from PIL import Image
import multiprocessing
from pathlib import Path
import concurrent.futures
from functools import partial

from sklearn.cluster import KMeans

def kmean_sampling(w2c, n_clusters):
    position = -(np.moveaxis(w2c[:, :3, :3], 2, 1) @ w2c[:, :3, [3]]).squeeze(2)
    lookat = np.moveaxis(w2c[:, :3, :3], 2, 1)[:, :3, 2]
    X = np.concatenate([
        position / np.median(np.linalg.norm(position - position.mean(0), axis=1)),
        lookat
    ], axis=1)
    cluster_centers = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X).cluster_centers_
    sel = np.linalg.norm(cluster_centers[:, None] - X, axis=2).argmin(1)
    sel = np.unique(sel)
    return sel


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('fold', type=int)
    parser.add_argument('--outroot', type=Path, default=Path('configs/test_views_sampling/'))
    parser.add_argument('--num_inputs', default=[16, 32, 64, 128, 256, -1], type=int, nargs='+')
    args = parser.parse_args()

    print(f"Dataset     : {args.fold}")
    print(f"Heldout fold: {args.fold}")
    print(f"#inputs     : {args.num_inputs}")

    outputs = {k: {} for k in args.num_inputs}

    with h5py.File(args.dataset, 'r') as f:
        for name in tqdm.tqdm(f.keys()):
            scene = f[name]
            n_frames = scene.attrs['n_frames']
            w2c = f[name]['w2c'][:]

            heldout = [i for i in range(n_frames) if i % args.fold == 0]
            candidate = [i for i in range(n_frames) if i % args.fold != 0]

            for num in args.num_inputs:
                if len(candidate) <= num or num == -1:
                    # Select all frames
                    outputs[num][name] = {
                        'input': candidate,
                        'heldout': heldout,
                    }
                else:
                    samp = kmean_sampling(np.stack([w2c[i] for i in candidate]), n_clusters=num)
                    outputs[num][name] = {
                        'input': [candidate[i] for i in samp],
                        'heldout': heldout,
                    }

    for num in args.num_inputs:
        if num == -1:
            outpath = args.outroot / f"{args.dataset.parent.stem}_fold{args.fold}_all.json"
        else:
            outpath = args.outroot / f"{args.dataset.parent.stem}_fold{args.fold}_kmeans{num}.json"

        if outpath.exists():
            print(f"Skip: {outpath} existed")
        else:
            print(f"Save to: {outpath}")
            with open(outpath, 'w') as f:
                json.dump(outputs[num], f, indent=2)
