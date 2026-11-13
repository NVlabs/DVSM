import os
import cv2
import glob
import time
import json
import h5py
import tqdm
import shutil
import tarfile
import numpy as np
from PIL import Image
import multiprocessing
from pathlib import Path
from functools import partial


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outroot', type=Path,
        default=Path('data/hdf5/dl3dv_10K'))
    args = parser.parse_args()

    # Read test split
    with open(args.outroot / 'dl3dv_benchmark.txt') as f:
        test_names = set(f.read().strip().split('\n'))

    # Parse all scenes
    all_paths = sorted(args.outroot.glob('scenes/*/*.hdf5'))

    # Composite split
    files = dict(train=[], test=[])
    for path in all_paths:
        if path.stem in test_names:
            files['test'].append(path)
        else:
            files['train'].append(path)

    print("#train:", len(files['train']))
    print("#test :", len(files['test']))

    # Composite the two split
    for name, file_lst in files.items():
        f = h5py.File(args.outroot / f'{name}.hdf5', 'w')
        for path in tqdm.tqdm(file_lst):
            fullname = path.stem
            f[fullname] = h5py.ExternalLink(path.absolute(), fullname)
        f.flush()
        f.close()
