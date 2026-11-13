import os
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


def merge_subset(args, total):
    ith, subset_root = args
    paths = sorted(list(subset_root.glob("*hdf5")))
    outpath = subset_root.with_suffix('.hdf5')
    if len(paths) == 0:
        logging.info(f"Skip. Find no hdf5 file in {subset_root}")
        return

    with h5py.File(outpath, 'w') as writer:
        for path in paths:
            with h5py.File(path, 'r') as reader:
                for group_name in reader.keys():
                    reader.copy(group_name, writer)
        writer.flush()

    shutil.rmtree(subset_root)

    logging.info(f"Done: {outpath} ({ith:4d}/{total:4d})")


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Forming data split of SpatialVID."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/hdf5/spatialvid_hq"),
    )
    parser.add_argument("--n_valid", default=300, type=int)
    args = parser.parse_args()

    max_workers = 32

    # Find each subset to process
    subset_lst = sorted(path for path in (args.root / "scenes").glob('*/*') if path.is_dir())
    logging.info(f"Found {len(subset_lst)} subset to merge")

    # Merge seperate files into one for each subset
    if len(subset_lst) == 0:
        logging.info(f"Skip. All subset seems to be merged already.")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(max_workers, len(subset_lst))) as executor:
            worker_function = partial(merge_subset, total=len(subset_lst))
            list(executor.map(worker_function, enumerate(subset_lst)))

    # Produce all in one split
    all_path = args.root / 'all.hdf5'
    if all_path.exists():
        logging.info(f"Skip. {all_path} already existed.")
    else:
        with h5py.File(all_path, 'w') as writer:
            for path in tqdm.tqdm(list((args.root / "scenes").glob('*/*hdf5'))):
                with h5py.File(path, 'r') as reader:
                    for scene_name in reader.keys():
                        assert scene_name not in writer
                        writer[scene_name] = h5py.ExternalLink(path.absolute(), scene_name)

    # Produce train/test split
    with h5py.File(all_path, 'r') as reader:
        key2path = {k: item.file.filename for k, item in reader.items()}
    logging.info(f"Found {len(key2path)} videos.")

    # Parse meta data
    metadata = pd.read_csv(args.root / "SpatialVID_HQ_metadata.csv")
    ori_n_rows = len(metadata)

    check = lambda row: (f"group_{int(row['group id']):04d}_{row['id']}" in key2path)
    metadata = metadata[metadata.apply(check, axis=1)]
    logging.info(f"Found {len(metadata)} meta data (removed {ori_n_rows - len(metadata)} rows).")

    meta_dynamic = metadata.loc[metadata['dynamicRatio'] > metadata['dynamicRatio'].mean() + metadata['dynamicRatio'].std()]
    meta_aesthetic = metadata.loc[metadata['aesthetic score'] > metadata['aesthetic score'].mean() + metadata['aesthetic score'].std()]
    logging.info(f"Found {len(meta_dynamic)} rows with higher dynamic.")
    logging.info(f"Found {len(meta_aesthetic)} rows with higher aesthetic.")

    # print(metadata[['dynamicRatio', 'aesthetic score']].describe())
    # print(meta_dynamic[['dynamicRatio', 'aesthetic score']].describe())
    # print(meta_aesthetic[['dynamicRatio', 'aesthetic score']].describe())

    # Construct training and testing splits
    test_keys = {
        f"group_{int(row['group id']):04d}_{row['id']}"
        for _, row in meta_dynamic.iloc[np.round(np.linspace(0, len(meta_dynamic)-1, args.n_valid)).astype(int)].iterrows()
    }

    with h5py.File(args.root / f'train.hdf5', 'w') as f_tr, h5py.File(args.root / f'test.hdf5', 'w') as f_te:
        for ith, row in metadata.iterrows():
            key = f"group_{int(row['group id']):04d}_{row['id']}"
            if key in test_keys:
                f_te[key] = h5py.ExternalLink(key2path[key], key)
            else:
                f_tr[key] = h5py.ExternalLink(key2path[key], key)

    # Construct higher quality split
    with h5py.File(args.root / f'train_higher_dynamic.hdf5', 'w') as f_tr:
        for ith, row in meta_dynamic.iterrows():
            key = f"group_{int(row['group id']):04d}_{row['id']}"
            if key not in test_keys:
                f_tr[key] = h5py.ExternalLink(key2path[key], key)

    with h5py.File(args.root / f'train_higher_aesthetic.hdf5', 'w') as f_tr:
        for ith, row in meta_aesthetic.iterrows():
            key = f"group_{int(row['group id']):04d}_{row['id']}"
            if key not in test_keys:
                f_tr[key] = h5py.ExternalLink(key2path[key], key)
