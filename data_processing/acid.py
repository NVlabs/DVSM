import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import cv2
import glob
import time
import json
import h5py
import tqdm
import torch
import shutil
import natsort
import pycolmap
import numpy as np
from PIL import Image
from io import BytesIO
import multiprocessing
from pathlib import Path
import concurrent.futures
from functools import partial


def process_scene(scene, args, split):
    # Load all scenes in a torch file
    try:
        data_path = args.root / split / f"{scene}.torch"
        data_lst = torch.load(data_path, map_location='cpu', weights_only=False)
    except:
        print(f"Fail to load: {data_path}")
        return

    # Prepare scene dataset
    writer = h5py.File(args.outroot / split / f'{scene}.hdf5', 'w')

    # Process each scene
    for data in data_lst:
        assert len(data['timestamps']) == len(data['cameras'])
        assert len(data['timestamps']) == len(data['images'])
        N = len(data['timestamps'])

        # Gather all camera params
        fx, fy, cx_p, cy_p = data['cameras'][:, :4].T
        fname_lst = [str(t) for t in data['timestamps'].tolist()]
        fovx_lst = (2 * torch.arctan(0.5 / fx)).tolist()
        fovy_lst = (2 * torch.arctan(0.5 / fy)).tolist()
        cx_p_lst = cx_p.tolist()
        cy_p_lst = cy_p.tolist()
        w2c = data['cameras'][:, 6:].reshape(N, 3, 4).cpu().numpy()
        W, H = Image.open(BytesIO(data['images'][0].numpy().tobytes())).size

        if np.any(np.isnan(w2c)):
            print(f"{scene}/{data['key']} has invalid w2c")
            sys.exit()

        # Prepare scene writer
        fullname = str(data['key'])
        w_scene = writer.create_group(fullname)

        # Write cameras and scene meta
        w_scene.create_dataset('name', data=fname_lst, dtype=h5py.string_dtype())
        w_scene.create_dataset("w2c", data=w2c, dtype=np.float32, chunks=(1,3,4))
        w_scene.attrs['n_frames'] = N

        if np.allclose(fovx_lst, fovx_lst[0]):
            w_scene.attrs['fovx'] = fovx_lst[0]
        else:
            w_scene.create_dataset('fovx', data=fovx_lst, dtype=np.float32)

        if np.allclose(fovy_lst, fovy_lst[0]):
            w_scene.attrs['fovy'] = fovy_lst[0]
        else:
            w_scene.create_dataset('fovy', data=fovy_lst, dtype=np.float32)

        if np.allclose(cx_p_lst, cx_p_lst[0]):
            w_scene.attrs['cx_p'] = cx_p_lst[0]
        else:
            w_scene.create_dataset('cx_p', data=cx_p_lst, dtype=np.float32)

        if np.allclose(cy_p_lst, cy_p_lst[0]):
            w_scene.attrs['cy_p'] = cy_p_lst[0]
        else:
            w_scene.create_dataset('cy_p', data=cy_p_lst, dtype=np.float32)

        # Write all images
        w_image = w_scene.create_dataset("image",
            (N,H,W,3), dtype=np.uint8, chunks=(1,H,W,3), compression="gzip")

        try:
            for top, image in enumerate(data['images']):
                image = np.array(Image.open(BytesIO(image.numpy().tobytes())))
                assert image.shape[:2] == (H, W)
                w_image[top] = image
        except:
            print(f"Fail to read image: {data_path} / {fullname}")
            del writer[fullname]

    # Close writer
    writer.flush()
    writer.close()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path,
        default=Path('data/download/acid/'))
    parser.add_argument('--split_lst', nargs='+', default=['test', 'train', 'validation'])
    parser.add_argument('--outroot', type=Path,
        default=Path('data/hdf5/acid/scenes'))
    parser.add_argument('--max_workers', default=32, type=int)
    args = parser.parse_args()

    # Create the output directory
    args.outroot.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in args.split_lst:

        (args.outroot / split).mkdir(parents=True, exist_ok=True)

        # Parse scene list
        scene_lst = sorted([path.stem for path in (args.root / split).glob('*.torch')])

        # Process each scene
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.max_workers, len(scene_lst))) as executor:
            worker_function = partial(process_scene, args=args, split=split)
            list(tqdm.tqdm(
                executor.map(worker_function, scene_lst),
                total=len(scene_lst),
                miniters=1,
            ))

        # Composite split
        all_files = sorted((args.outroot / split).glob('*hdf5'))

        f = h5py.File(args.outroot.parent / f'{split}.hdf5', 'w')
        for path in tqdm.tqdm(all_files, miniters=1):
            with h5py.File(path, 'r') as target_f:
                fullname_lst = sorted(list(target_f.keys()))
            for fullname in fullname_lst:
                f[fullname] = h5py.ExternalLink(path.absolute(), fullname)
        f.flush()
        f.close()
