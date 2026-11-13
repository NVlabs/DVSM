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


def focal2fov(focal, length):
    return float(2 * np.arctan(0.5 * length / focal))


def process_scene(scene_root, args):

    fullname = scene_root.stem
    outpath = args.outroot / args.fold / f'{fullname}.hdf5'
    if args.skip_done and outpath.exists():
        return

    # Gather all frames
    with open(scene_root / 'transforms.json') as f:
        meta = json.load(f)

    frames = []
    for frame in meta['frames']:
        name = Path(frame['file_path']).stem
        c2w = np.array(frame['transform_matrix'])
        if not (scene_root / 'images_4' / f"{name}.png").exists():
            print(f"Skip due to image not found: {scene_root.stem} {frame['file_path']}")
            return
        if np.isnan(c2w).sum() > 0 or abs(np.linalg.det(c2w) - 1) > 1e-5:
            print(f"Skip due to invalid pose: {scene_root.stem} {frame['file_path']}")
            return
        frames.append({'name': name, 'c2w': c2w})

    N = len(frames)
    try:
        image = np.array(Image.open(scene_root / 'images_4' / f"{frames[0]['name']}.png"))
    except:
        print(f"Skip due to image corruption: {scene_root.stem}")
        return
    H, W = image.shape[:2]

    # Parse intrinsic
    w_ = meta['w']
    h_ = meta['h']
    fx_ = meta['fl_x']
    fy_ = meta['fl_y']
    cx_ = meta['cx']
    cy_ = meta['cy']
    distort = np.array([meta['k1'], meta['k2'], meta['p1'], meta['p2']])

    fx, fy, cx, cy = fx_/w_*W ,fy_/h_*H, cx_/w_*W, cy_/h_*H
    intr = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
    undist_intr, roi = cv2.getOptimalNewCameraMatrix(intr, distort, (W,H), 0, (W,H))

    # Get undistorted intrinsic
    image = cv2.undistort(image, intr, distort, None, undist_intr)
    H, W = image.shape[:2]
    fx, fy, cx, cy = undist_intr[0,0], undist_intr[1,1], undist_intr[0,2], undist_intr[1,2]
    fovx = focal2fov(fx, W)
    fovy = focal2fov(fy, H)
    cx_p = float(cx / W)
    cy_p = float(cy / H)

    # Prepare scene dataset
    writer = h5py.File(outpath, 'w')
    w_scene = writer.create_group(fullname)
    w_w2c = w_scene.create_dataset("w2c", (N,3,4), dtype=np.float32, chunks=(1,3,4))
    w_image = w_scene.create_dataset("image",
        (N,H,W,3), dtype=np.uint8, chunks=(1,H,W,3), compression="gzip")

    # Pack all scenes
    fname_lst = []
    for top, frame in enumerate(frames):
        fname_lst.append(frame['name'])

        # Load image
        image_path = scene_root / 'images_4' / f"{frame['name']}.png"
        try:
            image = np.array(Image.open(image_path))
        except:
            print(f"Skip due to image corruption: {scene_root.stem}")
            writer.close()
            os.remove(outpath)
            return
        image = cv2.undistort(image, intr, distort, None, undist_intr)
        if image.shape[:2] != (H, W):
            print(f"Skip due to too inconsistent image shape: {scene_root.stem}")
            writer.close()
            os.remove(outpath)
            return

        # Read camera pose
        c2w = frame['c2w']
        c2w[0:3, 1:3] *= -1
        c2w = c2w[[1,0,2,3], :]
        c2w[2,:] *= -1
        w2c = np.linalg.inv(c2w)

        # Write to dataset
        w_w2c[top] = np.copy(w2c[:3])
        w_image[top] = image

    # Write other meta data
    w_scene.create_dataset('name', data=fname_lst, dtype=h5py.string_dtype())
    w_scene.attrs['n_frames'] = len(fname_lst)
    w_scene.attrs['fovx'] = fovx
    w_scene.attrs['fovy'] = fovy
    w_scene.attrs['cx_p'] = cx_p
    w_scene.attrs['cy_p'] = cy_p

    # Close writer
    writer.flush()
    writer.close()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path,
        default=Path('data/download/dl3dv_10K/'))
    parser.add_argument('--outroot', type=Path,
        default=Path('data/hdf5/dl3dv_10K/scenes'))
    parser.add_argument('--fold', required=True)
    parser.add_argument('--max_workers', default=16, type=int)
    parser.add_argument('--skip_done', action='store_true')
    args = parser.parse_args()

    # Parse scene list
    scene_root_lst = []
    for scene_root in (args.root / args.fold).iterdir():
        if (scene_root / 'transforms.json').exists():
            scene_root_lst.append(scene_root)

    print(f"Found {len(scene_root_lst)} scenes in {args.fold}")

    # Create the output directory
    (args.outroot / args.fold).mkdir(parents=True, exist_ok=True)

    # Process each scene
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.max_workers, len(scene_root_lst))) as executor:
        worker_function = partial(process_scene, args=args)
        list(tqdm.tqdm(
            executor.map(worker_function, scene_root_lst),
            total=len(scene_root_lst),
            # miniters=max(1, int(len(scene_root_lst) / 100)),
            miniters=1,
        ))
