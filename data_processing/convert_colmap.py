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


def process_scene(scene, args):
    # Unpack scene names
    fullname = scene
    scene_path = args.src_root / scene

    # Gather all frames
    sparse_path = scene_path / args.sfm_path
    sfm = pycolmap.Reconstruction(sparse_path)
    keys = natsort.natsorted(
        sfm.images.keys(),
        key = lambda k : sfm.images[k].name)
    
    keys = [key for key in keys if (scene_path / args.img_path / sfm.images[key].name).exists()]

    N = len(keys)
    W, H = Image.open(scene_path / args.img_path / sfm.images[keys[0]].name).size
    if args.down > 1:
        W = int(W / args.down)
        H = int(H / args.down)

    # Prepare scene dataset
    writer = h5py.File(args.dst_root / 'scenes' / f'{fullname}.hdf5', 'w')
    w_scene = writer.create_group(fullname)
    w_w2c = w_scene.create_dataset("w2c", (N,3,4), dtype=np.float32, chunks=(1,3,4))
    w_image = w_scene.create_dataset("image",
        (N,H,W,3), dtype=np.uint8, chunks=(1,H,W,3), compression="gzip")

    # Pack all scenes
    fname_lst = []
    fovx_lst = []
    fovy_lst = []
    cx_p_lst = []
    cy_p_lst = []
    for top, key in enumerate(keys):
        frame = sfm.images[key]
        fname_lst.append(Path(frame.name).stem)

        # Load image
        image_path = scene_path / args.img_path / frame.name
        image = Image.open(image_path)
        if args.down > 1:
            image = image.resize((int(image.size[0]/args.down), int(image.size[1]/args.down)))
        image = np.array(image)
        assert image.shape[:2] == (H, W)

        # Read camera matrix
        if frame.camera.model.name == "OPENCV":
            focal_x, focal_y, cx, cy = frame.camera.params[:4]  # TODO: undistort
            fovx = focal2fov(focal_x, frame.camera.width)
            fovy = focal2fov(focal_y, frame.camera.height)
            cx_p = float(cx / frame.camera.width)
            cy_p = float(cy / frame.camera.height)
        elif frame.camera.model.name == "PINHOLE":
            focal_x, focal_y, cx, cy = frame.camera.params  # This is at the original resolution
            fovx = focal2fov(focal_x, frame.camera.width)
            fovy = focal2fov(focal_y, frame.camera.height)
            cx_p = float(cx / frame.camera.width)
            cy_p = float(cy / frame.camera.height)
        elif frame.camera.model.name == "SIMPLE_PINHOLE":
            focal_x, cx, cy = frame.camera.params
            fovx = focal2fov(focal_x, frame.camera.width)
            fovy = focal2fov(focal_x, frame.camera.height)
            cx_p = float(cx / frame.camera.width)
            cy_p = float(cy / frame.camera.height)
        else:
            raise NotImplementedError(f"Unknown camera type: {frame.camera.model.name}")

        w2c = frame.cam_from_world().matrix()

        fovx_lst.append(fovx)
        fovy_lst.append(fovy)
        cx_p_lst.append(cx_p)
        cy_p_lst.append(cy_p)

        # Write to dataset
        w_w2c[top] = w2c
        w_image[top] = image

    # Write other meta data
    w_scene.create_dataset('name', data=fname_lst, dtype=h5py.string_dtype())
    w_scene.attrs['n_frames'] = len(fname_lst)

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

    # Close writer
    writer.flush()
    writer.close()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src_root', type=Path)
    parser.add_argument('dst_root', type=Path)
    parser.add_argument('--sfm_path', default='sparse/0')
    parser.add_argument('--img_path', default='images')
    parser.add_argument('--down', default=1, type=float)
    parser.add_argument('--max_workers', default=32, type=int)
    args = parser.parse_args()

    # Create the output directory
    (args.dst_root / 'scenes').mkdir(parents=True, exist_ok=True)

    # Parse scene list
    scene_lst = [path.stem for path in args.src_root.glob('*') if (path / args.sfm_path).exists()]
    if len(scene_lst) == 0:
        print(f"No scenes found under {args.src_root}")
        import sys; sys.exit()
    print(f"Found {len(scene_lst)} scenes")

    # Process each scene
    # for scene in tqdm.tqdm(scene_lst):
    #     process_scene(scene, args)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.max_workers, len(scene_lst))) as executor:
        worker_function = partial(process_scene, args=args)
        list(tqdm.tqdm(
            executor.map(worker_function, scene_lst),
            total=len(scene_lst),
            miniters=1,
        ))

    # # Composite split
    with h5py.File(args.dst_root / f'all.hdf5', 'w') as f:
        for path in sorted((args.dst_root / 'scenes').glob('*hdf5')):
            fullname = path.stem
            f[fullname] = h5py.ExternalLink(path, fullname)
