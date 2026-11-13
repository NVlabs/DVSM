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


def focal2fov(focal, length):
    return 2 * np.arctan(0.5 * length / focal)


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    Args:
        quaternions: Quaternion tensor [..., 4] (order: i, j, k, r)
        eps: Small value for numerical stability

    Returns:
        Rotation matrices [..., 3, 3]
    """
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def pose_from_quaternion(pose):
    """
    Convert pose from quaternion representation to transformation matrix.

    Args:
        pose: Pose tensor [..., 7] where first 3 elements are translation (t)
              and last 4 elements are quaternion rotation (r)

    Returns:
        w2c_matrix: World-to-camera transformation matrices [..., 3, 4]
    """
    # Input is w2c, pose(n,7) or (n,v,7), output is (N,3,4) w2c matrix
    # Tensor format from https://github.com/pointrix-project/Geomotion/blob/6ab0c364f1b44ab4ea190085dbf068f62b42727c/geomotion/model/cameras.py#L6
    if type(pose) == np.ndarray:
        pose = torch.tensor(pose)
    if len(pose.shape) == 1:
        pose = pose[None]
    quat_t = pose[..., :3]  # Translation
    quat_r = pose[..., 3:]  # Quaternion rotation
    w2c_matrix = torch.zeros((*list(pose.shape)[:-1], 3, 4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)
    return w2c_matrix


def process_video(todo, inroot, outroot, gid):
    sid, vid = todo
    outroot = outroot / f"group_{gid:04d}" / f"{sid:02d}"
    outroot.mkdir(parents=True, exist_ok=True)
    outpath = outroot / f"{vid}.hdf5"

    intrinsics = np.load(inroot / f"annotations/group_{gid:04d}/{vid}/intrinsics.npy")
    poses = np.load(inroot / f"annotations/group_{gid:04d}/{vid}/poses.npy")
    if len(intrinsics) != len(poses):
        logging.info(f"Skip inconsistent poses.npy: group_{gid:04d}/{vid} {len(intrinsics)} != {len(poses)}.")
        return

    indices = []
    with open(inroot / f"annotations/group_{gid:04d}/{vid}/indexes.txt") as f:
        for line in f:
            if line.startswith('#'):
                continue
            indices.append(int(line.strip().split()[1]))
    indices = set(indices)
    if len(indices) != len(intrinsics):
        logging.info(f"Skip inconsistent indexes.txt: group_{gid:04d}/{vid} {len(indices)} != {len(intrinsics)}.")
        return

    # Process annotation
    fovx_lst = focal2fov(intrinsics[:, 0], 1)
    fovy_lst = focal2fov(intrinsics[:, 1], 1)
    cx_p_lst = intrinsics[:, 2]
    cy_p_lst = intrinsics[:, 3]
    w2c_lst = pose_from_quaternion(poses).numpy()
    assert not np.any(np.isnan(fovx_lst))
    assert not np.any(np.isnan(fovy_lst))
    assert not np.any(np.isnan(cx_p_lst))
    assert not np.any(np.isnan(cy_p_lst))
    assert not np.any(np.isnan(w2c_lst))

    # Prepare scene dataset
    writer = h5py.File(outpath, 'w')
    w_scene = writer.create_group(f"group_{gid:04d}_{vid}")
    w_w2c = w_scene.create_dataset("w2c", (len(indices),3,4), dtype=np.float32, chunks=(1,3,4))
    w_image = w_scene.create_dataset("image",
        (len(indices),720,1280,3), dtype=np.uint8, chunks=(1,720,1280,3), compression="gzip")

    # Pack the selected frames
    idx = -1
    ith = -1
    fname_lst = []
    for frame in iio.imiter(inroot / f"videos/group_{gid:04d}/{vid}.mp4"):
        ith += 1
        if ith not in indices:
            continue

        # Write current frame to dataset
        idx += 1
        w_w2c[idx] = np.ascontiguousarray(w2c_lst[idx])
        w_image[idx] = np.ascontiguousarray(frame)
        fname_lst.append(f"{ith:05d}")

        if idx >= len(indices) - 1:
            break

    if idx != len(indices) - 1:
        logging.info(f"Skip inconsistent #frames: group_{gid:04d}/{vid} {idx+1} != {len(indices)}.")
        writer.flush()
        writer.close()
        os.remove(outpath)
        return

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


class ArchiveState:
    def __init__(self, download_dir, metadata, folding=(0, 1), mod_lst=['annotations', 'videos'], wait=10):
        self.download_dir = download_dir
        self.metadata = metadata
        self.group_id_lst = sorted(list(self.metadata['group id'].unique()))
        self.mod_lst = mod_lst
        self.wait = wait

        # Filter a sub group
        self.group_id_lst = [v for v in self.group_id_lst if v % args.folding[1] == args.folding[0]]
        logging.info(f"TODO group_id: {self.group_id_lst}")

        # Init file sizes table
        self.size_table = {}
        self.downloaded = {}
        for gid in self.group_id_lst:
            self.size_table[gid] = {mod: (-1, -1) for mod in self.mod_lst}
            self.downloaded[gid] = {mod: False for mod in self.mod_lst}
        
        # Init status
        self.status = {gid: 'downloading' for gid in self.group_id_lst}

        # Init check
        self.check_download()
    
    def check_download(self):
        for gid in self.group_id_lst:
            if self.status[gid] != 'downloading':
                continue
            for mod in self.mod_lst:
                archive_path = self.download_dir / f"{mod}/group_{gid:04d}.tar.gz"
                if archive_path.exists():
                    nowsize = os.path.getsize(archive_path)
                    nowtime = time.time()
                    oldsize, oldtime = self.size_table[gid][mod]
                    if nowsize != oldsize:
                        self.size_table[gid][mod] = (nowsize, nowtime)
                    elif (nowtime - oldtime >= self.wait):
                        self.downloaded[gid][mod] = 1
            if all(self.downloaded[gid].values()):
                self.status[gid] = 'ready'
    
    def find_next_ready(self):
        for gid in self.group_id_lst:
            if self.status[gid] == 'ready':
                return gid
        return -1
    
    def mark_done(self, gid):
        self.status[gid] = 'done'

    def is_all_done(self):
        return all(v == 'done' for v in self.status.values())

    def decompress(self, gid):
        for mod in self.mod_lst:
            archive_path = self.download_dir / f"{mod}/group_{gid:04d}.tar.gz"
            logging.info(f"Decompressing {archive_path}...")
            subprocess.run(["tar", "-xzf", archive_path, "-C", self.download_dir], check=True)
            logging.info(f"Decompressed {archive_path}.")

    def remove_decompressed(self, gid):
        for mod in self.mod_lst:
            decomp_path = self.download_dir / 'SpatialVID' / f"{mod}/group_{gid:04d}"
            assert decomp_path.exists()
            logging.info(f"Removing {decomp_path}...")
            shutil.rmtree(decomp_path)
            logging.info(f"Removed {decomp_path}.")

    def process(self, gid, max_workers=32):
        # Process all videos
        vid_lst = sorted(list(self.metadata[self.metadata['group id'] == gid]['id']))

        inroot = self.download_dir / 'SpatialVID'
        outroot = self.download_dir / 'hdf5'
        todo_lst = [(i//100, vid) for i, vid in enumerate(vid_lst)]
        for sid, vid in todo_lst:
            (outroot / f"group_{gid:04d}" / f"{sid:02d}").mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing {len(vid_lst)} videos from group_{gid:04d}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(max_workers, len(todo_lst))) as executor:
            worker_function = partial(process_video, inroot=inroot, outroot=outroot, gid=gid)
            _ = list(executor.map(worker_function, todo_lst))
        logging.info(f"Processed {len(vid_lst)} videos from group_{gid:04d}.")



if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Process SpatialVID dataset into HDF5 on the fly."
    )
    parser.add_argument(
        "--download_dir",
        type=Path,
        help="Local directory to save dataset",
        default=Path("data/download/SpatialVID_HQ/"),
    )
    parser.add_argument(
        "--final_dir",
        type=Path,
        default=Path("data/hdf5/spatialvid_hq"),
    )
    parser.add_argument(
        "--folding", default=(0, 1), nargs=2, type=int
    )
    args = parser.parse_args()

    # Create the output directory
    (args.final_dir / "scenes").mkdir(parents=True, exist_ok=True)

    # Parse all group
    metadata = pd.read_csv(args.download_dir / "data/train/SpatialVID_HQ_metadata.csv")

    # Monitor downloading files
    archive_state = ArchiveState(download_dir=args.download_dir, metadata=metadata, folding=args.folding)

    # Mark already processed
    for gid in archive_state.group_id_lst:
        if (args.final_dir / "scenes" / f"group_{gid:04d}").exists():
            archive_state.mark_done(gid)
            logging.info(f"Already done: group_{gid:04d}")

    # Process downloaded split
    while not archive_state.is_all_done():
        archive_state.check_download()

        # Find downloaded file
        gid = archive_state.find_next_ready()
        if gid != -1:
            archive_state.decompress(gid)
            archive_state.process(gid)

            # Move to final_dir
            (args.download_dir / 'hdf5' / f"group_{gid:04d}").rename(args.final_dir / "scenes" / f"group_{gid:04d}")

            archive_state.mark_done(gid)
            archive_state.remove_decompressed(gid)

        time.sleep(1)
