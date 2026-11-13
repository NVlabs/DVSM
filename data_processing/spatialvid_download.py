import argparse
from huggingface_hub import hf_hub_download, snapshot_download

import pandas as pd
from tqdm import tqdm
from pathlib import Path


def download(repo_id, output_dir, hub_path):
    outpath = args.output_dir / hub_path
    if outpath.exists():
        print(f"Skip existed {outpath}", flush=True)
    else:
        print(f"Downloading file to {outpath}...", end=' ', flush=True)
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=hub_path,
            local_dir=output_dir,
            resume_download=True)
        print('done', flush=True)
    return outpath


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Download SpatialVID dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        choices=["SpatialVID", "SpatialVID-HQ"],
        required=True,
        help="Dataset type to download (SpatialVID or SpatialVID-HQ)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Local directory to save dataset",
        default=Path("data/download/SpatialVID_HQ/"),
    )
    args = parser.parse_args()

    repo_id = f"SpatialVID/{args.repo_id}"

    # Download metadata
    metapath = download(
        repo_id,
        args.output_dir,
        hub_path=f"data/train/{args.repo_id.replace('-', '_')}_metadata.csv")
    metadata = pd.read_csv(metapath)

    # Download each group id
    group_id_lst = sorted(list(metadata['group id'].unique()))
    for gid in tqdm(group_id_lst):
        for mod in ['videos', 'annotations']:
            hub_path = f"{mod}/group_{gid:04d}.tar.gz"
            download(repo_id, args.output_dir, hub_path=hub_path)
