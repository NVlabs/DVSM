# Data processing scripts

| Script | Purpose |
|---|---|
| [`convert_colmap.py`](convert_colmap.py) | Converts COLMAP (`sparse/0/`) or nerfstudio `transforms.json` scenes into per-scene HDF5 files, then composites them into one `all.hdf5` via external links. The go-to converter for custom/self-captured scenes. |
| [`acid.py`](acid.py) | Converts pixelSplat/ZeroNVS-style per-scene `.torch` files (as used by RealEstate10K, ACID) into per-scene HDF5 files. |
| [`dl3dv_10k.py`](dl3dv_10k.py) | Converts one DL3DV-10K fold (e.g. `1K`, `2K`, ...) of nerfstudio `transforms.json` scenes into per-scene HDF5 files. Meant to be run once per fold. |
| [`dl3dv_split.py`](dl3dv_split.py) | Composites all per-scene DL3DV HDF5 files (across folds) into `train.hdf5` / `test.hdf5`, using `dl3dv_benchmark.txt` to decide which scenes are held out for the benchmark test split. |
| [`spatialvid_download.py`](spatialvid_download.py) | Downloads the SpatialVID / SpatialVID-HQ dataset archives from Hugging Face Hub. |
| [`spatialvid.py`](spatialvid.py) | Long-running pipeline that watches the SpatialVID download directory, decompresses each downloaded archive group as it arrives, and converts its videos + camera annotations into per-scene HDF5 files. Meant to run concurrently with `spatialvid_download.py`. |
| [`spatialvid_finalize.py`](spatialvid_finalize.py) | Merges per-scene SpatialVID HDF5 files within each group into one HDF5 file per group, and forms the final train/valid split. |
| [`produce_hdf5_meta.py`](produce_hdf5_meta.py) | Bulk-(re)generates the cached stats JSON (section 3 below) for HDF5 files matching a glob pattern, without needing to first load them through the dataloader. |
| [`produce_test_views_sampling.py`](produce_test_views_sampling.py) | Generates a presampled split JSON (section 4 below) for benchmark evaluation: reserves a fixed-stride heldout set per scene, and k-means-clusters the remaining frames by camera pose to pick a fixed, diverse set of input views for one or more input-view budgets (`--num_inputs`). |
| [`list_dataset.py`](list_dataset.py) | Quick inspection tool: prints per-scene resolution and summary stats (aspect ratios, min/avg/max frame count and pixel count) for a single HDF5 file. Useful for sanity-checking a dataset after conversion. |
