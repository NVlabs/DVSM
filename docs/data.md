# Preparing a Custom Dataset

This guide explains the on-disk data format DVSM expects, so you can bring
your own posed-image dataset (from COLMAP, nerfstudio, or a custom capture
pipeline) into training/evaluation. It documents the same format implemented
in [`../dvsm/dataset.py`](../dvsm/dataset.py) — read the module docstring
there for the authoritative, code-level spec.

All commands below are meant to be run from the repository root (not from
inside `data_processing/`), matching the examples in
[`../docs/train.md`](../docs/train.md).

At a high level, one dataset (e.g. "my_scenes") is:
1. A single **HDF5 file** holding all scenes' images and camera poses.
2. A **JSON config** entry that tells the dataloader where that HDF5 file is
   and how to sample from it.
3. An optional cached **stats JSON** (auto-generated) and, for evaluation
   only, a **presampled split JSON** that fixes which frames are input vs.
   target views.

## 1. HDF5 scene layout

Each top-level key in the HDF5 file is a scene name, and its value is a group
with the following contents:

```
<scene_name>/                          (h5py.Group)
├── attrs
│   ├── n_frames : int                 # number of frames in the scene
│   ├── fovx     : float  (optional)   # shared horizontal FOV, radians
│   ├── fovy     : float  (optional)   # shared vertical FOV, radians
│   ├── cx_p     : float  (optional)   # shared principal point x, normalized [0,1]
│   └── cy_p     : float  (optional)   # shared principal point y, normalized [0,1]
├── image : uint8[N, H, W, 3]          # RGB frames, required
├── w2c   : float32[N, 4, 4] or [N, 3, 4]  # world-to-camera extrinsics, required
├── name  : bytes[N]                   # per-frame name/id, required (used for bookkeeping/debug)
├── fovx  : float32[N]                 # per-frame horizontal FOV, required only if attrs.fovx absent
├── fovy  : float32[N]                 # per-frame vertical FOV, required only if attrs.fovy absent
├── cx_p  : float32[N]                 # per-frame principal point x, required only if attrs.cx_p absent
└── cy_p  : float32[N]                 # per-frame principal point y, required only if attrs.cy_p absent
```

Notes:
- Use the `attrs` form whenever all frames in a scene share the same
  intrinsics (e.g. a single camera with fixed FOV) — this is smaller and
  faster to read. Fall back to the per-frame dataset form only if intrinsics
  actually vary across frames (e.g. multi-camera captures, or per-frame
  crops).
- `w2c` is a plain 3x4 (or 4x4, bottom row ignored) rigid transform matrix;
  only the rotation (`[:3,:3]`) and translation (`[:3,3]`) parts are read.
- FOV is used instead of raw focal length so intrinsics stay correct
  regardless of the resolution the images are stored/decoded at. Convert a
  pixel focal length `f` (at width/height `L`) to FOV with
  `fov = 2 * atan(0.5 * L / f)`.
- Store images uncompressed or gzip-compressed with `chunks=(1, H, W, 3)` so
  random single-frame reads stay fast (see `convert_colmap.py` for an
  example writer).
- Frame order matters only in that it defines each frame's index — the
  dataloader samples arbitrary subsets/subsequences by index, not sequential
  order, except when treating the last frames of a sequence as a fixed
  heldout set (see `num_frames_test` below).
- It's common to write one HDF5 file per scene first, then merge them with
  [`h5py.ExternalLink`](https://docs.h5py.org/en/stable/high/group.html#external-links)
  into a single composite file (`train.hdf5`, `test.hdf5`, ...) — this keeps
  per-scene conversion parallelizable while still presenting one file to the
  dataloader. See `convert_colmap.py`'s `all.hdf5` step for an example.

## 2. Datasets composition JSON

This is the JSON file passed to training/eval (e.g. `configs/*.json`,
referenced by `--dataset.train_path` / `--dataset.valid_path`). It maps a
dataset name to its metadata:

```json
{
  "my_scenes": {
    "path": "/data/hdf5/my_scenes/train.hdf5",
    "weight": 1.0,
    "min_frame_skip": 1,
    "max_frame_skip": 30,
    "use_heldout_set": false,
    "extra_skip": 0
  }
}
```

| Field | Default | Meaning |
|---|---|---|
| `path` | required | Path to the HDF5 file described above |
| `weight` | `1.0` | Relative sampling probability among all datasets in this config (normalized internally) |
| `min_frame_skip` | `1` | Minimum stride between consecutively sampled input frames. **Set to a negative value** to instead sample input frames evenly spread across the whole sequence (good for turntable/orbit-style captures where you want full coverage rather than a short contiguous clip) |
| `max_frame_skip` | none (whole sequence) | Maximum stride between consecutively sampled input frames |
| `use_heldout_set` | `false` | If true, target (supervision) views are always drawn from the scene's fixed heldout tail — see `num_frames_test` below — instead of from frames near the sampled input window |
| `extra_skip` | `0` | How many extra frames beyond the sampled input window's span may be picked as target views (ignored when `use_heldout_set` is true) |
| `samp` | required only for **evaluation** configs | Path to a presampled split JSON, see section 4 |

Multiple dataset entries can be mixed in one config — each training batch
sequence is drawn from one dataset, chosen according to `weight`.

The configs shipped in `configs/` use paths of the form
`data/hdf5/<dataset>/<split>.hdf5`, relative to the repository root. Place or
symlink your HDF5 files there, or edit the `path` entries.

## 3. Cached stats JSON (auto-generated)

The first time a `path` is loaded, the dataloader scans the HDF5 file for
scene names, frame counts, and resolutions, and caches the result next to the
HDF5 file as `<path-without-extension>.json`:

```json
{
  "scene_names": ["scene_0001", "scene_0002", ...],
  "num_frames": [312, 480, ...],
  "resolutions": [[1080, 1920], [1080, 1920], ...],
  "num_frames_test": [0, 30, ...]
}
```

- You normally don't need to write this file yourself — just make sure it's
  writable/regeneratable, or delete it if you change the underlying HDF5
  file. It can also be pre-built in bulk with
  `data_processing/produce_hdf5_meta.py`.
- `num_frames_test[i]` (optional, defaults to all-zero) marks the last `M`
  frames of scene `i` as a **fixed heldout tail**: those frames are never
  sampled as input frames, and are the only frames sampled as target views
  when `use_heldout_set` is set for that dataset. This is how DL3DV's
  benchmark split reserves specific frames per scene for evaluation — add
  `num_frames_test` manually to the cached JSON (or regenerate it with a
  custom script) if your dataset needs the same behavior.

## 4. Presampled split JSON (evaluation only)

`PresampledTestingDataset` (used for benchmark evaluation, not training)
reads a fixed input/target split per scene instead of sampling one at
runtime, so evaluation numbers are reproducible run to run:

```json
{
  "scene_0001": {
    "input":   [0, 8, 16, 24, 32, 40, 48, 56],
    "heldout": [4, 12, 20, 28, 36, 44, 52, 60]
  }
}
```

Generate this automatically with
`data_processing/produce_test_views_sampling.py`, which picks a fixed-stride
heldout set and k-means-clusters the remaining frames by camera pose to
select a diverse, fixed set of input views for a given input-view budget.

## Putting it together: converting a custom scene

The most common starting point is a folder of images with known (or
estimatable) camera poses. Two ready-made converters exist:

- **COLMAP / nerfstudio `transforms.json` scenes**:
  ```bash
  python data_processing/convert_colmap.py /path/to/colmap_root /data/hdf5/my_dataset
  ```
  This reads each scene's `sparse/0/` COLMAP reconstruction, writes one HDF5
  file per scene under `/data/hdf5/my_dataset/scenes/`, then composites them
  into `/data/hdf5/my_dataset/all.hdf5` via external links. Point your
  dataset JSON's `"path"` at `all.hdf5`.
- **pixelSplat/ZeroNVS-style `.torch` scenes** (e.g. RealEstate10K, ACID):
  ```bash
  python data_processing/acid.py --root /data/raw/my_dataset --outroot /data/hdf5/my_dataset/scenes --split_lst train test
  ```

For any other pose format, write a small conversion script following the
same pattern as `convert_colmap.py`: for each scene, compute per-frame
`w2c` and FOV/principal-point intrinsics, then write them into the HDF5
layout from section 1.

Once you have an HDF5 file, add an entry for it to a dataset JSON (section
2), point `configs/*.json` at it, and follow [../docs/train.md](../docs/train.md)
for how to launch training.

## Helpful scripts

Please check [data_processing](../data_processing/) for some useful scripts to work with existing dataset format.
