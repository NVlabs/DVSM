import json
import h5py
import numpy as np
from pathlib import Path

import torch


'''
Dataset spec
============

A "mixed dataset" is a collection of individually-recorded video/scan datasets
(e.g. RealEstate10K, DL3DV, ScanNet++, ...), each stored as one HDF5 file, plus
a JSON file describing how to mix and sample them.

1. Datasets composition JSON (`datasets_json_path` argument)
--------------------------------------------------------------
A dict mapping a dataset name to its metadata, e.g.:
    {
      "<ds_name>": {
        "path": "<path/to/dataset.h5>",       # required, see HDF5 layout below
        "weight": 1.0,                        # optional, sampling probability weight (default 1.0)
        "min_frame_skip": 1,                  # optional, min stride between sampled frames;
                                               #   if < 0, frames are sampled evenly across the whole
                                               #   sequence instead of by stride (see sample_frames)
        "max_frame_skip": null,               # optional, max stride between sampled frames
                                               #   (default: no limit, i.e. whole sequence length)
        "use_heldout_set": false,             # optional, whether target views must come from the
                                               #   dataset's own heldout tail (see num_frames_test below)
        "extra_skip": 0,                      # optional, how far outside the sampled input subsequence
                                               #   a heldout target view may be picked from (see sample_target)
        "samp": "<path/to/split.json>"        # required only for PresampledTestingDataset, see below
      },
      ...
    }

2. Per-dataset HDF5 file layout (`meta["path"]`)
--------------------------------------------------------------
Each top-level HDF5 key is a scene name, containing a group with:
    attrs:
        n_frames: int                      # number of frames in the scene
        fovx, fovy: float                  # optional, shared horizontal/vertical FOV (radians) for
                                            #   all frames; if absent, per-frame datasets are used instead
        cx_p, cy_p: float                  # optional, shared principal point (normalized) for all frames
    datasets:
        image: uint8[N, H, W, 3]           # RGB frames
        w2c: float[N, 4, 4] (or [N, 3, 4]) # world-to-camera extrinsics; only the top-left 3x3
                                            #   rotation and first 3 rows of translation are used
        name: bytes[N]                     # per-frame file/frame name, for bookkeeping
        fovx, fovy, cx_p, cy_p: float[N]   # per-frame intrinsics, used when the matching attrs are absent

3. Cached stats JSON (`<dataset path>.json`, auto-generated alongside the HDF5 file)
--------------------------------------------------------------
Scanning an HDF5 file for scene names/resolutions/frame counts is slow, so the
result is cached next to the HDF5 file the first time it's loaded:
    {
      "scene_names": [...],                # sorted list of scene names
      "num_frames": [...],                 # n_frames per scene, aligned with scene_names
      "resolutions": [[H, W], ...],        # image resolution per scene
      "num_frames_test": [...]             # optional, size of a heldout tail per scene (see below);
                                            #   defaults to all-zero if absent
    }
Some datasets reserve the LAST `num_frames_test[i]` frames of a scene as a fixed
heldout set that training never samples as input, and that is used as the sole
source of target views when `use_heldout_set` is set for that dataset.

4. Presampled test split JSON (`meta["samp"]`, used only by PresampledTestingDataset)
--------------------------------------------------------------
A dict mapping scene name to a fixed input/target frame-index split, e.g.:
    {
      "<scene_name>": {
        "input": [i0, i1, ...],            # frame indices used as input/source views
        "heldout": [j0, j1, ...]           # frame indices used as target views to evaluate against
      },
      ...
    }
'''


class MixedTrainingDataset(torch.utils.data.Dataset):
    '''
    Load one image from the mixed dataset to serve MixedBatchRandomSampler.
    '''
    def __init__(self, datasets_json_path, randomize_static_time=False, max_ts=60., keep_unscaled_image=False):
        # Parse datasets composition (see module docstring, section 1)
        if isinstance(datasets_json_path, dict):
            datasets_json = datasets_json_path
        else:
            with open(datasets_json_path) as f:
                datasets_json = json.load(f)

        self.randomize_static_time = randomize_static_time
        self.max_ts = max_ts
        self.keep_unscaled_image = keep_unscaled_image

        # Init loaders. All lists below are aligned by dataset index i_ds.
        self.reader_lst = []
        self.path_lst = []
        self.ds_name_lst = []
        self.weight_lst = []
        self.min_frame_skip_lst = [] # When set to <0, random sample from entire sequence
        self.max_frame_skip_lst = []
        self.num_frames_test_lst = [] # Some dataset has another heldout set at the end of a sequence
        self.use_heldout_set = []
        self.extra_skip_lst = []
        self.scene_names_lst = [] # scene_names_lst[i_ds][i_scene] = scene name string
        self.image_res_lst = [] # image_res_lst[i_ds][i_scene] = (H, W)
        self.num_frames_lst = [] # num_frames_lst[i_ds][i_scene] = total frame count of the scene
        self.total_scenes = 0

        for i_ds, (ds_name, meta) in enumerate(datasets_json.items()):
            self.reader_lst.append(None) # Should be init by each process
            self.path_lst.append(meta["path"])
            self.ds_name_lst.append(ds_name)
            self.weight_lst.append(meta.get("weight", 1.0))
            self.min_frame_skip_lst.append(meta.get("min_frame_skip", 1))
            self.max_frame_skip_lst.append(meta.get("max_frame_skip", None))
            self.use_heldout_set.append(meta.get("use_heldout_set", False))
            self.extra_skip_lst.append(meta.get("extra_skip", 0))

            # Build (or load) the cached per-dataset stats file (see module docstring, section 3)
            path = Path(meta["path"])
            statpath = path.with_suffix('.json')
            if not statpath.exists():
                print(f"Preparing {statpath} for the first time")
                with h5py.File(path, 'r') as f:
                    scene_names = sorted(list(f.keys()))
                    num_frames = []
                    resolutions = []
                    for i_scene, sn in enumerate(scene_names):
                        num_frames.append(int(f[sn].attrs['n_frames']))
                        resolutions.append(f[sn]['image'].shape[1:3])
                with open(statpath, 'w') as f:
                    json.dump({
                        'scene_names': scene_names,
                        'num_frames': num_frames,
                        'resolutions': resolutions,
                    }, f)

            with open(statpath) as f:
                stat = json.load(f)
            self.scene_names_lst.append(stat['scene_names'])
            self.total_scenes += len(stat['scene_names'])
            self.image_res_lst.append(stat['resolutions'])
            self.num_frames_lst.append(stat['num_frames'])
            if 'num_frames_test' in stat:
                self.num_frames_test_lst.append(stat['num_frames_test'])
            else:
                self.num_frames_test_lst.append([0] * len(stat['scene_names']))

        self.weight_lst = np.array(self.weight_lst)
        self.weight_lst /= self.weight_lst.sum()

    def __repr__(self):
        return '\n'.join(
            f'  {self.ds_name_lst[i]:30s}: size={len(self.scene_names_lst[i]):6d} / prob={self.weight_lst[i]*100:.2f}%'
            for i in range(len(self.path_lst))
        )

    def __len__(self):
        return self.total_scenes

    def sample_frames(self, i_ds, i_scene, num_frames):
        '''
        Sample `num_frames` input/source frame indices for one scene of dataset i_ds.

        Two sampling modes, chosen per-dataset by `min_frame_skip`:
          - min_frame_skip < 0: evenly sample `num_frames` indices spanning
            (almost) the entire usable sequence.
          - min_frame_skip >= 0: sample a contiguous-ish subsequence where the
            gap between consecutive frames is drawn from
            [min_frame_skip, max_frame_skip], then shuffle. Falls back to
            sampling without replacement, or to recursively sampling half as
            many frames and duplicating them, if the sequence is too short.
        '''
        num_total = self.num_frames_lst[i_ds][i_scene]
        min_frame_skip = self.min_frame_skip_lst[i_ds]
        max_frame_skip = self.max_frame_skip_lst[i_ds]

        # Some dataset has individual heldout set at the end of a sequence.
        # Never sample them.
        num_total = num_total - self.num_frames_test_lst[i_ds][i_scene]

        if min_frame_skip < 0:
            # Evanly sample throughout the sequence
            pad = 1 + round(num_total / num_frames * 0.5)
            frame_start = np.random.randint(0, pad)
            frame_end = num_total - 1 - np.random.randint(0, pad)
            frame_indices = np.round(np.linspace(frame_start, frame_end, num_frames)).astype(int)
            return frame_indices

        if max_frame_skip is None:
            max_frame_skip = num_total

        # Draw num_frames-1 random gaps and accumulate them into frame offsets
        # relative to a (yet undetermined) start frame.
        frame_skip = np.random.randint(low=min_frame_skip, high=max_frame_skip + 1, size=[num_frames - 1])
        frame_skip = np.concatenate([[0], frame_skip])
        frame_offset = np.cumsum(frame_skip)
        if num_total > frame_offset[-1]:
            # The sampled span fits in the sequence: pick a random start and use it as-is
            frame_start = np.random.randint(0, num_total - frame_offset[-1])
            frame_indices = frame_start + frame_offset
        elif num_total > num_frames:
            # Span too long for this sequence: fall back to a uniform random subset
            frame_indices = np.random.choice(num_total, size=num_frames, replace=False)
        elif num_total < 2:
            # Degenerate scene (0 or 1 frame): repeat the same frame
            frame_indices = np.zeros([num_frames], dtype=int)
        else:
            # Not enough frames even for a uniform subset: sample half of total frame and
            # duplicate to target length
            frame_indices = self.sample_frames(i_ds, i_scene, num_total//2)
            frame_indices = np.resize(frame_indices, num_frames)

        np.random.shuffle(frame_indices)
        return frame_indices

    def sample_target(self, i_ds, i_scene, frame_indices, num_target):
        '''
        Sample `num_target` heldout/target view indices for a scene, given the
        already-sampled input `frame_indices`.

        If the dataset is configured with `use_heldout_set`, targets are always
        drawn from the scene's fixed heldout tail (see module docstring, section 3).
        Otherwise targets are drawn from frames near the input subsequence: the
        window [min(frame_indices)-extra_skip, max(frame_indices)+extra_skip]
        excluding the input frames themselves. `extra_skip` (per-dataset) controls
        how far outside the input span target views may be picked from.
        '''
        if self.num_frames_test_lst[i_ds] is not None and self.use_heldout_set[i_ds]:
            N = self.num_frames_lst[i_ds][i_scene]
            M = self.num_frames_test_lst[i_ds][i_scene]
            candidates = list(range(N-M, N))
        else:
            num_total = self.num_frames_lst[i_ds][i_scene] - self.num_frames_test_lst[i_ds][i_scene]
            extra_skip = self.extra_skip_lst[i_ds] # Allow target views outside the input subseq
            frame_indices = set(frame_indices)
            candidates = [
                i
                for i in range(min(frame_indices)-extra_skip, max(frame_indices)+extra_skip+1)
                if i not in frame_indices and i >= 0 and i < num_total
            ]
            if len(candidates) == 0:
                # No valid neighbor exists (e.g. a 1-frame input window with extra_skip=0):
                # fall back to reusing input frames as targets.
                candidates = list(frame_indices)

        if num_target < len(candidates):
            target_indices = np.random.choice(candidates, size=num_target, replace=False)
        else:
            # Not enough distinct candidates: use all of them plus random duplicates
            target_indices = np.concatenate([
                candidates,
                np.random.choice(candidates, size=num_target-len(candidates), replace=True)])
        return target_indices

    @staticmethod
    def aspect_ratio_crop(frame, ap_ratio, cp_ratio):
        '''
        In-place center-crop `frame['image']` (and adjust `frame['intr']` to match)
        to a target aspect ratio `ap_ratio` (width/height), optionally combined with
        a zoom-in crop of `cp_ratio` (fraction of each side cropped away, e.g. 0.1
        zooms in by trimming 10% off width and height before the aspect crop).
        No-op if both ratios are non-positive (i.e. keep native aspect, no zoom).
        '''
        if ap_ratio <= 0 and cp_ratio <= 0:
            return

        # Find the maximum resolution ratio
        H, W = frame['image'].shape[-2:]
        ori_ap_ratio = W / H
        if ap_ratio < ori_ap_ratio:
            h_ratio = 1.
            w_ratio = ap_ratio / ori_ap_ratio
        else:
            h_ratio = ori_ap_ratio / ap_ratio
            w_ratio = 1.

        # Zoom-in by cropping
        if cp_ratio > 0:
            h_ratio *= (1 - cp_ratio)
            w_ratio *= (1 - cp_ratio)

        # Derive the top-left corner position
        ir = round((1 - h_ratio) * 0.5 * H)
        ic = round((1 - w_ratio) * 0.5 * W)
        h = round(h_ratio * H)
        w = round(w_ratio * W)

        # Crop image
        frame['image'] = frame['image'][..., ir:ir+h, ic:ic+w]

        # Apply the crop to intrinsics
        fx, fy, cx, cy = frame['intr']
        fx = fx * W / w
        fy = fy * H / h
        cx = (cx * W - ic) / w
        cy = (cy * H - ir) / h
        frame['intr'] = np.array([fx, fy, cx, cy], dtype=np.float32)

    def __getitem__(self, samp):
        '''
        Read one frame specified by the batch sampler.

        `samp` is a tuple produced by MixedBatchRandomSampler.sample_batch_indices:
            (batch_id, is_heldout, i_ds, i_scene, i_frame, ap_ratio, cp_ratio, t_size)
        where batch_id groups frames belonging to the same sequence within a batch,
        is_heldout marks target (True) vs input (False) views, (i_ds, i_scene, i_frame)
        addresses the source frame, and (ap_ratio, cp_ratio, t_size) control the
        crop/resize applied to it (see aspect_ratio_crop).

        Returns a dict with one frame's data: image (float CHW in [0,1]), camera
        pose (c2w_r, c2w_t), normalized intrinsics (intr = [fx, fy, cx, cy]),
        timestamp (ts), and bookkeeping fields (batch_id, is_heldout, ds_name,
        scene_name, frame_name).
        '''

        # Unpack loading target setup
        batch_id, is_heldout, i_ds, i_scene, i_frame, ap_ratio, cp_ratio, t_size = samp
        ds_name = self.ds_name_lst[i_ds]
        scene_name = self.scene_names_lst[i_ds][i_scene]

        # Prepare the data reader (lazily opened per worker process, since
        # h5py file handles cannot be shared/forked across processes)
        if self.reader_lst[i_ds] is None:
            self.reader_lst[i_ds] = h5py.File(self.path_lst[i_ds], 'r')
        data = self.reader_lst[i_ds][scene_name]
        attrs = dict(data.attrs.items())

        # Load the frame
        image = torch.tensor(data['image'][i_frame], dtype=torch.uint8).moveaxis(-1, 0).contiguous()
        image = image.float() / 255.

        # Parse extrinsic: HDF5 stores world-to-camera (w2c); convert to
        # camera-to-world (c2w) rotation/translation, which is what the model consumes
        w2c = data['w2c'][i_frame]
        w2c_r = w2c[:3, :3]
        w2c_t = w2c[:3, 3]
        c2w_r = np.copy(w2c_r.T)
        c2w_t = -c2w_r @ w2c_t

        # Parse intrinsic (normalized): fovx/fovy/cx_p/cy_p are shared across all
        # frames of a scene when stored as HDF5 attrs, otherwise per-frame datasets
        fovx = attrs['fovx'] if 'fovx' in attrs else data['fovx'][i_frame]
        fovy = attrs['fovy'] if 'fovy' in attrs else data['fovy'][i_frame]
        cx = attrs['cx_p'] if 'cx_p' in attrs else data['cx_p'][i_frame]
        cy = attrs['cy_p'] if 'cy_p' in attrs else data['cy_p'][i_frame]
        fx = 0.5 / np.tan(0.5*fovx)
        fy = 0.5 / np.tan(0.5*fovy)

        # Parse frame time: all scenes are static, so timestamp is either fixed at 0
        # or randomized within [0, max_ts] so the model also learns to handle
        # static content at arbitrary times (see randomize_static_time)
        if self.randomize_static_time:
            ts = np.random.uniform(0, self.max_ts)
        else:
            ts = 0

        frame = dict(
            batch_id=batch_id,
            ts=ts,
            is_heldout=is_heldout,
            ds_name=ds_name,
            scene_name=scene_name,
            frame_name=data['name'][i_frame].decode(),
            image=image,
            c2w_r=c2w_r,
            c2w_t=c2w_t,
            intr=np.array([fx, fy, cx, cy], dtype=np.float32),
        )

        # Apply cropping
        self.aspect_ratio_crop(frame, ap_ratio, cp_ratio)

        # Resize to target resolution
        if self.keep_unscaled_image:
            frame['src_image'] = frame['image']
        if frame['image'].shape[-2:] != t_size:
            frame['image'] = torch.nn.functional.interpolate(
                frame['image'].unsqueeze(0),
                size=t_size, mode="bilinear", align_corners=False, antialias=True).squeeze(0)

        return frame


class MixedBatchRandomSampler(torch.utils.data.Sampler):
    '''
    Sample a batch consisting of #seq x #img_per_seq.

    Each yielded batch packs `num_seq` sequences (scenes), each contributing
    `num_img_per_seq` input frames and some number of heldout target frames
    (`num_target_per_gpu // num_seq`), all sharing the same target resolution
    and aspect ratio for this batch. `num_seq` is derived from
    `num_input_per_gpu // num_img_per_seq` so the total input frame count per
    GPU stays roughly constant regardless of how many frames each sequence uses.
    Batch items are consumed by MixedTrainingDataset.__getitem__ and later
    regrouped into tensors by `batchify_sequences`.
    '''
    def __init__(self,
                 mixed_dataset,
                 min_img_per_seq,
                 max_img_per_seq,
                 num_target_per_gpu,
                 num_input_per_gpu,
                 num_pixels,
                 patch_size,
                 remain_iter,
                 divisible_only=False,
                 fixed_img_per_seq=False,
                 ap_ratio_lst="",
                 crop_pad_ratio=0.):
        self.mixed_dataset = mixed_dataset
        self.len_candidate = list(range(min_img_per_seq, max_img_per_seq+1))
        if divisible_only and num_input_per_gpu > 0:
            self.len_candidate = [v for v in self.len_candidate if (num_input_per_gpu%v==0)]
            assert len(self.len_candidate) > 0
        self.fixed_img_per_seq = fixed_img_per_seq
        self.fixed_it = 0
        self.num_target_per_gpu = num_target_per_gpu
        self.num_input_per_gpu = num_input_per_gpu
        self.num_pixels = num_pixels
        self.patch_size = patch_size
        self.remain_iter = remain_iter

        # The image aspect ratios on a gpu should all be the same
        if ap_ratio_lst != "":
            self.ap_ratio_lst = [
                float(r.split(':')[0]) / float(r.split(':')[1])
                for r in ap_ratio_lst.split(',')]
        else:
            self.ap_ratio_lst = [0.]

        # Zoom-in into the image central
        self.crop_pad_ratio = crop_pad_ratio

    def sample_batch_indices(self):
        # Determine #seq x #imgs_per_seq on current gpu
        if self.fixed_img_per_seq:
            num_img_per_seq = self.len_candidate[self.fixed_it]
            self.fixed_it = (self.fixed_it + 1) % len(self.len_candidate)
        else:
            num_img_per_seq = np.random.choice(self.len_candidate)
        num_seq = max(1, self.num_input_per_gpu // num_img_per_seq)
        num_target_per_seq = self.num_target_per_gpu // num_seq

        # Sample aspect ratio
        ap_ratio = np.random.choice(self.ap_ratio_lst)

        # Sample padding crop
        if self.crop_pad_ratio > 0:
            cp_ratio = np.random.uniform(0, self.crop_pad_ratio)
        else:
            cp_ratio = 0

        # Sample the batch from the mixed datasets
        batch_indices = []
        for batch_id in range(num_seq):
            i_ds = np.random.choice(len(self.mixed_dataset.weight_lst), p=self.mixed_dataset.weight_lst)
            i_scene = np.random.choice(len(self.mixed_dataset.scene_names_lst[i_ds]))

            # Calculate target resolution
            if ap_ratio > 0:
                th = (self.num_pixels / (ap_ratio * self.patch_size * self.patch_size)) ** 0.5
                tw = th * ap_ratio
            else:
                # The batch size should be 1 or all the datasets are in same aspect ratio
                sh, sw = self.mixed_dataset.image_res_lst[i_ds][i_scene]
                th = (self.num_pixels / (sw/sh * self.patch_size * self.patch_size)) ** 0.5
                tw = th * sw/sh
            th = round(th) * self.patch_size
            tw = round(tw) * self.patch_size

            # Sample frames
            frame_indices = self.mixed_dataset.sample_frames(i_ds, i_scene, num_img_per_seq)
            target_indices = self.mixed_dataset.sample_target(i_ds, i_scene, frame_indices, num_target_per_seq)

            for i_frame in frame_indices:
                batch_indices.append((
                    batch_id, False, i_ds, i_scene, i_frame,
                    ap_ratio, cp_ratio, (th, tw)))
            for i_frame in target_indices:
                batch_indices.append((
                    batch_id, True, i_ds, i_scene, i_frame,
                    ap_ratio, cp_ratio, (th, tw)))

        return batch_indices

    def __iter__(self):
        for _ in range(self.remain_iter):
            yield self.sample_batch_indices()

    def __len__(self):
        return self.remain_iter

    @staticmethod
    def batchify_sequences(batch):
        '''
        Collate a flat list of per-frame dicts (as produced by
        MixedTrainingDataset.__getitem__, in the order emitted by
        sample_batch_indices: input frames then target frames, per sequence,
        concatenated across sequences) into batched input_views/target_views
        tensors of shape [num_seq, seq_len, ...].
        '''
        # Recover batch dimension
        num_img_per_seq = sum(int(item['batch_id']==0 and not item['is_heldout']) for item in batch)
        num_target_per_seq = sum(int(item['batch_id']==0 and item['is_heldout']) for item in batch)
        seq_size = num_img_per_seq + num_target_per_seq
        num_seq = len(batch) // seq_size

        # Sanity check
        assert len(batch) == num_seq * seq_size
        for i in range(len(batch)):
            assert batch[i]['batch_id'] == batch[i - i%seq_size]['batch_id']
            assert (not batch[i]['is_heldout']) == (i%seq_size < num_img_per_seq)
            assert batch[i]['is_heldout'] == (i%seq_size >= num_img_per_seq)

        # Batchify
        ds_names = [batch[i*seq_size]['ds_name'] for i in range(num_seq)]
        scene_names = [batch[i*seq_size]['scene_name'] for i in range(num_seq)]
        frame_names = np.array([batch[i]['frame_name'] for i in range(len(batch))])
        ts = torch.tensor([batch[i]['ts'] for i in range(len(batch))], dtype=torch.float32)
        image = torch.stack([batch[i]['image'] for i in range(len(batch))])
        c2w_r = torch.tensor(np.stack([batch[i]['c2w_r'] for i in range(len(batch))]), dtype=torch.float32)
        c2w_t = torch.tensor(np.stack([batch[i]['c2w_t'] for i in range(len(batch))]), dtype=torch.float32)
        intr = torch.tensor(np.stack([batch[i]['intr'] for i in range(len(batch))]), dtype=torch.float32)

        # Reshape
        frame_names = np.array(frame_names).reshape(num_seq, seq_size)
        ts = ts.unflatten(0, (num_seq, seq_size))
        image = image.unflatten(0, (num_seq, seq_size))
        c2w_r = c2w_r.unflatten(0, (num_seq, seq_size))
        c2w_t = c2w_t.unflatten(0, (num_seq, seq_size))
        intr = intr.unflatten(0, (num_seq, seq_size))

        # Split into input and heldout target views
        input_views = dict(
            frame_names=frame_names[:, :num_img_per_seq],
            ts=ts[:, :num_img_per_seq].contiguous(),
            image=image[:, :num_img_per_seq].contiguous(),
            c2w_r=c2w_r[:, :num_img_per_seq].contiguous(),
            c2w_t=c2w_t[:, :num_img_per_seq].contiguous(),
            intr=intr[:, :num_img_per_seq].contiguous(),
        )
        target_views = dict(
            frame_names=frame_names[:, num_img_per_seq:],
            ts=ts[:, num_img_per_seq:].contiguous(),
            image=image[:, num_img_per_seq:].contiguous(),
            c2w_r=c2w_r[:, num_img_per_seq:].contiguous(),
            c2w_t=c2w_t[:, num_img_per_seq:].contiguous(),
            intr=intr[:, num_img_per_seq:].contiguous(),
        )

        # Batchify source frames if there is
        if 'src_image' in batch[0]:
            src_image = torch.stack([batch[i]['src_image'] for i in range(len(batch))])
            src_image = src_image.unflatten(0, (num_seq, seq_size))
            input_views['src_image'] = src_image[:, :num_img_per_seq].contiguous()
            target_views['src_image'] = src_image[:, num_img_per_seq:].contiguous()

        return dict(
            ds_names=ds_names,
            scene_names=scene_names,
            frame_names=frame_names,
            input_views=input_views,
            target_views=target_views)


class PresampledTestingDataset(MixedTrainingDataset):
    '''
    Load one scene from the mixed dataset where the views are sampled and fixed.

    Unlike MixedTrainingDataset, the input/target frame split for each scene is
    not sampled at runtime but read from a presampled split JSON file per
    dataset (`meta["samp"]`, see module docstring section 4), so that
    evaluation runs are deterministic and reproducible across runs. One
    `__getitem__` call returns one fully-batched scene (all its input and
    target views), rather than a single frame.
    '''
    def __init__(self, datasets_json_path, num_pixels, patch_size, ap_ratio="16:9", cp_ratio=0., keep_unscaled_image=False):
        # Parse datasets composition
        if isinstance(datasets_json_path, dict):
            datasets_json = datasets_json_path
        else:
            with open(datasets_json_path) as f:
                datasets_json = json.load(f)

        self.randomize_static_time = False
        self.keep_unscaled_image = keep_unscaled_image

        self.num_pixels = num_pixels
        self.patch_size = patch_size
        ap_ratio = ap_ratio.split(':')
        self.ap_ratio = float(ap_ratio[0]) / float(ap_ratio[1])
        self.cp_ratio = cp_ratio

        # Init loaders
        self.reader_lst = []
        self.path_lst = []
        self.ds_name_lst = []
        self.ds_name_idx = {}
        self.scene_names_lst = []
        self.image_res_lst = []
        self.input_frame_idx_lst = []
        self.target_frame_idx_lst = []
        self.remap_idx = []
        self.total_scenes = 0

        for i_ds, (ds_name, meta) in enumerate(datasets_json.items()):
            self.reader_lst.append(None) # Should be init by each process
            self.path_lst.append(meta["path"])
            self.ds_name_lst.append(ds_name)
            self.ds_name_idx[ds_name] = i_ds

            path = Path(meta["path"])
            statpath = path.with_suffix('.json')
            if statpath.exists():
                with open(statpath) as f:
                    stat = json.load(f)
                res_dict = {scene_name: res for scene_name, res in zip(stat['scene_names'], stat['resolutions'])}
            else:
                with h5py.File(path, 'r') as f:
                    res_dict = {}
                    for scene_name in f.keys():
                        res_dict[scene_name] = f[scene_name]['image'].shape[1:3]

            with open(meta["samp"]) as f:
                samp = json.load(f)

            self.scene_names_lst.append([])
            self.input_frame_idx_lst.append([])
            self.target_frame_idx_lst.append([])
            self.image_res_lst.append([])
            self.total_scenes += len(samp)

            for i_scene, (scene_name, split) in enumerate(samp.items()):
                assert scene_name in res_dict
                self.scene_names_lst[-1].append(scene_name)
                self.image_res_lst[-1].append(res_dict[scene_name])
                self.input_frame_idx_lst[-1].append(split['input'])
                self.target_frame_idx_lst[-1].append(split['heldout'])
                self.remap_idx.append((i_ds, i_scene))

    def __repr__(self):
        return '\n'.join(
            f'  {self.ds_name_lst[i]:30s}: size={len(self.scene_names_lst[i]):6d}'
            for i in range(len(self.path_lst))
        )

    def __len__(self):
        return self.total_scenes

    def __getitem__(self, idx):
        '''
        Read one sequence of a scene.
        '''
        # Remap the flatten index
        i_ds, i_scene = self.remap_idx[idx]

        # Caculate target resolution
        if self.ap_ratio > 0:
            th = (self.num_pixels / (self.ap_ratio * self.patch_size * self.patch_size)) ** 0.5
            tw = th * self.ap_ratio
        else:
            # The batch size should be 1 or all the datasets are in same aspect ratio
            sh, sw = self.image_res_lst[i_ds][i_scene]
            s_ap_ratio = sw / sh
            th = (self.num_pixels / (s_ap_ratio * self.patch_size * self.patch_size)) ** 0.5
            tw = th * s_ap_ratio

        th = round(th) * self.patch_size
        tw = round(tw) * self.patch_size

        # Read each frame
        batch = [
            *[
                super(PresampledTestingDataset, self).__getitem__((
                    0, False, i_ds, i_scene, i_frame,
                    self.ap_ratio, self.cp_ratio, (th, tw)))
                for i_frame in self.input_frame_idx_lst[i_ds][i_scene]
            ],
            *[
                super(PresampledTestingDataset, self).__getitem__((
                    0, True, i_ds, i_scene, i_frame,
                    self.ap_ratio, self.cp_ratio, (th, tw)))
                for i_frame in self.target_frame_idx_lst[i_ds][i_scene]
            ],
        ]

        # Batchify the scene
        batch = MixedBatchRandomSampler.batchify_sequences(batch)

        return batch
