import torch


class PosePreprocessor:
    def __init__(self, c2w_r, c2w_t, subset_idx=None, scale_factor=0.75,
                 pose_avg_strategy='down'):
        """
        Compute a series of [offset -> scale -> rotation] for normalizing poses.
        c2w_r:        [b, v, 3, 3]
        c2w_t:        [b, v, 3]
        subset_idx:   [v] or None. Use a subset of view to compute normalization.
        scale_factor: Rescale factor for the normalized camera
        pose_avg_strategy: Preserve the average 'down' or 'forward' axis when
                           orthogonalizing the average rotation.
        """

        assert len(c2w_r.shape) == 4 and c2w_r.shape[-2:] == (3, 3)
        assert len(c2w_t.shape) == 3 and c2w_t.shape[2] == 3
        assert c2w_r.shape[:2] == c2w_t.shape[:2]

        # Compute SE(3) normalizing factor
        if subset_idx is not None:
            assert len(subset_idx.shape) == 1 and len(subset_idx) == c2w_r.shape[1]
            avg_t = c2w_t[:, subset_idx].mean(1)
            avg_r = compute_avg_rotation(c2w_r[:, subset_idx], pose_avg_strategy)
        else:
            avg_t = c2w_t.mean(1)
            avg_r = compute_avg_rotation(c2w_r, pose_avg_strategy)

        # Target transformation for normalization
        self.r = avg_r.moveaxis(2,1).contiguous()
        self.t = (-self.r @ avg_t.unsqueeze(-1)).squeeze(-1).unsqueeze(1).contiguous()

        # Target scene rescaling
        new_t = torch.einsum('bij, bvj -> bvi', self.r, c2w_t) + self.t
        post_scale = new_t.abs().amax((1,2), keepdim=True)
        self.s = scale_factor / post_scale

    def __call__(self, c2w_r, c2w_t):
        c2w_r = torch.einsum('bij, bvjk -> bvik', self.r, c2w_r)
        c2w_t = torch.einsum('bij, bvj -> bvi', self.r, c2w_t) + self.t
        c2w_t = c2w_t * self.s
        return c2w_r, c2w_t


def compute_avg_rotation(c2w_r, strategy='down'):
    # c2w_r: [b, v, 3, 3]
    avg_down = c2w_r[..., 1].mean(1)
    avg_forward = c2w_r[..., 2].mean(1)

    if strategy == 'down':
        avg_down = torch.nn.functional.normalize(avg_down, dim=-1)
    elif strategy == 'forward':
        avg_forward = torch.nn.functional.normalize(avg_forward, dim=-1)
    else:
        raise ValueError(
            f"Unknown pose averaging strategy {strategy!r}; "
            "expected 'down' or 'forward'")

    avg_right = torch.nn.functional.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
    if strategy == 'down':
        avg_forward = torch.nn.functional.normalize(torch.cross(avg_right, avg_down, dim=-1), dim=-1)
    else:
        avg_down = torch.nn.functional.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)
    avg_r = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    return avg_r
