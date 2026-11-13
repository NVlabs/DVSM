import torch


class PluckerEncoding(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        if method == 'plucker':
            self.pose_enc = compute_plucker
            self.dim = 6
        elif method == 'plucker_aug':
            self.pose_enc = compute_plucker_aug
            self.dim = 9
        elif method == 'plucker_custom':
            self.pose_enc = compute_plucker_custom
            self.dim = 6
        elif method == 'concat':
            self.pose_enc = simple_concat
            self.dim = 6
        else:
            raise NotImplementedError

    def forward(self, intr, c2w_r, c2w_t, h, w):
        '''
        Encode camera parameters.
        Args:
            intr (torch.tensor): [b, v, 4]
            c2w_r (torch.tensor): [b, v, 3, 3]
            c2w_t (torch.tensor): [b, v, 3]
            (h, w): size of the ray map
        '''
        # Sanity check
        assert len(intr.shape) == 3 and intr.shape[2] == 4
        assert len(c2w_r.shape) == 4 and c2w_r.shape[-2:] == (3, 3)
        assert len(c2w_t.shape) == 3 and c2w_t.shape[2] == 3
        assert intr.shape[:2] == c2w_r.shape[:2]
        assert intr.shape[:2] == c2w_t.shape[:2]

        # Encode camera parameters
        ray_o, ray_d = compute_rays(intr, c2w_r, c2w_t, h, w)
        plucker = self.pose_enc(ray_o, ray_d)

        return plucker



def compute_rays(intr, c2w_r, c2w_t, h, w):
    '''
    Args:
        intr (torch.tensor): [b, v, 4]
        c2w_r (torch.tensor): [b, v, 3, 3]
        c2w_t (torch.tensor): [b, v, 3]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    '''
    b, v = intr.shape[:2]

    # Unpack intrinsics into [b*v, 1] and denormalized
    fx, fy, cx, cy = intr.flatten(0, 1).T.unsqueeze(-1).contiguous()
    fx = fx * w
    fy = fy * h
    cx = cx * w
    cy = cy * h

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=intr.device) + 0.5
    idx_y = torch.arange(h, device=intr.device) + 0.5
    idx_x = idx_x[None, :].expand(h, -1)  # [h, w]
    idx_y = idx_y[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    x = (idx_x - cx) / fx     # [b*v, h*w]
    y = (idx_y - cy) / fy     # [b*v, h*w]
    z = torch.ones_like(x)    # [b*v, h*w]

    # Transform to world coordinate
    c2w_r = c2w_r.flatten(0, 1)                          # [b*v, 3, 3]
    ray_d = torch.stack([x, y, z], dim=1)                # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w_r, ray_d)                      # [b*v, 3, h*w]
    ray_d = torch.nn.functional.normalize(ray_d, dim=1)  # [b*v, 3, h*w]

    ray_d = ray_d.reshape(b,v,3,h,w)                     # [b, v, 3, h, w]
    ray_o = c2w_t.view(b,v,3,1,1).expand(b,v,3,h,w)      # [b, v, 3, h, w]

    return ray_o, ray_d


def simple_concat(ray_o, ray_d):
    return torch.cat([ray_o, ray_d], dim=2)


def compute_plucker(ray_o, ray_d):
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    plucker = torch.cat([o_cross_d, ray_d], dim=2)
    return plucker


def compute_plucker_aug(ray_o, ray_d):
    o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
    nearest_pts = ray_o + o_dot_d * ray_d
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    plucker = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
    return plucker


def compute_plucker_custom(ray_o, ray_d):
    o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
    nearest_pts = ray_o + o_dot_d * ray_d
    plucker = torch.cat([ray_d, nearest_pts], dim=2)
    return plucker
