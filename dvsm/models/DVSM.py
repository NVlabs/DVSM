import os
import math
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, snapshot_download

from .pose_preprocessor import PosePreprocessor
from .plucker import PluckerEncoding
from .backbones import DINO, RADIO
from .blocks import BlockCM, BlockICM, BlockIMCM, BlockCMIM, BlockCMTB, BlockIMCMTB, BlockIMTB


def _pose_features(c2w_r, c2w_t):
    """6D per-view feature: camera position ⊕ central-axis ray direction."""
    forward_axis = c2w_r[..., :, 2]  # camera z-axis in world coords
    return torch.cat([c2w_t, forward_axis], dim=-1)  # [B, V, 6]


@torch.no_grad()
def _batched_kmeans(x, K, n_iter=20):
    """Per-batch k-means on points x: [B, V, D] → centroids [B, K, D], labels [B, V].

    Initializes centroids by spreading them as `K` evenly-spaced views in the input
    sequence (good enough proxy for farthest-point init at small K). Falls back to
    closest-point assignment if a centroid drifts to an empty cluster.
    """
    B, V, D = x.shape
    if V <= K:
        # Each view its own chunk; pad missing chunks by repeating the last view's centroid.
        idx = torch.arange(K, device=x.device).clamp(max=V - 1)
        centroids = x[:, idx, :].clone()
        labels = torch.arange(V, device=x.device).unsqueeze(0).expand(B, -1).clamp(max=K - 1)
        return centroids, labels

    # Even-spaced init across the view sequence
    init_idx = torch.linspace(0, V - 1, K, device=x.device).round().long()
    centroids = x[:, init_idx, :].clone()  # [B, K, D]

    for _ in range(n_iter):
        # Assign each view to nearest centroid
        # dists: [B, V, K] = ||x - c||²
        dists = (x.unsqueeze(2) - centroids.unsqueeze(1)).pow(2).sum(-1)
        labels = dists.argmin(-1)  # [B, V]

        # Update: centroid_k = mean of points with label k (per batch)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=K).to(x.dtype)  # [B, V, K]
        counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, K, 1]
        new_centroids = torch.einsum('bvk,bvd->bkd', one_hot, x) / counts

        # Empty-cluster guard: if a chunk got 0 points, keep its old centroid
        empty = (one_hot.sum(dim=1) == 0).unsqueeze(-1)  # [B, K, 1]
        centroids = torch.where(empty, centroids, new_centroids)

    # Final assignment with updated centroids
    dists = (x.unsqueeze(2) - centroids.unsqueeze(1)).pow(2).sum(-1)
    labels = dists.argmin(-1)
    return centroids, labels


class DecoderViewSynthesisModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 patch_size=8,
                 latent_dim=768,
                 pose_enc='plucker',
                 pose_avg_strategy='down',
                 render_up_scale=0.,
                 recon_up_scale=0.,
                 auto_divisible_pad=False,

                 backbone='',
                 freeze_backbone=False,

                 num_register=0,

                 block_type='IMCM',
                 num_layers=12,
                 head_dim=64,
                 expand_ratio=4,
                 use_bias=False,
                 use_qk_norm=True,
                 use_gating=False,
                 use_swiglumlp=False,
                 use_sandwich=False,
                 ttb_feedback=True,
                 ttb_use_gating=True,
                 ttb_head_dim=0,
                 ttb_inter_multi=4.0,
                 ttb_base_lr=0.01,
                 ttb_muon_steps=5,
                 ttb_num_steps=3,
                 ttb_target_residual=True,
                 ttb_pre_cm_target=True,
                 ttb_use_mse_loss=True,
                 ttb_per_step_norm=False,
                 ttb_share_qk_proj=True,
                 ttb_per_cluster_K=0,
                 ttb_per_cluster_boundary_ratio=0.7,
                 use_amp=True,
                 use_compile=True,
                 grad_checkpoint=True,
                 **block_extra_kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.plucker_enc = PluckerEncoding(pose_enc)
        self.pose_avg_strategy = pose_avg_strategy
        self.render_up_scale = render_up_scale
        self.recon_up_scale = recon_up_scale
        self.auto_divisible_pad = auto_divisible_pad
        self.ttb_per_cluster_K = int(ttb_per_cluster_K)
        self.ttb_per_cluster_boundary_ratio = float(ttb_per_cluster_boundary_ratio)
        self._in_per_cluster_call = False

        if use_amp:
            self.amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.amp_dtype = None

        # Input layers
        self.inp_cam_embd = nn.Sequential(
            Rearrange("b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)", ph=self.patch_size, pw=self.patch_size),
            nn.Linear(self.plucker_enc.dim * (self.patch_size**2), latent_dim, bias=False))
        self.inp_img_embd = nn.Sequential(
            Rearrange("b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)", ph=self.patch_size, pw=self.patch_size),
            nn.Linear(3 * (self.patch_size**2), latent_dim, bias=False))
        self.inp_norm = nn.LayerNorm(latent_dim, bias=False)

        if num_register > 0:
            self.register_tokens = torch.nn.Parameter(torch.randn(1,1,num_register,self.latent_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        # Compose a series of repeated blocks
        if block_type == 'CM':
            BLOCK = BlockCM
        elif block_type == 'ICM':
            BLOCK = BlockICM
        elif block_type == 'IMCM':
            BLOCK = BlockIMCM
        elif block_type == 'CMIM':
            BLOCK = BlockCMIM
        elif block_type == 'CMTB':
            BLOCK = BlockCMTB
        elif block_type == 'IMCMTB':
            BLOCK = BlockIMCMTB
        elif block_type == 'IMTB':
            BLOCK = BlockIMTB
        else:
            raise NotImplementedError(f"Unknown block_type={block_type}")

        block_kwargs = dict(
            dim=latent_dim,
            head_dim=head_dim,
            bias=use_bias,
            use_qk_norm=use_qk_norm,
            use_gating=use_gating,
            expand_ratio=expand_ratio,
            use_swiglumlp=use_swiglumlp,
            use_sandwich=use_sandwich,
            ttb_feedback=ttb_feedback,
            ttb_use_gating=ttb_use_gating,
            ttb_head_dim=ttb_head_dim,
            ttb_inter_multi=ttb_inter_multi,
            ttb_base_lr=ttb_base_lr,
            ttb_muon_steps=ttb_muon_steps,
            ttb_num_steps=ttb_num_steps,
            ttb_target_residual=ttb_target_residual,
            ttb_pre_cm_target=ttb_pre_cm_target,
            ttb_use_mse_loss=ttb_use_mse_loss,
            ttb_per_step_norm=ttb_per_step_norm,
            ttb_share_qk_proj=ttb_share_qk_proj,
            **block_extra_kwargs,
        )
        self.blocks = nn.ModuleList([BLOCK(**block_kwargs) for _ in range(num_layers)])
        self.use_dual_stream = block_type in ('CMTB', 'IMCMTB', 'IMTB')

        # Output layers
        self.rgb_head = nn.Sequential(
            nn.LayerNorm(latent_dim, bias=False),
            nn.Linear(latent_dim, (self.patch_size**2) * 3, bias=False),
            nn.Sigmoid())

        # Initialize all linear layers
        for name, m in self.named_modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # Initialize the output projection of a residual
        special_std = 0.02 / math.sqrt(sum(blk.num_residual for blk in self.blocks))
        for blk in self.blocks:
            blk.init_residual(special_std)

        # Apply torch.compile
        if use_compile:
            self.rgb_head.forward = torch.compile(self.rgb_head.forward, dynamic=True)
            for blk in self.blocks:
                blk.apply_compile()

        # Apply activation checkpointing
        if grad_checkpoint:
            for blk in self.blocks:
                blk.apply_checkpoint()

        # Use pre-trained foundation feature extractor
        if 'dino' in backbone:
            self.backbone = DINO(backbone, latent_dim, grad_checkpoint, use_compile)
        elif 'radio' in backbone:
            self.backbone = RADIO(backbone, latent_dim, grad_checkpoint, use_compile)
        else:
            assert backbone == '', f"Unknown backbone model: {backbone}"
            self.backbone = None

        if self.backbone is not None:
            assert self.patch_size == self.backbone.patch_size, "Inconsistent patch size"
            for param in self.backbone.parameters():
                if freeze_backbone:
                    param.requires_grad = False
                else:
                    param.use_lr_scaler = True

    def forward(self, image, intr, c2w_r, c2w_t, pose_preproc_done=False):
        ''' Process images and return a rendering function for the input scenes. '''

        # Per-cluster boundary-adaptive short-circuit: K independent V_sub
        # sub-forwards each running on its own k-means cluster (local pose
        # norm + per-cluster cross-attn). See PerClusterRenderer for details.
        if self.ttb_per_cluster_K > 0 and not self._in_per_cluster_call:
            # K>0 enables eval-only per-cluster mode. Cap at V (more clusters
            # than views is meaningless) and skip when K=1 (degenerate to
            # single-cluster = baseline single forward).
            K = min(self.ttb_per_cluster_K, image.shape[1])
            if K > 1:
                return PerClusterRenderer(
                    self, image, intr, c2w_r, c2w_t, K=K,
                    boundary_ratio=self.ttb_per_cluster_boundary_ratio,
                )

        # Normalize camera poses
        if pose_preproc_done:
            pose_preprocessor = None
        else:
            pose_preprocessor = PosePreprocessor(
                c2w_r, c2w_t, pose_avg_strategy=self.pose_avg_strategy)
            c2w_r, c2w_t = pose_preprocessor(c2w_r, c2w_t)

        # Process input images
        if self.recon_up_scale > 1:
            B, V = image.shape[:2]
            image = torch.nn.functional.interpolate(
                image.flatten(0,1),
                scale_factor=self.recon_up_scale, mode='bilinear').unflatten(0,(B,V))

        B, V, _, H, W = image.shape
        plucker_map = self.plucker_enc(intr, c2w_r, c2w_t, H, W)

        # TODO: camera param may need to update for PRope? Move it to data loader?
        if self.auto_divisible_pad:
            padH = (self.patch_size - (H % self.patch_size)) % self.patch_size
            padW = (self.patch_size - (W % self.patch_size)) % self.patch_size
            if padH > 0 or padW > 0:
                image = torch.nn.functional.pad(image, (0, padW, 0, padH))
                plucker_map = torch.nn.functional.pad(plucker_map, (0, padW, 0, padH))

        # Running the model
        with torch.autocast(enabled=self.amp_dtype is not None, device_type="cuda", dtype=self.amp_dtype):

            # Input tokenization
            x_cam_embd = self.inp_cam_embd(plucker_map)
            x_img_embd = self.inp_img_embd(image)
            x = x_cam_embd + x_img_embd

            # Foundation prior feature injection
            if self.backbone is not None:
                x = x + self.backbone(image)

            # Append register tokens
            if self.register_tokens is not None:
                register_tokens = self.register_tokens.repeat(B,V,1,1)
                x = torch.cat([register_tokens, x], dim=2)

            # Input token normalization
            x = self.inp_norm(x)

            # Transformer feed-forward
            kv_lst = []
            if self.use_dual_stream:
                # Dual-stream: prepare cam-only tokens and concatenate along V
                if self.register_tokens is not None:
                    x_cam = torch.cat([register_tokens, x_cam_embd], dim=2)
                else:
                    x_cam = x_cam_embd
                x_cam = self.inp_norm(x_cam)
                x = torch.cat([x, x_cam], dim=1)  # [B, 2V, L, D]

            for block in self.blocks:
                x, kv = block(x)
                kv_lst.append(kv)

        return NeuralRenderer(pose_preprocessor, self, kv_lst)

    @classmethod
    def from_hf_pretrained(cls, variant, repo_id="nvidia/DVSM"):
        local_dir = snapshot_download(repo_id=repo_id, allow_patterns=os.path.join(variant, "*"))
        model = cls.from_pretrained(os.path.join(local_dir, variant))
        return model


class PerClusterRenderer:
    def __init__(self, model, image, intr, c2w_r, c2w_t, K, boundary_ratio):
        self.model = model
        pose_feat = _pose_features(c2w_r, c2w_t)                          # [B, V, D]
        B, V, _ = pose_feat.shape
        K_eff = min(K, V)
        centroids, _ = _batched_kmeans(pose_feat, K_eff)                  # [B, K, D]
        dists_sq = (pose_feat.unsqueeze(2) - centroids.unsqueeze(1)).pow(2).sum(-1)  # [B, V, K]

        # Per-view membership: primary (always) + secondary when boundary.
        top2 = dists_sq.topk(min(2, K_eff), dim=-1, largest=False)
        primary = top2.indices[..., 0]                                    # [B, V]
        membership = torch.zeros(B, V, K_eff, dtype=torch.bool, device=pose_feat.device)
        membership.scatter_(2, primary.unsqueeze(-1), True)
        if K_eff >= 2 and boundary_ratio > 0:
            d1_sq = top2.values[..., 0]
            d2_sq = top2.values[..., 1]
            secondary = top2.indices[..., 1]
            # Squared form of d1/d2 > boundary_ratio.
            is_boundary = (d1_sq / d2_sq.clamp(min=1e-12)) > (boundary_ratio ** 2)
            sec_mask = torch.zeros_like(membership)
            sec_mask.scatter_(2, secondary.unsqueeze(-1), True)
            membership = membership | (sec_mask & is_boundary.unsqueeze(-1))

        self.cluster_centroids = centroids
        self.cluster_renderers = []
        assert B == 1, "PerClusterRenderer is implemented for B=1 (use --rend_bs 1)"
        was_in = model._in_per_cluster_call
        model._in_per_cluster_call = True
        try:
            for k in range(K_eff):
                idx_k = membership[0, :, k].nonzero(as_tuple=True)[0]      # variable size
                sub_renderer = model(
                    image[:, idx_k].to(intr.device),
                    intr[:, idx_k], c2w_r[:, idx_k], c2w_t[:, idx_k],
                )
                self.cluster_renderers.append(sub_renderer)
        finally:
            model._in_per_cluster_call = was_in

    def __call__(self, intr, c2w_r, c2w_t, H, W, ts=None):
        B, T = intr.shape[:2]
        assert B == 1, "PerClusterRenderer expects B=1 (use --rend_bs 1)"
        tgt_pose = _pose_features(c2w_r, c2w_t)
        d = (tgt_pose.unsqueeze(2) - self.cluster_centroids.unsqueeze(1)).pow(2).sum(-1)
        cluster_idx = d.argmin(dim=-1)                                    # [B, T]
        outputs = []
        for t in range(T):
            k = int(cluster_idx[0, t].item())
            out_t = self.cluster_renderers[k](
                intr[:, t:t+1], c2w_r[:, t:t+1], c2w_t[:, t:t+1], H, W, ts)
            outputs.append(out_t)
        return torch.cat(outputs, dim=1)


class NeuralRenderer:
    def __init__(self, pose_preprocessor, model, kv_lst):
        self.pose_preprocessor = pose_preprocessor
        self.model = model
        self.kv_lst = kv_lst
        assert len(self.model.blocks) == len(self.kv_lst)

    def __call__(self, intr, c2w_r, c2w_t, H, W, ts=None):

        # Process poses
        if self.pose_preprocessor is not None:
            c2w_r, c2w_t = self.pose_preprocessor(c2w_r, c2w_t)

        # Process input images
        if self.model.render_up_scale > 1:
            H_, W_ = H, W
            H = round(H * self.model.render_up_scale)
            W = round(W * self.model.render_up_scale)

        B, V = intr.shape[:2]
        plucker_map = self.model.plucker_enc(intr, c2w_r, c2w_t, H, W)

        padH, padW = 0, 0
        if self.model.auto_divisible_pad:
            padH = (self.model.patch_size - (H % self.model.patch_size)) % self.model.patch_size
            padW = (self.model.patch_size - (W % self.model.patch_size)) % self.model.patch_size
            if padH > 0 or padW > 0:
                plucker_map = torch.nn.functional.pad(plucker_map, (0, padW, 0, padH))

        # Run renderer
        with torch.autocast(enabled=self.model.amp_dtype is not None, device_type="cuda", dtype=self.model.amp_dtype):

            # Input tokenization
            x = self.model.inp_cam_embd(plucker_map)

            # Append register tokens
            if self.model.register_tokens is not None:
                register_tokens = self.model.register_tokens.repeat(B,V,1,1)
                x = torch.cat([register_tokens, x], dim=2)

            # Input token normalization
            x = self.model.inp_norm(x)

            # Transformer feed-forward
            for block, kv in zip(self.model.blocks, self.kv_lst):
                x, _ = block(x, kv=kv)

            # Remove register tokens
            if self.model.register_tokens is not None:
                x = x[:, :, register_tokens.shape[2]:]

            # Predict rgb
            image = self.model.rgb_head(x)
            image = rearrange(
                image,
                "b v (hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
                hh=(H+padH) // self.model.patch_size,
                ww=(W+padW) // self.model.patch_size,
                ph=self.model.patch_size,
                pw=self.model.patch_size,
                c=3)

            if self.model.auto_divisible_pad and (padH > 0 or padW > 0):
                image = image[..., :H, :W]

            if self.model.render_up_scale > 1:
                image = torch.nn.functional.interpolate(
                    image.flatten(0,1),
                    size=(H_, W_), mode='bilinear', antialias=True).unflatten(0,(B,V))

        return image


def resolve_checkpoint_path(ckpt_root, ckpt_name='latest.pt'):
    """Return the weight file inside a checkpoint directory.

    <ckpt_root>/<ckpt_name> (train.py output) if it exists, otherwise
    <ckpt_root>/model.safetensors (Hugging Face release).
    """
    for name in (ckpt_name, 'model.safetensors'):
        path = os.path.join(ckpt_root, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_root}: expected {ckpt_name} or model.safetensors")


def load_checkpoint(model, ckpt_path, strict=True):
    """Load weights from a train.py checkpoint (*.pt) or a Hugging Face one (*.safetensors).

    ckpt_path can also be a checkpoint directory (see resolve_checkpoint_path).
    Returns (missing_keys, unexpected_keys, iteration). The iteration is None
    when the checkpoint does not record it (Hugging Face release).
    """
    if os.path.isdir(ckpt_path):
        ckpt_path = resolve_checkpoint_path(ckpt_path)

    if str(ckpt_path).endswith('.safetensors'):
        from safetensors.torch import load_model
        missing, unexpected = load_model(model, str(ckpt_path), strict=strict, device='cpu')
        return list(missing), list(unexpected), None

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    info = model.load_state_dict(ckpt['state_dict'], strict=strict)
    return info.missing_keys, info.unexpected_keys, ckpt.get('iter')
