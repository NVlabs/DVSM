import torch

from .modules import checkpoint_wrapper, MLP, SwiGluMLP, IntraViewAttention, CrossViewAttention
from .ttb import FastWeightGluMLPMultiheadTTB, TTBOperator

'''
Basic layers
    I: IntraViewAttention
    C: CrossViewAttention
    M: MLP
    TB: TTB (Test-Time Baking)

Block types
    IM
    CM
    ICM
    IMCM   (DVSM: IntraView + MLP + CrossView + MLP)
    CMIM
    CMTB   (CrossView + MLP + TTB)
    IMCMTB (TTB: IntraView + MLP + CrossView + MLP + TTB)
    IMTB   (TTB--: IntraView + MLP + TTB, dual-stream recon, ttb_feedback controls apply to x_img)
'''


def _build_ttb_op_order(total_tokens, update, apply, num_steps=1):
    if num_steps > 1 and update:
        # Multi-step: repeat full-batch update, apply only on last step
        ops = [TTBOperator(start=0, end=total_tokens, update=True, apply=False)
               for _ in range(num_steps - 1)]
        ops.append(TTBOperator(start=0, end=total_tokens, update=True, apply=apply))
        return ops
    return [TTBOperator(start=0, end=total_tokens, update=update, apply=apply)]


class BlockBase(torch.nn.Module):
    def __init__(self,
        dim,
        head_dim,
        bias=False,
        use_qk_norm=True,
        use_gating=False,
        expand_ratio=4,
        use_swiglumlp=False,
        use_sandwich=False,
        **kwargs,  # ignore TTB/fast-weight kwargs forwarded to base blocks
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.bias = bias
        self.use_qk_norm = use_qk_norm
        self.use_gating = use_gating
        self.use_sandwich = use_sandwich

        self.norm_mlp = torch.nn.LayerNorm(dim, bias=bias)
        if use_swiglumlp:
            self.mlp = SwiGluMLP(dim=dim, expand_ratio=expand_ratio, bias=bias)
        else:
            self.mlp = MLP(dim=dim, expand_ratio=expand_ratio, bias=bias)

    @property
    def num_residual(self):
        return 1

    def fw_attn(self, x, kv):
        raise NotImplementedError

    def fw_mlp(self, x):
        x = x + self.mlp(self.norm_mlp(x))
        return x

    def apply_compile(self):
        self.fw_attn = torch.compile(self.fw_attn, dynamic=True)
        self.fw_mlp = torch.compile(self.fw_mlp, dynamic=True)

    def apply_checkpoint(self):
        self.forward = checkpoint_wrapper(self)

    def forward(self, x, kv=None):
        x, kv = self.fw_attn(x, kv)
        x = self.fw_mlp(x)
        return x, kv


class BlockIM(BlockBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = torch.nn.LayerNorm(self.dim, bias=self.bias)
        self.attn = IntraViewAttention(
            dim=self.dim,
            head_dim=self.head_dim,
            bias=self.bias,
            use_qk_norm=self.use_qk_norm,
            use_gating=self.use_gating,
            use_sandwich=self.use_sandwich)

    @property
    def num_residual(self):
        return 2

    def init_residual(self, std):
        torch.nn.init.normal_(self.attn.fc.weight, std=std)
        torch.nn.init.normal_(self.mlp.mlp[-1].weight, std=std)

    def fw_attn(self, x, kv=None):
        B, V, L, D = x.shape
        x = x.view(B*V, L, D)
        x = x + self.attn(self.norm1(x))
        x = x.view(B, V, L, D)
        return x, kv


class BlockCM(BlockBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = torch.nn.LayerNorm(self.dim, bias=self.bias)
        self.attn = CrossViewAttention(
            dim=self.dim,
            head_dim=self.head_dim,
            bias=self.bias,
            use_qk_norm=self.use_qk_norm,
            use_gating=self.use_gating,
            use_sandwich=self.use_sandwich)

    @property
    def num_residual(self):
        return 2

    def init_residual(self, std):
        torch.nn.init.normal_(self.attn.fc.weight, std=std)
        torch.nn.init.normal_(self.mlp.mlp[-1].weight, std=std)

    def fw_attn(self, x, kv=None):
        B, V, L, D = x.shape
        x = x.view(B, V*L, D)
        skip, kv = self.attn(self.norm1(x), kv=kv)
        x = x + skip
        x = x.view(B, V, L, D)
        return x, kv


class BlockICM(BlockBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = torch.nn.LayerNorm(self.dim, bias=self.bias)
        self.attn_intra = IntraViewAttention(
            dim=self.dim,
            head_dim=self.head_dim,
            bias=self.bias,
            use_qk_norm=self.use_qk_norm,
            use_gating=self.use_gating,
            use_sandwich=self.use_sandwich)

        self.norm2 = torch.nn.LayerNorm(self.dim, bias=self.bias)
        self.attn_cross = CrossViewAttention(
            dim=self.dim,
            head_dim=self.head_dim,
            bias=self.bias,
            use_qk_norm=self.use_qk_norm,
            use_gating=self.use_gating,
            use_sandwich=self.use_sandwich)

    @property
    def num_residual(self):
        return 3

    def init_residual(self, std):
        torch.nn.init.normal_(self.attn_intra.fc.weight, std=std)
        torch.nn.init.normal_(self.attn_cross.fc.weight, std=std)
        torch.nn.init.normal_(self.mlp.mlp[-1].weight, std=std)

    def fw_attn(self, x, kv=None):
        B, V, L, D = x.shape
        x = x.view(B*V, L, D)
        x = x + self.attn_intra(self.norm1(x))
        x = x.view(B, V*L, D)
        skip, kv = self.attn_cross(self.norm2(x), kv=kv)
        x = x + skip
        x = x.view(B, V, L, D)
        return x, kv


class BlockSequence(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.blklst = None

    @property
    def num_residual(self):
        return sum(blk.num_residual for blk in self.blklst)

    def init_residual(self, std):
        for blk in self.blklst:
            blk.init_residual(std)

    def apply_compile(self):
        for blk in self.blklst:
            blk.apply_compile()

    def apply_checkpoint(self):
        for blk in self.blklst:
            blk.apply_checkpoint()

    def forward(self, x, kv=None):
        for blk in self.blklst:
            x, kv = blk(x, kv)
        return x, kv

class BlockIMCM(BlockSequence):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.blklst = torch.nn.ModuleList([
            BlockIM(*args, **kwargs),
            BlockCM(*args, **kwargs)])

class BlockCMIM(BlockSequence):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.blklst = torch.nn.ModuleList([
            BlockCM(*args, **kwargs),
            BlockIM(*args, **kwargs)])


class BlockTBBase(BlockBase):
    """Shared base for TTB blocks.

    Owns the TTB parameter signature, builds the SwiGLU fast-weight module from
    the block-level ttb_* kwargs, and implements the (identical) rendering
    branch. Subclasses implement only the reconstruction `forward`.
    """
    def __init__(self, *args, ttb_feedback=False,
                 ttb_use_gating=False, ttb_head_dim=0, ttb_inter_multi=2,
                 ttb_base_lr=0.01, ttb_muon_steps=5, ttb_num_steps=1,
                 ttb_target_residual=True, ttb_pre_cm_target=False,
                 ttb_use_mse_loss=False,
                 ttb_share_qk_proj=False,
                 ttb_per_step_norm=False,
                 inner_grad_checkpoint=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ttb_feedback = ttb_feedback
        self.ttb_num_steps = ttb_num_steps
        self.ttb_target_residual = ttb_target_residual
        self.ttb_pre_cm_target = ttb_pre_cm_target

        self.norm_ttb = torch.nn.LayerNorm(self.dim, bias=self.bias)
        self.ttb = FastWeightGluMLPMultiheadTTB(
            dim=self.dim, head_dim=ttb_head_dim or self.dim, inter_multi=ttb_inter_multi,
            bias=self.bias, base_lr=ttb_base_lr,
            muon_update_steps=ttb_muon_steps,
            use_gating=ttb_use_gating,
            use_mse_loss=ttb_use_mse_loss,
            share_qk_proj=ttb_share_qk_proj,
            per_step_norm=ttb_per_step_norm,
            inner_grad_checkpoint=inner_grad_checkpoint)

    def apply_compile(self):
        self.forward = torch.compile(self.forward, dynamic=True)

    def _ttb_recon(self, x_cam, x_img_target, n_tokens):
        """Reconstruction-stage inner-loop update + apply.
        Bakes the mapping x_cam -> (target) into the fast weights and applies it
        to x_cam. Returns (ttb_out, ttb_states)."""
        x_cam_normed = self.norm_ttb(x_cam)
        if self.ttb_target_residual:
            ttb_v = self.norm_ttb(x_img_target) - x_cam_normed
        else:
            ttb_v = self.norm_ttb(x_img_target)
        op_order = _build_ttb_op_order(
            n_tokens, update=True, apply=True, num_steps=self.ttb_num_steps)
        return self.ttb(
            x_query=x_cam_normed, info={"ttb_op_order": op_order},
            x_key=x_cam_normed, x_value=ttb_v)

    def _render(self, x, kv):
        """Rendering-stage TTB apply only + MLP (shared by all TTB blocks)."""
        B, V, L, D = x.shape
        x = x.view(B, V*L, D)
        info = {"ttb_op_order": [
            TTBOperator(start=0, end=V*L, update=False, apply=True)
        ]}
        info.update(kv)
        ttb_out, _ = self.ttb(x_query=self.norm_ttb(x), info=info)
        x = x + ttb_out
        x = x + self.mlp(self.norm_mlp(x))
        return x.view(B, V, L, D), kv


class BlockCMTB(BlockTBBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cross-view attention (produces the bake target for the TTB module)
        self.norm1 = torch.nn.LayerNorm(self.dim, bias=self.bias)
        self.attn = CrossViewAttention(
            dim=self.dim,
            head_dim=self.head_dim,
            bias=self.bias,
            use_qk_norm=self.use_qk_norm,
            use_gating=self.use_gating,
            use_sandwich=self.use_sandwich)

    @property
    def num_residual(self):
        return 2  # x_img: cross_attn + mlp; x_cam: ttb + mlp

    def init_residual(self, std):
        torch.nn.init.normal_(self.attn.fc.weight, std=std)
        torch.nn.init.normal_(self.mlp.mlp[-1].weight, std=std)
        torch.nn.init.normal_(self.ttb.c_proj.weight, std=std)

    def forward(self, x, kv=None):
        if kv is not None:
            return self._render(x, kv)

        # ===== Reconstruction: x = [x_img; x_cam] with 2V views =====
        B, V, L, D = x.shape
        assert V % 2 == 0
        V2 = V // 2
        x_img = x[:, :V2].reshape(B, V2*L, D)
        x_cam = x[:, V2:].reshape(B, V2*L, D)

        # Cross-view attention on x_img
        attn_delta, _ = self.attn(self.norm1(x_img), kv=None)
        x_img = x_img + attn_delta

        # Optionally use post-attn pre-MLP x_img as the TTB target
        x_img_target = x_img if self.ttb_pre_cm_target else None

        # MLP on x_img
        x_img = x_img + self.mlp(self.norm_mlp(x_img))
        if x_img_target is None:
            x_img_target = x_img

        # TTB update + apply on x_cam
        ttb_out, ttb_states = self._ttb_recon(x_cam, x_img_target, V2*L)
        x_cam = x_cam + ttb_out
        if self.ttb_feedback:
            x_img = x_img + ttb_out
        x_cam = x_cam + self.mlp(self.norm_mlp(x_cam))

        x = torch.cat([x_img.view(B, V2, L, D),
                       x_cam.view(B, V2, L, D)], dim=1)
        return x, ttb_states


class BlockIMCMTB(BlockSequence):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # BlockIM ignores the extra TTB kwargs; BlockCMTB consumes them.
        self.blklst = torch.nn.ModuleList([
            BlockIM(*args, **kwargs),
            BlockCMTB(*args, **kwargs)])

    def apply_checkpoint(self):
        self.forward = checkpoint_wrapper(self)


class BlockTB(BlockTBBase):
    @property
    def num_residual(self):
        return 2  # ttb + mlp

    def init_residual(self, std):
        torch.nn.init.normal_(self.ttb.c_proj.weight, std=std)
        torch.nn.init.normal_(self.mlp.mlp[-1].weight, std=std)

    def forward(self, x, kv=None):
        if kv is not None:
            return self._render(x, kv)

        # ===== Reconstruction: x = [x_img; x_cam] with 2V views =====
        B, V, L, D = x.shape
        assert V % 2 == 0
        V2 = V // 2
        x_img = x[:, :V2].reshape(B, V2*L, D)
        x_cam = x[:, V2:].reshape(B, V2*L, D)

        # TTB update + apply (no cross-view attention; target is x_img directly)
        ttb_out, ttb_states = self._ttb_recon(x_cam, x_img, V2*L)
        x_cam = x_cam + ttb_out
        if self.ttb_feedback:
            x_img = x_img + ttb_out

        # MLP
        x_cam = x_cam + self.mlp(self.norm_mlp(x_cam))
        x_img = x_img + self.mlp(self.norm_mlp(x_img))

        # Re-concatenate [x_img; x_cam]
        x = torch.cat([x_img.view(B, V2, L, D),
                       x_cam.view(B, V2, L, D)], dim=1)
        return x, ttb_states


class BlockIMTB(BlockSequence):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # BlockIM ignores the extra TTB kwargs; BlockTB consumes them.
        self.blklst = torch.nn.ModuleList([
            BlockIM(*args, **kwargs),
            BlockTB(*args, **kwargs)])

    def apply_checkpoint(self):
        self.forward = checkpoint_wrapper(self)
