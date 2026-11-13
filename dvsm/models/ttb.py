import collections
import math

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

TTBOperator = collections.namedtuple("TTBOperator", ["start", "end", "update", "apply"])

@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


def inner_loop(w0_now, w1_now, w2_now, ki, vi, lr0i, lr1i, lr2i,
               use_mse,
               muon_update_steps, n_tokens,
               use_weight_norm, w0_norm, w1_norm, w2_norm):

    gate_before_act = ki @ w0_now       # [b, l, dh] = [b, l, d] @ [b, d, dh]
    hidden_before_mul = ki @ w2_now     # [b, l, dh] = [b, l, d] @ [b, d, dh]
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    if use_mse:
        # MSE: gradient descent on ||f_W(k) - v||²
        output_i = hidden @ w1_now          # [b, l, d] — forward prediction
        error = output_i - vi               # [b, l, d] — prediction error

        dhidden = error @ w1_now.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_raw = -(hidden * lr1i).transpose(-1, -2) @ error
        w0_raw = -(ki * lr0i).transpose(-1, -2) @ dgate_before_act
        w2_raw = -(ki * lr2i).transpose(-1, -2) @ dhidden_before_mul
    else:
        # Hebbian: gradient ascent on <f_W(k), v>
        dhidden = vi @ w1_now.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_raw = (hidden * lr1i).transpose(-1, -2) @ vi
        w0_raw = (ki * lr0i).transpose(-1, -2) @ dgate_before_act
        w2_raw = (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul

    if muon_update_steps > 0:
        w0_grad = zeropower_via_newtonschulz5(w0_raw, muon_update_steps)
        w1_grad = zeropower_via_newtonschulz5(w1_raw, muon_update_steps)
        w2_grad = zeropower_via_newtonschulz5(w2_raw, muon_update_steps)
    else:
        w0_grad = w0_raw / n_tokens
        w1_grad = w1_raw / n_tokens
        w2_grad = w2_raw / n_tokens

    w0_now = w0_now + w0_grad
    w1_now = w1_now + w1_grad
    w2_now = w2_now + w2_grad

    if use_weight_norm:
        w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
        w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
        w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

    return w0_now, w1_now, w2_now


def _fast_weight_kernel_body(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttb_ua_order: list,
    muon_update_steps: int = 0,
    use_mse: bool = False,
    per_step_norm: bool = False,
    inner_grad_checkpoint: bool = False,
):
    """
    Forward: (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    # Sanity check: all entries use all tokens, update all-or-none, apply last only
    num_update = 0
    for idx, (start, end, update, apply) in enumerate(ttb_ua_order):
        assert start == 0 and end == q.shape[1]
        num_update += int(update)
        if idx == len(ttb_ua_order) - 1:
            assert apply
        else:
            assert not apply
    assert num_update == 0 or num_update == len(ttb_ua_order)

    n_tokens = q.shape[1]

    # Inner loop fast weight update
    for idx in range(len(ttb_ua_order)):
        if ttb_ua_order[idx].update:
            loop_kwargs = dict(
                w0_now=w0, w1_now=w1, w2_now=w2,
                ki=k, vi=v, lr0i=lr0, lr1i=lr1, lr2i=lr2,
                use_mse=use_mse,
                muon_update_steps=muon_update_steps, n_tokens=n_tokens,
                use_weight_norm=per_step_norm,
                w0_norm=w0_norm, w1_norm=w1_norm, w2_norm=w2_norm)
            if inner_grad_checkpoint:
                w0, w1, w2 = torch.utils.checkpoint.checkpoint(
                    inner_loop, **loop_kwargs, use_reentrant=False)
            else:
                w0, w1, w2 = inner_loop(**loop_kwargs)

    # Apply with updated weights
    output = (F.silu(q @ w0, inplace=True) * (q @ w2)) @ w1

    return output, w0, w1, w2


fast_weight_swish_glu_weight_norm_mini_batch_apply = torch.compile(_fast_weight_kernel_body)


class FastWeightGluMLPMultiheadTTB(nn.Module):
    """TTB (Test-Time Baking) module.

    Distills cross-view attention into fast weights using separate q/k/v sources:
    - q: from post-MLP tokens (for apply/query)
    - k: from pre-cross-attn tokens (update input)
    - v: from cross-attn delta (update target)

    The fast weights learn: f_W(input) ≈ CrossViewAttn(input)
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        use_gating=False,
        use_mse_loss=False,
        share_qk_proj=False,
        per_step_norm=False,
        inner_grad_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.use_gating = use_gating
        self.per_step_norm = per_step_norm
        self.inner_grad_checkpoint = inner_grad_checkpoint
        self.use_mse_loss = use_mse_loss
        self.share_qk_proj = share_qk_proj
        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )

        # Q/K/V projections: shared or separate
        if share_qk_proj:
            self.to_qk = nn.Linear(dim, dim, bias=bias)
        else:
            self.to_q = nn.Linear(dim, dim, bias=bias)
            self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        self.o_norm = torch.nn.RMSNorm(self.dim, eps=1e-5, elementwise_affine=True)

        if use_gating:
            self.gate_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x_query: torch.Tensor, info={}, x_key=None, x_value=None):
        """
        Args:
            x_query: (b, l, d) — tokens to query fast weights (post-MLP, normed)
            info: dict with "ttb_op_order" and optionally "w0", "w1", "w2" states
            x_key:   (b, l, d) — pre-cross-attn tokens for update (None during rendering)
            x_value: (b, l, d) — cross-attn delta for update target (None during rendering)
        """
        B = x_query.shape[0]
        H = self.num_heads

        # Project query (always needed for apply)
        qk_proj = self.to_qk if self.share_qk_proj else self.to_q
        q = F.silu(qk_proj(x_query))
        q = rearrange(q, "b l (h d) -> (b h) l d", h=H)
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x_query.dtype)

        if x_key is not None:
            # Reconstruction: project key and value from their respective sources
            k_proj = self.to_qk if self.share_qk_proj else self.to_k
            k = F.silu(k_proj(x_key))
            v = F.silu(self.to_v(x_value))
            k = rearrange(k, "b l (h d) -> (b h) l d", h=H)
            v = rearrange(v, "b l (h d) -> (b h) l d", h=H)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x_key.dtype)
            v = v / (v.norm(dim=2, keepdim=True) + 1e-5).to(x_value.dtype)

            # Learning rate computed from key tokens
            with torch.autocast(device_type="cuda", enabled=False):
                lr = self.lr_fc(x_key.float())

            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

            lr0, lr1, lr2 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d",
                lrs=3, h=H
            )
        else:
            # Rendering: dummy k/v/lr (update=False, never accessed by inner loop)
            k = q.new_zeros(q.shape[0], 0, q.shape[2])
            v = k
            lr0 = lr1 = lr2 = q.new_zeros(q.shape[0], 0, 1)

        if "w0" in info:
            assert "w1" in info and "w2" in info
            sw0 = info["w0"]
            sw1 = info["w1"]
            sw2 = info["w2"]
        else:
            sw0 = self.w0.repeat(B, 1, 1)
            sw1 = self.w1.repeat(B, 1, 1)
            sw2 = self.w2.repeat(B, 1, 1)

        shared_out, sw0, sw1, sw2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            sw0, sw1, sw2, q, k, v, lr0, lr1, lr2, info["ttb_op_order"],
            muon_update_steps=self.muon_update_steps,
            use_mse=self.use_mse_loss,
            per_step_norm=self.per_step_norm,
            inner_grad_checkpoint=self.inner_grad_checkpoint,
        )

        output = shared_out

        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=H, b=B
        )
        output = self.o_norm(output)  # (b, l, dim)

        if self.use_gating:
            output = output * F.silu(self.gate_proj(output))

        out = self.c_proj(output)

        state = {"w0": sw0, "w1": sw1, "w2": sw2}
        return out, state

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")


