import torch
import flash_attn
import torch.utils.checkpoint


def checkpoint_wrapper(module):
    # Wrap model forward function to automatically use activation checkpointing
    ori_forward = module.forward
    def new_forward(*args, **kwargs):
        if module.training:
            output = torch.utils.checkpoint.checkpoint(
                ori_forward, *args, **kwargs, use_reentrant=False)
        else:
            output = ori_forward(*args, **kwargs)
        return output
    return new_forward


class Contiguous(torch.nn.Module):
    def forward(self, x):
        return x.contiguous()


class MLP(torch.nn.Module):
    def __init__(self, dim, expand_ratio=4, bias=False, activation=torch.nn.GELU):
        """
        Args:
            dim: Input dimension
            expand_ratio: Multiplier for hidden dimension
            bias: Whether to use bias in linear layers
            activation: Activation function
        """
        super().__init__()
        hidden_dim = round(dim * expand_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=bias),
            activation(),
            torch.nn.Linear(hidden_dim, dim, bias=bias))

    def forward(self, x):
        return self.mlp(x)


class SwiGluMLP(torch.nn.Module):
    def __init__(self, dim, expand_ratio=4, bias=False):
        """
        Args:
            dim: Input dimension
            expand_ratio: Multiplier for hidden dimension
            bias: Whether to use bias in linear layers
            activation: Activation function
        """
        super().__init__()
        hidden_dim = round(dim * expand_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=bias),
            torch.nn.Linear(dim, hidden_dim, bias=bias),
            torch.nn.Linear(hidden_dim, dim, bias=bias))

    def forward(self, x):
        gate = torch.nn.functional.silu(self.mlp[0](x), inplace=True)
        return self.mlp[2](gate * self.mlp[1](x))


class IntraViewAttention(torch.nn.Module):
    def __init__(self,
        dim,
        head_dim,
        bias=False,
        use_qk_norm=True,
        use_gating=False,
        use_sandwich=False,
        out_dim=None,
    ):
        """
        dim: Input dimension
        head_dim: Dimension of each attention head
        bias: Whether to use bias in QKV and output projection
        use_qk_norm: Whether to use Q-K normalization
        use_gating: Whether to apply gating before output ffn
        """
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} should be divisible by head_dim={head_dim}"
        num_head = dim // head_dim

        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.head_dim = head_dim
        self.num_head = num_head
        self.use_qk_norm = use_qk_norm
        self.use_gating = use_gating
        self.use_sandwich = use_sandwich
        self.attn_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.to_qkv = torch.nn.Linear(dim, 3 * dim, bias=bias)
        self.fc = torch.nn.Linear(dim, self.out_dim, bias=bias)

        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = torch.nn.RMSNorm(head_dim, eps=1e-5)
            self.k_norm = torch.nn.RMSNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

        # Optional gating
        if self.use_gating:
            self.gate = torch.nn.Sequential(
                torch.nn.Linear(dim, dim, bias=False),
                torch.nn.Sigmoid())
        # Optional normalization
        if self.use_sandwich:
            self.sandwich = torch.nn.LayerNorm(dim, bias=False)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        # Predict qkv packed as (batch, seq_len, 3, num_head, head_dim)
        qkv = self.to_qkv(x).unflatten(2, (3, self.num_head, self.head_dim))

        # Unpack (q, k, v) each as (batch, seq_len, num_head, head_dim)
        q, k, v = qkv.moveaxis(2,0).contiguous()

        # Apply qk normalization if enabled
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention
        with torch.autocast(enabled=False, device_type="cuda"):
            q = q.to(self.attn_dtype)
            k = k.to(self.attn_dtype)
            v = v.to(self.attn_dtype)

        out = flash_attn.flash_attn_func(q, k, v)

        # Output layer to yield (batch, seq_len, dim)
        out = out.flatten(2,3)
        if self.use_gating:
            out = out * self.gate(x)
        out = self.fc(out)
        if self.use_sandwich:
            out = self.sandwich(out)

        return out


class CrossViewAttention(torch.nn.Module):
    def __init__(self,
        dim,
        head_dim,
        bias=False,
        use_qk_norm=True,
        use_gating=False,
        use_sandwich=False,
        out_dim=None,
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            bias: Whether to use bias in QKV and output projection
            use_qk_norm: Whether to use Q-K normalization
            use_gating: Whether to apply gating before output ffn
        """
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} should be divisible by head_dim={head_dim}"
        num_head = dim // head_dim

        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.head_dim = head_dim
        self.num_head = num_head
        self.use_qk_norm = use_qk_norm
        self.use_gating = use_gating
        self.use_sandwich = use_sandwich
        self.attn_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.to_q = torch.nn.Linear(dim, dim, bias=bias)
        self.to_kv = torch.nn.Linear(dim, 2 * dim, bias=bias)
        self.fc = torch.nn.Linear(dim, self.out_dim, bias=bias)

        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = torch.nn.RMSNorm(head_dim, eps=1e-5)
            self.k_norm = torch.nn.RMSNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

        # Optional gating
        if self.use_gating:
            self.gate = torch.nn.Sequential(
                torch.nn.Linear(dim, dim, bias=False),
                torch.nn.Sigmoid())
        # Optional normalization
        if self.use_sandwich:
            self.sandwich = torch.nn.LayerNorm(dim, bias=False)

    def forward(self, x, kv=None):
        """
        x: [B, L, D]
        kv: ([B, M, D], [B, M, D]) If provided, x query on kv instead of doing self attention
        """
        q = self.to_q(x)

        # To shape of (batch, seq_len, num_head, head_dim)
        q = q.unflatten(2, (self.num_head, self.head_dim))
        q = self.q_norm(q)
        with torch.autocast(enabled=False, device_type="cuda"):
            q = q.to(self.attn_dtype)

        # Prepare (k, v)
        if kv is None:
            kv = self.to_kv(x)
            kv = kv.unflatten(2, (2, self.num_head, self.head_dim))
            k, v = kv.moveaxis(2, 0).contiguous()
            k = self.k_norm(k)
            with torch.autocast(enabled=False, device_type="cuda"):
                k = k.to(self.attn_dtype)
                v = v.to(self.attn_dtype)
        else:
            k, v = kv
            assert len(x) == len(k) and len(x) == len(v)

        # Attention
        out = flash_attn.flash_attn_func(q, k, v)

        # Output layer into (batch, seq_len, dim)
        out = out.flatten(2, 3)
        if self.use_gating:
            out = out * self.gate(x)
        out = self.fc(out)
        if self.use_sandwich:
            out = self.sandwich(out)

        return out, (k, v)
