"""Viewer-side rendering optimizations. All wrappers; never touches dvsm/*
weights or training paths. Only the viewer process loads this module.

Optimizations (mirrored from facebookresearch/lagernvs):
  1. FastPluckerEncoding   — cache camera-frame ray dirs per (H, W); per-frame
                             cost is one batched matmul + cross product.
  2. GraphRenderer         — per-(H, W, V) CUDA Graph capture + replay around
                             a NeuralRenderer. First call at a new size warms
                             cuDNN then captures; later calls just copy inputs
                             and replay.
  3. cuDNN warmup          — n_warmup forward passes on a side stream before
                             capture (built into GraphRenderer).
  4. Async JPEG push       — see render_loop; ThreadPoolExecutor overlaps
                             viser set_background_image with the next render.
"""
from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn as nn


class FastPluckerEncoding(nn.Module):
    """Drop-in replacement for `dvsm.models.plucker.PluckerEncoding` that
    caches the camera-frame ray directions for a given (H, W) and the current
    intrinsics. Per-frame cost is then:
        ray_d = normalize(c2w_r @ dirs_local)
        plucker = pose_enc(c2w_t, ray_d)

    The cache key is (H, W). Intrinsics are assumed to stay constant within a
    single viewer session (true in viser_app/viewer.py — fov is set once from
    the first input view and the only thing changing per frame is c2w).
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        # Keep the original module around so we still delegate the plucker
        # concat (compute_plucker / compute_plucker_aug / ...) to whatever the
        # checkpoint trained against.
        self._original = original
        self.dim = original.dim
        self.pose_enc = original.pose_enc

        self._cache_lock = threading.Lock()
        # Per-(H, W) cache. Entries are kept alive for the full session so any
        # CUDA Graph that captured a kernel reading from an entry's GPU memory
        # keeps a valid pointer (the captured kernel runs at replay without
        # going through Python, so we can't invalidate per-graph references).
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    @staticmethod
    def _compute_dirs_local(intr: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Camera-frame ray directions. Depends only on (intr, h, w).
        Mirrors the first half of `dvsm.models.plucker.compute_rays`."""
        b, v = intr.shape[:2]
        fx, fy, cx, cy = intr.flatten(0, 1).T.unsqueeze(-1).contiguous()
        fx = fx * w
        fy = fy * h
        cx = cx * w
        cy = cy * h
        idx_x = torch.arange(w, device=intr.device) + 0.5
        idx_y = torch.arange(h, device=intr.device) + 0.5
        idx_x = idx_x[None, :].expand(h, -1)
        idx_y = idx_y[:, None].expand(-1, w)
        idx_x = idx_x.flatten().expand(b * v, -1)
        idx_y = idx_y.flatten().expand(b * v, -1)
        x = (idx_x - cx) / fx
        y = (idx_y - cy) / fy
        z = torch.ones_like(x)
        dirs = torch.stack([x, y, z], dim=1)  # [B*V, 3, H*W]
        return dirs.reshape(b, v, 3, h, w).contiguous()

    def forward(self, intr, c2w_r, c2w_t, h, w):
        # Cache check is python-level (no GPU sync). Inside a CUDA Graph
        # capture, this if-branch evaluates once at capture time; the cache
        # is pre-warmed before capture so we take the cache-hit path and only
        # those kernels (bmm + normalize + plucker concat) get recorded.
        with self._cache_lock:
            dirs_local = self._cache.get((h, w))
            if dirs_local is None:
                dirs_local = self._compute_dirs_local(intr, h, w)
                self._cache[(h, w)] = dirs_local

        b, v = intr.shape[:2]
        c2w_r_bv = c2w_r.reshape(b * v, 3, 3)
        ray_d = torch.bmm(c2w_r_bv, dirs_local.reshape(b * v, 3, h * w))
        ray_d = torch.nn.functional.normalize(ray_d, dim=1)
        ray_d = ray_d.reshape(b, v, 3, h, w)
        ray_o = c2w_t.view(b, v, 3, 1, 1).expand(b, v, 3, h, w)
        return self.pose_enc(ray_o, ray_d)

    def warm_cache_for(self, intr: torch.Tensor, h: int, w: int) -> None:
        """Pre-populate the cache for (H, W). Call before CUDA Graph capture
        so the cache-hit path is what gets recorded."""
        with self._cache_lock:
            if (h, w) not in self._cache:
                self._cache[(h, w)] = self._compute_dirs_local(intr, h, w)


class GraphRenderer:
    """CUDA Graph wrapper around a NeuralRenderer.

    Each unique (H, W, V) gets its own captured graph + static IO buffers.
    First call at a new size:
        warmup -> capture -> replay (slow, ~tens of ms).
    Subsequent calls at the same size:
        copy live inputs into static buffers -> replay (fast).

    Returns a `.clone()` of the static output so the caller can freely mutate
    or send it to viser without aliasing the graph's internal buffer.
    """

    def __init__(self, renderer, n_warmup: int = 5, max_graphs: int = 3):
        self.renderer = renderer
        self.n_warmup = n_warmup
        self.max_graphs = max_graphs
        self._graphs: dict[tuple, dict[str, Any]] = {}
        self._pool = torch.cuda.graph_pool_handle()
        self._lock = threading.Lock()
        # Identify the underlying FastPluckerEncoding (if installed) so we can
        # pre-warm its cache before capture. Otherwise the captured graph would
        # record the cache-miss branch the first time.
        self._fast_plucker = renderer.model.plucker_enc \
            if isinstance(renderer.model.plucker_enc, FastPluckerEncoding) else None

    # Make the wrapper look like a NeuralRenderer to downstream consumers
    # (render_loop reads `renderer.model.patch_size`, etc.).
    @property
    def model(self):
        return self.renderer.model

    def _capture(self, intr, c2w_r, c2w_t, H, W):
        static_intr = intr.clone().contiguous()
        static_r = c2w_r.clone().contiguous()
        static_t = c2w_t.clone().contiguous()

        # Pre-warm the plucker cache against the static intr — otherwise the
        # cache-miss branch (which allocates a new dirs_local on the default
        # allocator) gets captured, and the next capture at a different (H,W)
        # would re-trigger that branch with stale memory.
        if self._fast_plucker is not None:
            self._fast_plucker.warm_cache_for(static_intr, H, W)

        # cuDNN warmup on a side stream so the graph captures the optimal
        # kernel selection, not the first-time-autotune kernels.
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.n_warmup):
                    _ = self.renderer(static_intr, static_r, static_t, H, W)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                static_out = self.renderer(static_intr, static_r, static_t, H, W)

        return {
            "intr": static_intr,
            "r": static_r,
            "t": static_t,
            "out": static_out,
            "graph": graph,
        }

    def __call__(self, intr, c2w_r, c2w_t, H, W, ts=None):
        # ts is ignored — the baked render path doesn't use it.
        V = int(intr.shape[1])
        key = (int(H), int(W), V)
        with self._lock:
            entry = self._graphs.get(key)
            if entry is None:
                if len(self._graphs) >= self.max_graphs:
                    # LRU: dicts preserve insertion order, pop the oldest.
                    oldest = next(iter(self._graphs))
                    self._graphs.pop(oldest)
                entry = self._capture(intr, c2w_r, c2w_t, H, W)
                self._graphs[key] = entry

        entry["intr"].copy_(intr)
        entry["r"].copy_(c2w_r)
        entry["t"].copy_(c2w_t)
        entry["graph"].replay()
        return entry["out"].clone()


def install_optimizations(renderer, use_fast_plucker: bool = True,
                          use_cuda_graph: bool = True,
                          n_warmup: int = 5,
                          max_graphs: int = 3):
    """Apply both optimizations to a NeuralRenderer (in place + wrap). Returns
    the (possibly wrapped) renderer that the viewer should call from now on.
    Safe to call once at viewer startup; never invoked from training."""
    if use_fast_plucker:
        renderer.model.plucker_enc = FastPluckerEncoding(renderer.model.plucker_enc)
    if use_cuda_graph:
        renderer = GraphRenderer(renderer, n_warmup=n_warmup, max_graphs=max_graphs)
    return renderer
