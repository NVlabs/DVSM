"""Stage B: viser server serving real-time NVS from a pre-computed recon artifact.

Loads the model checkpoint plus the .recon.pt artifact produced by
recon_scene.py, reconstructs the NeuralRenderer in memory (no input images
needed), and exposes it over a viser web client. Each connected client gets a
dedicated render thread that produces frames in response to camera updates,
with dynamic downscale while the user is dragging the camera.

Usage:
    python -m viser_app.viewer \
        --ckpt_root outputs/dl3dv/960p_ttb_dvsm_minus \
        --recon outputs/recon/960p_ttb_dvsm_minus_scene0.recon.pt \
        --port 8080

Open the printed URL in a browser, or SSH-forward the port:
    ssh -L 8080:<jobnode>:8080 <login_node>
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import asyncio
import ssl

import numpy as np
import torch
import viser
import viser._tunnel as _viser_tunnel


# Replace viser's share-URL tunnel relay coroutine with a version that also
# catches ssl.SSLError during connection teardown. Without this, an
# "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" raised while closing a relayed SSL
# stream propagates out and kills the tunnel's multiprocessing.Process, which
# permanently breaks the share URL. _simple_proxy is referenced by module name
# from _make_tunnel, so this monkey-patch is picked up on every relay call.
# Patch must run BEFORE request_share_url() spawns the child process; under
# Linux fork start method the child inherits the patched module.
async def _patched_simple_proxy(local_host, local_port, remote_host, remote_port, close_event):
    ssl_context = ssl.create_default_context()

    async def close_writer(w):
        try:
            if not w.is_closing():
                w.close()
            await w.wait_closed()
        except (ConnectionError, ssl.SSLError, OSError):
            pass

    async def relay(r, w):
        try:
            while True:
                data = await r.read(4096)
                if len(data) == 0:
                    break
                w.write(data)
                await w.drain()
        except Exception:
            pass
        finally:
            await close_writer(w)

    while True:
        local_w = None
        remote_w = None
        try:
            local_r, local_w = await asyncio.open_connection(local_host, local_port)
            remote_r, remote_w = await asyncio.open_connection(
                remote_host, remote_port, ssl=ssl_context,
            )
            await asyncio.wait(
                [
                    asyncio.gather(
                        asyncio.create_task(relay(local_r, remote_w)),
                        asyncio.create_task(relay(remote_r, local_w)),
                    ),
                    asyncio.create_task(close_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception:
            pass
        finally:
            if local_w is not None:
                await close_writer(local_w)
            if remote_w is not None:
                await close_writer(remote_w)
        if close_event.is_set():
            break

_viser_tunnel._simple_proxy = _patched_simple_proxy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dvsm.models.DVSM import DecoderViewSynthesisModel, NeuralRenderer
from dvsm.models.pose_preprocessor import PosePreprocessor
from viser_app.cam_convention import dvsm_cam_to_viser, viser_cam_to_dvsm
from viser_app.render import (
    WORLD_DVSM_TO_VISER as _WORLD_DVSM_TO_VISER,
    apply_pp_single as _apply_pp_single,
    build_renderer,
    load_model,
)
from viser_app.viewer_optim import install_optimizations


# -----------------------------------------------------------------------------
# Per-client render state + loop
# -----------------------------------------------------------------------------

class RenderState:
    """Thread-safe state shared between camera callback and render thread."""

    def __init__(self):
        self.dirty = threading.Event()
        self.dirty.set()
        self.snapshot = None  # (position, wxyz, fov_y, aspect)
        self.last_update_t = time.monotonic()
        self.lock = threading.Lock()

    def update_from_camera(self, cam: viser.CameraHandle):
        with self.lock:
            self.snapshot = (
                np.asarray(cam.position, dtype=np.float32).copy(),
                np.asarray(cam.wxyz, dtype=np.float32).copy(),
                float(cam.fov),
                float(cam.aspect),
            )
            self.last_update_t = time.monotonic()
            self.dirty.set()

    def consume(self):
        with self.lock:
            snap = self.snapshot
            since = time.monotonic() - self.last_update_t
            self.dirty.clear()
        return snap, since


def render_loop(
    client: viser.ClientHandle,
    renderer: NeuralRenderer,
    state: RenderState,
    max_res_slider,
    interactive_scale_slider,
    fps_readout,
    pp_scale: float,
    bench_table: dict,
    interactive_threshold_s: float = 0.15,
    min_dim: int = 64,
    async_push: bool = True,
):
    patch = int(renderer.model.patch_size)
    Rw_cuda = _WORLD_DVSM_TO_VISER.cuda()
    def _snap(x):
        return max(patch, (int(x) // patch) * patch)
    last_render_t = 0.0
    bench_widths = sorted(bench_table.keys())

    def _bench_predict_s(W: int) -> float:
        """Pure-GPU predicted render time at this W (from startup benchmark)."""
        if not bench_widths:
            return 0.0
        if W <= bench_widths[0]:
            return bench_table[bench_widths[0]]
        if W >= bench_widths[-1]:
            return bench_table[bench_widths[-1]]
        for i in range(len(bench_widths) - 1):
            a, b = bench_widths[i], bench_widths[i + 1]
            if a <= W <= b:
                r = (W - a) / (b - a)
                return bench_table[a] + r * (bench_table[b] - bench_table[a])
        return bench_table[bench_widths[-1]]

    # EMA over the real iteration time (GPU + JPEG encode + websocket push).
    # This is the true cadence at which complete frames reach the client and
    # is the right cap for rate-limiting during drag. We per-W bucket it so
    # changing the slider doesn't pollute the average.
    ema_iter_s_by_W: dict[int, float] = {}
    EMA_ALPHA = 0.3

    # Inertia detection: three.js OrbitControls keep producing camera updates
    # for ~300-500 ms after the user releases the mouse (damping). Server-side
    # we track recent (t, pos) and detect a decreasing-velocity tail; skip
    # rendering mid-decay frames so the user perceives "mouse stop ≈ image stop".
    import collections
    pos_history: collections.deque = collections.deque(maxlen=5)

    # Async-push: overlap viser's JPEG encode + websocket send with the next
    # render. Single-worker so frames stay ordered. Each iteration waits for
    # the previous push to drain before queuing a new one.
    push_executor = None
    push_future = None
    if async_push:
        from concurrent.futures import ThreadPoolExecutor
        push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viser-push")
    while True:
        # Short timeout so we can transition out of interactive mode
        # (full-quality follow-up render) shortly after the user stops moving.
        if not state.dirty.wait(timeout=0.05):
            continue
        snap, since_update = state.consume()
        if snap is None:
            continue
        position, wxyz, fov_y, aspect = snap

        # Dynamic downscale: optionally drop resolution while moving for smoother
        # drag. Slider == 1.0 disables the downscale entirely.
        i_scale = float(interactive_scale_slider.value)
        is_interactive = (since_update < interactive_threshold_s) and (i_scale < 1.0)
        scale = i_scale if is_interactive else 1.0
        # If this render is at interactive quality, keep dirty set so we
        # re-render at full quality after the user stops moving.
        if is_interactive:
            state.dirty.set()
        W = _snap(max(min_dim, int(round(max_res_slider.value * scale))))
        H = _snap(max(min_dim, int(round(W / aspect))))

        # Per-resolution rate limit during active drag. Use the EMA of real
        # iteration time (incl. JPEG + websocket overhead) when we have one,
        # otherwise fall back to the GPU-only benchmark prediction.
        now_t = time.monotonic()
        predicted = ema_iter_s_by_W.get(W, _bench_predict_s(W))
        is_during_drag = since_update < 0.3

        # Track recent position for inertia/deceleration detection.
        pos_history.append((now_t, np.asarray(position, dtype=np.float64)))
        is_decelerating = False
        if len(pos_history) >= 3 and is_during_drag:
            (t_a, p_a), (t_b, p_b), (t_c, p_c) = pos_history[-3], pos_history[-2], pos_history[-1]
            v_ab = float(np.linalg.norm(p_b - p_a)) / max(t_b - t_a, 1e-3)
            v_bc = float(np.linalg.norm(p_c - p_b)) / max(t_c - t_b, 1e-3)
            # Decelerating: velocity dropped >40% AND is now small. Both
            # thresholds scale with current scene scale (pp.s makes scene unit
            # size O(1) so absolute 0.005 unit/s is a reasonable "tiny").
            is_decelerating = (v_bc < v_ab * 0.6) and (v_bc < 0.005)

        # Skip if: we're rate-limited OR we're in inertia tail (and not too long
        # since last render — if it's been a while, render anyway so the final
        # settled frame always goes through).
        if is_during_drag and (now_t - last_render_t) < predicted * 0.9:
            state.dirty.set()
            time.sleep(0.005)
            continue
        if is_decelerating and (now_t - last_render_t) < 0.6:
            state.dirty.set()
            time.sleep(0.020)
            continue
        iter_t0 = time.monotonic()

        intr, c2w_r, c2w_t = viser_cam_to_dvsm(
            position, wxyz, fov_y, aspect, device="cuda", dtype=torch.float32
        )
        # viser_cam_to_dvsm returns poses in viser world frame; rotate the
        # world back into DVSM's normalized frame before rendering. Rw is
        # self-inverse so the same matrix works in both directions. Then apply
        # pp.s scaling (which we deliberately omitted from the viser frame so
        # the controls feel natural at the scene's original metric scale).
        c2w_r = torch.einsum('ij,bvjk->bvik', Rw_cuda, c2w_r)
        c2w_t = torch.einsum('ij,bvj->bvi', Rw_cuda, c2w_t) * pp_scale

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                img = renderer(intr, c2w_r, c2w_t, H, W)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"[render] error: {e}", flush=True)
            time.sleep(0.1)
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0
        last_render_t = time.monotonic()

        rgb = (
            img[0, 0].clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
            .permute(1, 2, 0).contiguous().cpu().numpy()
        )
        if push_executor is not None:
            # Drain the previous push (keeps frame order, also surfaces send
            # errors). On disconnect we exit cleanly.
            if push_future is not None:
                try:
                    push_future.result()
                except Exception as e:
                    print(f"[render] set_background_image failed ({e}); exiting loop", flush=True)
                    push_executor.shutdown(wait=False, cancel_futures=True)
                    return
            push_future = push_executor.submit(
                lambda r=rgb: client.scene.set_background_image(
                    r, format="jpeg", jpeg_quality=60),
            )
        else:
            try:
                client.scene.set_background_image(rgb, format="jpeg", jpeg_quality=60)
            except Exception as e:
                print(f"[render] set_background_image failed ({e}); exiting loop", flush=True)
                return

        # Update EMA with the complete iteration time (GPU + encode + push).
        iter_s = time.monotonic() - iter_t0
        prev_ema = ema_iter_s_by_W.get(W)
        ema_iter_s_by_W[W] = (
            iter_s if prev_ema is None
            else EMA_ALPHA * iter_s + (1 - EMA_ALPHA) * prev_ema
        )
        eff_fps = 1.0 / max(ema_iter_s_by_W[W], 1e-6)
        fps_readout.value = (
            f"{eff_fps:.1f} fps  (gpu {dt_ms:.0f}ms, iter {iter_s * 1000:.0f}ms, "
            f"{W}x{H}{', interactive' if is_interactive else ''})"
        )


# -----------------------------------------------------------------------------
# Viser server setup
# -----------------------------------------------------------------------------

def add_input_view_frustums(server: viser.ViserServer, scene_meta: dict,
                            pp: PosePreprocessor, scale: float = 0.08):
    """Add a camera frustum per input view, in the DVSM normalized world frame
    (poses passed through `pp` before conversion to viser)."""
    c2w_r = scene_meta["input_c2w_r"]
    c2w_t = scene_meta["input_c2w_t"]
    intr = scene_meta["input_intr"]
    V = c2w_r.shape[0]
    handles = []
    for v in range(V):
        r_n, t_n = _apply_pp_single(pp, c2w_r[v], c2w_t[v])
        pos, wxyz, fov_y, aspect = dvsm_cam_to_viser(intr[v], r_n, t_n)
        h = server.scene.add_camera_frustum(
            name=f"/input_views/{v:03d}",
            fov=float(fov_y),
            aspect=float(aspect),
            scale=scale,
            color=(0, 200, 200),
            wxyz=tuple(map(float, wxyz)),
            position=tuple(map(float, pos)),
        )
        handles.append(h)
    return handles


def serve(args):
    print(f"[viewer] loading model from {args.ckpt_root}", flush=True)
    model = load_model(args.ckpt_root, args.ckpt_name, [])
    print(f"[viewer] loading recon artifact {args.recon}", flush=True)
    artifact = torch.load(args.recon, weights_only=False, map_location="cpu")
    renderer = build_renderer(model, artifact)
    scene_meta = artifact["scene_meta"]
    print(f"[viewer] V={scene_meta['n_input_views']} input res={scene_meta['input_res']}", flush=True)

    # Align viser's world frame to DVSM's normalized frame (full SE(3) + scale
    # to unit sphere via the saved PosePreprocessor). All viser-side positions
    # are expressed in this normalized frame; the renderer's own preprocessor
    # is bypassed (set to None below) so the same normalization doesn't get
    # applied twice.
    pp_cpu = PosePreprocessor.__new__(PosePreprocessor)
    pp_cpu.r = artifact["pose_preproc"]["r"].cpu()
    pp_cpu.t = artifact["pose_preproc"]["t"].cpu()
    pp_cpu.s = artifact["pose_preproc"]["s"].cpu()
    renderer.pose_preprocessor = None
    pp_scale = float(pp_cpu.s.reshape(-1)[0])
    print(f"[viewer] viser frame = pp.r + pp.t + Rw (no pp.s); renderer-side pp.s = {pp_scale:.4f}", flush=True)

    # Viewer-only optimizations (FastPlucker cache + CUDA Graph + cuDNN warmup).
    # No-op for training: nothing in dvsm/ is modified; train.py never imports
    # viser_app/*. Optimization wraps run only in this serve() process.
    if not args.no_optim:
        renderer = install_optimizations(
            renderer,
            use_fast_plucker=not args.no_fast_plucker,
            use_cuda_graph=not args.no_cuda_graph,
            n_warmup=args.graph_warmup,
            max_graphs=args.max_graphs,
        )
        print(f"[viewer] optim installed: fast_plucker={not args.no_fast_plucker} "
              f"cuda_graph={not args.no_cuda_graph} warmup={args.graph_warmup} "
              f"max_graphs={args.max_graphs}", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    # World is rotated into viser's native Y-up convention (see _WORLD_DVSM_TO_VISER),
    # so scene up = +y (three.js default).
    server.scene.set_up_direction("+y")
    if args.share:
        try:
            share_url = server.request_share_url()
            print(f"[viewer] share URL: {share_url}", flush=True)
        except Exception as e:
            print(f"[viewer] share URL request failed ({e}); fall back to SSH forward", flush=True)
    frustum_handles = add_input_view_frustums(server, scene_meta, pp_cpu)

    # Pre-compute viser camera state for every input view (for jump-to-view +
    # warm-up). Poses are normalized via pp first so they live in viser's
    # (= DVSM normalized) world frame.
    V_inp = int(scene_meta["n_input_views"])
    input_views_viser = []
    for v in range(V_inp):
        r_n, t_n = _apply_pp_single(
            pp_cpu,
            scene_meta["input_c2w_r"][v],
            scene_meta["input_c2w_t"][v],
        )
        pos_v, wxyz_v, fov_v, _asp_v = dvsm_cam_to_viser(
            scene_meta["input_intr"][v], r_n, t_n,
        )
        input_views_viser.append((
            tuple(map(float, pos_v)),
            tuple(map(float, wxyz_v)),
            float(fov_v),
        ))
    init_pos_t, init_wxyz_t, init_fov_f = input_views_viser[0]
    init_pos = np.asarray(init_pos_t, dtype=np.float32)
    init_wxyz = np.asarray(init_wxyz_t, dtype=np.float32)
    init_fov = init_fov_f
    _aspect0 = float(scene_meta["input_res"][1]) / float(scene_meta["input_res"][0])

    # Warm up the renderer (first call may compile / autotune).
    print(f"[viewer] warming up renderer...", flush=True)
    patch = int(model.patch_size)
    Ww = max(patch, (256 // patch) * patch)
    Hw = max(patch, (int(round(Ww / float(_aspect0))) // patch) * patch)
    Rw_cuda_warmup = _WORLD_DVSM_TO_VISER.cuda()
    with torch.no_grad():
        intr_w, r_w, t_w = viser_cam_to_dvsm(
            init_pos, init_wxyz, init_fov, float(_aspect0),
            device="cuda", dtype=torch.float32,
        )
        r_w = torch.einsum('ij,bvjk->bvik', Rw_cuda_warmup, r_w)
        t_w = torch.einsum('ij,bvj->bvi', Rw_cuda_warmup, t_w) * pp_scale
        _ = renderer(intr_w, r_w, t_w, Hw, Ww)
        torch.cuda.synchronize()

    # Benchmark render time at common W values so the render loop can rate-
    # limit during drag (avoiding piled-up stale renders that look like
    # "screen keeps moving after the mouse stopped").
    print(f"[viewer] benchmarking render time...", flush=True)
    bench_widths_to_test = [256, 384, 512, 640, 768, 960]
    bench_table = {}
    with torch.no_grad():
        for Wb in bench_widths_to_test:
            Wt = max(patch, (Wb // patch) * patch)
            Ht = max(patch, (int(round(Wt / float(_aspect0))) // patch) * patch)
            ts = []
            for _ in range(3):
                torch.cuda.synchronize()
                tb = time.perf_counter()
                _ = renderer(intr_w, r_w, t_w, Ht, Wt)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - tb)
            bench_table[Wt] = min(ts)
            print(f"[bench]  W={Wt:4d}  H={Ht:4d}  ->  {min(ts) * 1000:5.0f} ms  "
                  f"(~{1.0 / min(ts):.1f} fps)", flush=True)
    print(f"[viewer] ready  ➜  open http://localhost:{args.port}", flush=True)

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        print(f"[viewer] client {client.client_id} connected", flush=True)
        client.camera.position = init_pos_t
        client.camera.wxyz = init_wxyz_t
        client.camera.fov = init_fov_f
        # IMPORTANT: setting position alone shifts look_at by the same delta
        # (viser doc). Pin look_at to viser-frame origin = pp.r/t centroid of
        # input cameras ≈ scene center, so orbit drags rotate around the scene
        # instead of some skewed-off point.
        try:
            client.camera.look_at = (0.0, 0.0, 0.0)
        except Exception:
            pass

        with client.gui.add_folder("Render"):
            max_res = client.gui.add_slider(
                "Max W", min=256, max=960, step=64, initial_value=min(args.max_res, 960)
            )
            interactive_scale_slider = client.gui.add_slider(
                "Interactive scale", min=0.25, max=1.0, step=0.25, initial_value=1.0,
            )
            fps_txt = client.gui.add_text("FPS", "—", disabled=True)
        with client.gui.add_folder("Camera"):
            view_slider = client.gui.add_slider(
                "Jump to input view", min=0, max=max(0, V_inp - 1), step=1, initial_value=0
            )
            reset_btn = client.gui.add_button("Reset View")
        with client.gui.add_folder("Scene"):
            show_frustums = client.gui.add_checkbox("Show input view frustums", initial_value=True)

        # Frustums default to visible (mirroring checkbox initial state).
        for h in frustum_handles:
            h.visible = True

        def _jump_to(v: int, recenter: bool = False):
            """Jump camera to input view ``v``.

            recenter=False (default): keep the input view's own orientation —
                the camera-to-look_at vector stays as captured.
            recenter=True (reset button only): after jumping, re-pin look_at
                to viser-frame origin so orbit target is the scene center.
            """
            v = max(0, min(V_inp - 1, int(v)))
            p, q, f = input_views_viser[v]
            client.camera.position = p
            client.camera.wxyz = q
            client.camera.fov = f
            if recenter:
                try:
                    client.camera.look_at = (0.0, 0.0, 0.0)
                except Exception:
                    pass

        @view_slider.on_update
        def _(_):
            _jump_to(view_slider.value, recenter=False)

        @reset_btn.on_click
        def _(_):
            view_slider.value = 0
            _jump_to(0, recenter=True)

        @show_frustums.on_update
        def _(_):
            for h in frustum_handles:
                h.visible = show_frustums.value

        state = RenderState()

        @client.camera.on_update
        def _(_cam):
            state.update_from_camera(client.camera)

        # Re-render immediately when render-control sliders change, even if the
        # camera hasn't moved (e.g. user only adjusted Max W or Interactive scale).
        @max_res.on_update
        def _(_):
            state.update_from_camera(client.camera)

        @interactive_scale_slider.on_update
        def _(_):
            state.update_from_camera(client.camera)

        thread = threading.Thread(
            target=render_loop,
            args=(client, renderer, state, max_res, interactive_scale_slider,
                  fps_txt, pp_scale, bench_table),
            kwargs={"async_push": not args.no_async_push},
            daemon=True,
        )
        thread.start()

    @server.on_client_disconnect
    def _on_disconnect(client: viser.ClientHandle) -> None:
        print(f"[viewer] client {client.client_id} disconnected", flush=True)

    while True:
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=Path, required=True,
                        help="Checkpoint directory: a Hugging Face release (config.json + "
                             "model.safetensors) or a train.py output (config.yaml + latest.pt).")
    parser.add_argument("--ckpt_name", default="latest.pt",
                        help="Weight file inside a train.py output directory "
                             "(a Hugging Face release always uses model.safetensors).")
    parser.add_argument("--recon", type=Path, required=True,
                        help="Path to .recon.pt produced by recon_scene.py")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max_res", type=int, default=768,
                        help="Initial Max W slider value; must align with step=64 used by the slider.")
    parser.add_argument("--share", action="store_true",
                        help="Request a viser studio share URL so the viewer is reachable without SSH forward.")
    parser.add_argument("--no_optim", action="store_true",
                        help="Disable all viewer-side speedups (FastPlucker + CUDAGraph + async push). Debug only.")
    parser.add_argument("--no_fast_plucker", action="store_true",
                        help="Disable FastPluckerEncoding (cache + matmul shortcut).")
    parser.add_argument("--no_cuda_graph", action="store_true",
                        help="Disable CUDA Graph capture/replay around the renderer.")
    parser.add_argument("--no_async_push", action="store_true",
                        help="Push frames synchronously instead of overlapping with the next render.")
    parser.add_argument("--graph_warmup", type=int, default=5,
                        help="cuDNN warmup iterations before each CUDA Graph capture.")
    parser.add_argument("--max_graphs", type=int, default=6,
                        help="Max number of cached (H, W, V) CUDA Graphs (LRU).")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
