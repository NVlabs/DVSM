"""TTB-DVSM end-to-end Gradio demo: upload images/video -> π³ pose
estimation -> TTB-DVSM reconstruction -> viser viewer embedded as iframe.

Layout:
    [ uploader ][ sliders ][ Reconstruct button ]
    [             <iframe to viser viewer>            ]

Run on a GPU node:
    python -m viser_app.gradio_demo
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Shared with recon_scene.py so both extract video frames identically.
from viser_app.scene_loaders import (  # noqa: E402
    IMAGE_EXTS, VIDEO_EXTS, extract_video_frames as _extract_video_frames,
)
DEFAULT_CKPT = PROJECT_ROOT / "outputs/DVSM-TTB-p8-DL3DV-960p"  # overridden by --ckpt_root
ARTIFACTS_DIR = PROJECT_ROOT / "outputs/gradio_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ----- single viser subprocess management ----------------------------------

_state = {
    "viser_proc": None,         # subprocess.Popen
    "viser_port": None,         # local TCP port that proc binds (proxied at /viser-proxy/)
    "viser_log_tail": [],       # last ~20 lines for diagnostics
}


def _kill_viser():
    p = _state.get("viser_proc")
    if p is not None and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    _state["viser_proc"] = None
    _state["viser_port"] = None


def _drain_into_log(stream):
    """Continuously read from a subprocess stdout in a daemon thread so its
    pipe doesn't fill up and stall the child."""
    for line in iter(stream.readline, ""):
        print(f"[viser] {line}", end="")
        tail = _state["viser_log_tail"]
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    stream.close()


def _pick_free_port() -> int:
    """Pick a free TCP port by binding to 0 and reading it back."""
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _launch_viser(recon_path: Path, timeout: float = 120.0) -> int:
    """Start viser viewer on a pre-picked local port; return that port.
    iframe is served through our own reverse-proxy at /viser-proxy/ — no
    viser.studio relay involved."""
    _kill_viser()
    _state["viser_log_tail"].clear()
    port = _pick_free_port()
    cmd = [
        "python", "-u", "-m", "viser_app.viewer",
        "--ckpt_root", str(DEFAULT_CKPT),
        "--recon", str(recon_path),
        "--port", str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        start_new_session=True,
    )
    _state["viser_proc"] = proc
    _state["viser_port"] = port

    # Wait for the viewer's "ready" line (printed after the warmup+bench loop).
    ready_pattern = re.compile(r"\[viewer\] ready")
    deadline = time.monotonic() + timeout
    consumed_lines: list[str] = []
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(
                    "viser exited early:\n" + "".join(consumed_lines[-30:])
                )
            time.sleep(0.05)
            continue
        consumed_lines.append(line)
        print(f"[viser] {line}", end="")
        if ready_pattern.search(line):
            threading.Thread(target=_drain_into_log, args=(proc.stdout,),
                             daemon=True).start()
            return port
    raise RuntimeError(
        f"viser didn't reach ready state within {timeout:.0f}s. Tail:\n"
        + "".join(consumed_lines[-30:])
    )


# ----- Gradio callback ------------------------------------------------------



def reconstruct_and_view(input_files, n_frames: int, target_w: int,
                         progress=gr.Progress(track_tqdm=False)):
    if not DEFAULT_CKPT.exists():
        return f"<p style='color:#c66'>Checkpoint not found: {DEFAULT_CKPT}</p>"
    if not input_files:
        return "<p style='color:#c66'>Upload a video or at least 3 images.</p>"

    work = Path(tempfile.mkdtemp(prefix="gradio_pi3_"))
    img_dir = work / "images"
    img_dir.mkdir()

    paths = [Path(f.name if hasattr(f, "name") else f) for f in input_files]
    videos = [p for p in paths if p.suffix.lower() in VIDEO_EXTS]
    images = [p for p in paths if p.suffix.lower() in IMAGE_EXTS]
    n_frames = int(n_frames)

    if videos:
        # Use the first video; sample n_frames evenly-spaced frames.
        progress(0.03, desc=f"Extracting {n_frames} frames from video...")
        try:
            n = _extract_video_frames(videos[0], img_dir, n_frames)
        except Exception as e:
            return f"<pre style='color:#c66;white-space:pre-wrap'>Video decode failed: {e}</pre>"
        if n < 3:
            return f"<p style='color:#c66'>Only {n} frames extracted; need at least 3.</p>"
        n_inputs = n
    elif len(images) >= 3:
        # Use uploaded photos; sub-sample evenly if there are more than n_frames.
        if len(images) > n_frames:
            import numpy as _np
            idxs = _np.linspace(0, len(images) - 1, n_frames).round().astype(int)
            images = [images[i] for i in idxs]
        progress(0.05, desc=f"Copying {len(images)} uploaded files...")
        for src in images:
            shutil.copy(src, img_dir / src.name)
        n_inputs = len(images)
    else:
        return "<p style='color:#c66'>Upload a video or at least 3 images.</p>"

    # 2. Snap target dims to the input's actual aspect, treating the slider as
    # the LONG axis. Both axes are then rounded to multiples of patch_size=8.
    # Landscape input → target_w = slider, target_h derived.
    # Portrait input  → target_h = slider, target_w derived.
    try:
        from PIL import Image as _PILImage
        first_img = next(iter(sorted(img_dir.iterdir())))
        with _PILImage.open(first_img) as im:
            iw, ih = im.size
        input_aspect = iw / ih  # >1 landscape, <1 portrait
    except Exception:
        input_aspect = 16 / 9  # fallback
    long_axis = (int(target_w) // 8) * 8
    if input_aspect >= 1.0:
        target_w = long_axis
        target_h = int(round(long_axis / input_aspect))
    else:
        target_h = long_axis
        target_w = int(round(long_axis * input_aspect))
    target_h = max(8, (target_h // 8) * 8)
    target_w = max(8, (target_w // 8) * 8)
    print(f"[gradio] input aspect={input_aspect:.3f}  "
          f"recon size {target_w}x{target_h}", flush=True)

    out_path = ARTIFACTS_DIR / f"{work.name}.recon.pt"

    # 3. Recon (π³ pose + TTB-DVSM) via subprocess
    progress(0.15, desc=f"Running π³ + TTB-DVSM ({n_inputs} frames)...")
    cmd = [
        "python", "-u", "-m", "viser_app.recon_scene",
        "--pose", "pi3",
        "--ckpt_root", str(DEFAULT_CKPT),
        "--input", str(img_dir),
        "--target_h", str(target_h),
        "--target_w", str(target_w),
        "--max_views", str(int(n_inputs)),
        "--out", str(out_path),
    ]
    print(f"[gradio] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return (
            f"<pre style='color:#c66;white-space:pre-wrap'>Reconstruction failed\n\n"
            f"stdout (last 2000B):\n{proc.stdout[-2000:]}\n\n"
            f"stderr (last 2000B):\n{proc.stderr[-2000:]}</pre>"
        )

    # 4. Launch viser
    progress(0.85, desc="Launching viser viewer...")
    try:
        port = _launch_viser(out_path)
    except RuntimeError as e:
        return f"<pre style='color:#c66;white-space:pre-wrap'>Viser launch failed:\n{e}</pre>"

    progress(0.98, desc="Embedding viewer...")
    # iframe points at our own reverse-proxy (?_t=port forces reload on new scene).
    return (
        f"<div style='width:100%;aspect-ratio:16/9;border:1px solid #444;"
        f"border-radius:8px;background:#111;overflow:hidden'>"
        f"<iframe src='/viser-proxy/?_t={port}' style='width:100%;height:100%;border:0' "
        f"allow='clipboard-write'></iframe></div>"
    )


# ----- Layout ---------------------------------------------------------------


NVIDIA_GREEN = "#76B900"
NVIDIA_GREEN_DARK = "#5C8F00"

# Tailwind-style palette anchored on NVIDIA green so every "primary" element
# (buttons, sliders, focus rings, gallery selection, tab indicators, ...) draws
# from this hue instead of whatever Spark theme uses by default.
_NVIDIA_PALETTE = gr.themes.Color(
    name="nvidia_green",
    c50="#F0F8E0",
    c100="#E1F0C2",
    c200="#C7E690",
    c300="#A8D958",
    c400="#8BC53F",
    c500=NVIDIA_GREEN,   # base
    c600=NVIDIA_GREEN_DARK,
    c700="#4A7300",
    c800="#395800",
    c900="#283D00",
    c950="#1A2900",
)
_theme = gr.themes.Default(primary_hue=_NVIDIA_PALETTE).set(
    button_primary_background_fill=NVIDIA_GREEN,
    button_primary_background_fill_hover=NVIDIA_GREEN_DARK,
    button_primary_text_color="white",
    button_primary_text_color_hover="white",
    slider_color=NVIDIA_GREEN,
    color_accent=NVIDIA_GREEN,
    color_accent_soft=NVIDIA_GREEN,
)
# Catch-all CSS: force every Spark-themed accent/highlight green to NVIDIA green.
_custom_css = """
button.primary, button[variant='primary'], .gr-button-primary {
    background: #76B900 !important;
    background-image: none !important;
    border-color: #76B900 !important;
    box-shadow: none !important;
    color: #fff !important;
}
button.primary:hover, button[variant='primary']:hover, .gr-button-primary:hover {
    background: #5C8F00 !important;
    background-image: none !important;
    border-color: #5C8F00 !important;
}
/* Sliders: thumb + filled track */
input[type='range']::-webkit-slider-thumb { background: #76B900 !important; }
input[type='range']::-moz-range-thumb     { background: #76B900 !important; }
.gr-slider .filled, .slider .filled       { background: #76B900 !important; }
/* Focus rings / outlines on form elements */
*:focus-visible, *:focus { outline-color: #76B900 !important; }
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: #76B900 !important;
    box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.4) !important;
}
/* Gallery selection / hover ring */
.gallery .selected, .gr-gallery .selected,
.gallery .thumbnail:hover, .gr-gallery .thumbnail:hover {
    border-color: #76B900 !important;
}
/* Progress / loader bars */
.progress, .gr-loader, .loader { background: #76B900 !important; }
/* Tab indicators */
.tab-nav .selected, .gr-tab-nav .selected { border-bottom-color: #76B900 !important; }
"""
# Shown in the viewer column until the first reconstruction replaces it.
_VIEWER_PLACEHOLDER = (
    "<div style='width:100%;aspect-ratio:16/9;display:flex;flex-direction:column;"
    "align-items:center;justify-content:center;gap:10px;"
    "border:1px solid var(--border-color-primary);border-radius:8px;"
    "color:var(--body-text-color)'>"
    "<svg width='26' height='26' viewBox='0 0 26 26' aria-hidden='true'>"
    "<rect x='1.5' y='1.5' width='23' height='23' fill='none' stroke='#76B900' stroke-width='3'/>"
    "<rect x='1.5' y='1.5' width='11.5' height='23' fill='#76B900'/></svg>"
    "<b>Viser Viewer</b></div>"
)

with gr.Blocks(title="TTB-DVSM Demo", theme=_theme, css=_custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.HTML(
                "<h1 style='margin-bottom:0.1em'>TTB-DVSM</h1>"
                "<h2 style='font-style:normal;font-weight:500;font-size:1.6em;"
                "margin-top:0;color:var(--body-text-color)'>"
                "Test-time MLP Baking for Efficient Rendering of "
                "Decoder-only View Synthesis Models"
                "</h2>"
                "<p style='color:var(--body-text-color);font-size:1.1em;margin-top:0.5em;line-height:1.6'>"
                "TTB-DVSM bakes the scene into fast weights in a single forward pass, "
                "and the result is browseable in the viser viewer."
                "</p>"
                "<div style='color:var(--body-text-color);font-size:1.1em;margin-top:0.8em;line-height:1.7'>"
                "<div style='color:#76B900;font-weight:700;font-size:1.15em;margin-bottom:0.3em'>"
                "Getting Started</div>"
                "<b>1.</b> Drop 16–128 photos <i>or</i> a short orbit video into the input box.<br>"
                "<b>2.</b> Pick how many frames to use and the reconstruction width.<br>"
                "<b>3.</b> Click <b>Reconstruct</b>."
                "<div style='color:#76B900;font-weight:700;font-size:1.15em;margin:0.7em 0 0.3em'>"
                "Viewer controls</div>"
                "&nbsp;• <b>Left-drag</b> orbit &nbsp;&nbsp;"
                "<b>Right-drag</b> pan &nbsp;&nbsp;"
                "<b>Scroll</b> zoom<br>"
                "&nbsp;• <b>W A S D</b> fly through the scene &nbsp;&nbsp;"
                "<b>Q / E</b> down / up &nbsp;&nbsp;"
                "<b>Shift</b> faster<br>"
                "&nbsp;• Right-side panel: <b>Max W</b>, "
                "<b>Jump to input view</b>, <b>Reset View</b>."
                "</div>"
            )
            inputs = gr.File(
                file_count="multiple",
                file_types=["image", "video", ".mp4", ".mov", ".mkv",
                            ".webm", ".avi", ".m4v"],
                show_label=False,
                # Caps the uploaded-file list (it scrolls) so the sliders and
                # the button stay in view after dropping many photos.
                height=240,
            )
            n_frames = gr.Slider(8, 128, 32, step=4, label="Frames to sample")
            tw = gr.Slider(384, 960, 768, step=64,
                           label="Reconstruction width (long axis)")
            btn = gr.Button("Reconstruct", variant="primary")

        with gr.Column(scale=3):
            viewer_html = gr.HTML(_VIEWER_PLACEHOLDER, label="Viewer")

    btn.click(reconstruct_and_view,
              [inputs, n_frames, tw],
              viewer_html)


def _build_proxy_handlers():
    """Build reverse-proxy handlers (HTTP + WebSocket) that forward
    /viser-proxy/* to the local viser sub-process. iframe lives inside the
    gradio origin → no viser.studio relay required."""
    import asyncio
    import httpx
    import websockets as ws_pkg
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.websockets import WebSocket

    # 60s timeout: the initial HTML response from viser is ~2.8 MB, fine.
    http_client = httpx.AsyncClient(timeout=60.0)

    async def proxy_http(request: Request) -> Response:
        """HTTP reverse proxy: /viser-proxy[/<path>] -> http://localhost:<port>/<path>"""
        path = request.path_params.get("path", "")
        port = _state.get("viser_port")
        if port is None or _state.get("viser_proc") is None or \
                _state["viser_proc"].poll() is not None:
            return Response(b"No active viser session.", status_code=503)
        url = f"http://localhost:{port}/{path}"
        if request.url.query:
            url += "?" + request.url.query
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}
        body = await request.body()
        try:
            r = await http_client.request(
                request.method, url, headers=headers, content=body,
                follow_redirects=False,
            )
        except Exception as e:
            return Response(f"viser unreachable: {e}".encode(), status_code=502)
        resp_headers = {k: v for k, v in r.headers.items()
                        if k.lower() not in ("content-encoding", "transfer-encoding",
                                              "content-length", "connection")}
        return Response(content=r.content, status_code=r.status_code,
                        headers=resp_headers)

    async def proxy_ws(websocket: WebSocket) -> None:
        """WebSocket reverse proxy with `viser-v{N}` subprotocol passthrough."""
        path = websocket.path_params.get("path", "")
        port = _state.get("viser_port")
        offered = list(websocket.scope.get("subprotocols") or [])
        print(f"[proxy] ws upgrade path='{path}' port={port} "
              f"subprotocols={offered}", flush=True)
        if port is None:
            print("[proxy] no viser_port -> close", flush=True)
            await websocket.close(code=1011)
            return
        try:
            upstream = await ws_pkg.connect(
                f"ws://localhost:{port}/{path}",
                subprotocols=offered or None,
                max_size=50 * 1024 * 1024,
                compression=None,
            )
        except Exception as e:
            print(f"[proxy] upstream ws connect failed: {e!r}", flush=True)
            await websocket.close(code=1011)
            return
        print(f"[proxy] upstream connected, chose subprotocol={upstream.subprotocol}", flush=True)
        await websocket.accept(subprotocol=upstream.subprotocol)

        async def c2u():
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    b = msg.get("bytes")
                    t = msg.get("text")
                    if b is not None:
                        await upstream.send(b)
                    elif t is not None:
                        await upstream.send(t)
            except Exception:
                pass
            finally:
                try:
                    await upstream.close()
                except Exception:
                    pass

        async def u2c():
            try:
                async for msg in upstream:
                    if isinstance(msg, (bytes, bytearray)):
                        await websocket.send_bytes(bytes(msg))
                    else:
                        await websocket.send_text(msg)
            except Exception:
                pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

        await asyncio.gather(c2u(), u2c())

    # Use Starlette-native route registration to skip FastAPI's Pydantic-based
    # parameter validation (which incorrectly treats `request` / `req` as a
    # query field on bare-Request endpoints, giving 422).
    return proxy_http, proxy_ws


def main():
    """Launch gradio (which starts uvicorn) and patch in viser proxy routes."""
    import argparse
    import time

    global DEFAULT_CKPT

    parser = argparse.ArgumentParser(description="TTB-DVSM Gradio demo")
    parser.add_argument(
        "--ckpt_root",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Checkpoint directory (default: {DEFAULT_CKPT})",
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live share URL")
    args = parser.parse_args()

    DEFAULT_CKPT = args.ckpt_root.resolve()
    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {DEFAULT_CKPT}")
    print(f"[main] using checkpoint: {DEFAULT_CKPT}", flush=True)

    proxy_http, proxy_ws = _build_proxy_handlers()
    demo.queue(max_size=4)
    # Gradio 6.0: theme/css moved from Blocks() to launch(). Pass them here so
    # they actually take effect (otherwise gradio falls back to default orange).
    demo.launch(
        server_name="0.0.0.0", server_port=args.port,
        share=args.share, prevent_thread_lock=True,
        theme=_theme,
    )
    # Append our reverse-proxy routes onto gradio's running FastAPI app.
    # Starlette matches routes from .routes in order; appending puts them at
    # the end, but gradio's catch-all is a separate Mount so our specific
    # /viser-proxy/{path} routes still win.
    app = demo.app  # the FastAPI app gradio created
    http_methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
    app.add_route("/viser-proxy", proxy_http, methods=http_methods)
    app.add_route("/viser-proxy/{path:path}", proxy_http, methods=http_methods)
    app.router.add_websocket_route("/viser-proxy", proxy_ws)
    app.router.add_websocket_route("/viser-proxy/{path:path}", proxy_ws)
    print("[main] viser-proxy routes attached", flush=True)
    # Block forever.
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
