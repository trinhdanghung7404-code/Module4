"""
utils/tunnel.py
===============
Robust public HTTPS tunnel manager (Cloudflare / ngrok / LAN IP)
for mobile camera capture (Safari iOS & Android Chrome require HTTPS).
Automatically enforces IPv4 and includes auto-restart on network drops.
"""

import os
import re
import time
import socket
import shutil
import threading
import subprocess
from typing import Tuple, Optional, Callable

_current_public_url: str = "http://localhost:8000"
_tunnel_proc: Optional[subprocess.Popen] = None
_on_url_change_cb: Optional[Callable[[str], None]] = None
_monitor_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def get_public_url() -> str:
    """Return the currently active public URL."""
    return _current_public_url


def set_url_change_callback(cb: Callable[[str], None]):
    """Register callback invoked whenever the tunnel URL changes or reconnects."""
    global _on_url_change_cb
    _on_url_change_cb = cb


def _set_active_url(url: str):
    global _current_public_url
    _current_public_url = url
    if _on_url_change_cb:
        try:
            _on_url_change_cb(url)
        except Exception:
            pass


def _launch_cloudflared_proc(port: int, bin_path: str) -> Tuple[Optional[str], Optional[subprocess.Popen]]:
    """Spawns cloudflared with strict IPv4 and HTTP/2 protocol."""
    cmd = [
        bin_path,
        "tunnel",
        "--edge-ip-version", "4",
        "--protocol", "http2",
        "--no-autoupdate",
        "--url", f"http://localhost:{port}"
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )
    except Exception as e:
        print(f"[TUNNEL] Failed to launch cloudflared: {e}")
        return None, None

    found_url = None
    start_t = time.time()

    # Read initial lines from stderr to capture the generated URL
    while time.time() - start_t < 30:
        if proc.poll() is not None:
            break
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.1)
            continue
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            found_url = m.group(0)
            break

    # Drain pipes in background thread to prevent buffer overflow deadlock
    def _drain():
        try:
            while proc.poll() is None and not _stop_event.is_set():
                line = proc.stderr.readline()
                if not line:
                    time.sleep(0.1)
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()
    return found_url, proc


def _watchdog_loop(port: int, bin_path: str):
    """Background watchdog: ensures the tunnel stays alive across ISP resets."""
    global _tunnel_proc
    while not _stop_event.is_set():
        time.sleep(5)
        if _tunnel_proc is not None and _tunnel_proc.poll() is not None:
            print("[TUNNEL] Cloudflared process exited unexpectedly. Re-establishing connection...")
            time.sleep(2)
            url, proc = _launch_cloudflared_proc(port, bin_path)
            if url and proc:
                _tunnel_proc = proc
                _set_active_url(url)
                print(f"[TUNNEL] Re-established Cloudflare Tunnel: {url}")


def start_tunnel(port: int, cloudflared_path: Optional[str] = None) -> Tuple[str, Optional[subprocess.Popen]]:
    """
    Starts public HTTPS tunnel with fallback cascade:
    1. Cloudflared (IPv4 HTTP/2, Free HTTPS, 100% iOS/Android compatible)
    2. pyngrok
    3. Local LAN IP
    """
    global _tunnel_proc, _monitor_thread
    public_url = f"http://localhost:{port}"

    # 1. Cloudflared
    is_valid_bin = cloudflared_path and (os.path.exists(cloudflared_path) or shutil.which(cloudflared_path))
    if is_valid_bin:
        print("[INFO] Initializing Cloudflare HTTPS Tunnel (IPv4 / HTTP/2)...")
        url, proc = _launch_cloudflared_proc(port, cloudflared_path)
        if url and proc:
            public_url = url
            _tunnel_proc = proc
            _set_active_url(url)
            
            # Start watchdog
            if not _monitor_thread or not _monitor_thread.is_alive():
                _monitor_thread = threading.Thread(
                    target=_watchdog_loop,
                    args=(port, cloudflared_path),
                    daemon=True
                )
                _monitor_thread.start()

    # 2. Fallback to pyngrok
    if not public_url.startswith("https://"):
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url.replace("http://", "https://")
            _set_active_url(public_url)
        except Exception:
            pass

    # 3. Fallback to LAN IP
    if not public_url.startswith("https://"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            public_url = f"http://{local_ip}:{port}"
            _set_active_url(public_url)
        except Exception:
            pass

    return public_url, _tunnel_proc


def stop_tunnel():
    """Stops tunnel processes and watchdog."""
    global _tunnel_proc
    _stop_event.set()
    if _tunnel_proc:
        try:
            _tunnel_proc.terminate()
        except Exception:
            pass
        _tunnel_proc = None
