"""
utils/tunnel.py
===============
Utility to establish public HTTPS tunnel (Cloudflare / ngrok / LAN IP)
for mobile camera capture (Safari iOS requires HTTPS).
"""

import os
import re
import time
import socket
import shutil
import threading
import subprocess
from typing import Tuple, Optional


def _drain_pipe(pipe):
    """Đọc và xả buffer pipe liên tục để chống tràn bộ đệm OS làm treo tunnel."""
    try:
        for _ in iter(pipe.readline, ''):
            pass
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def start_tunnel(port: int, cloudflared_path: Optional[str] = None) -> Tuple[str, Optional[subprocess.Popen]]:
    """
    Khởi động HTTPS tunnel cho camera di động:
    1. Thử Cloudflared (miễn phí, có SSL chính thức, Safari iOS hỗ trợ 100%)
    2. Fallback sang pyngrok
    3. Fallback sang LAN IP
    """
    public_url = f"http://localhost:{port}"
    proc = None

    # 1. Thử Cloudflared
    is_valid_bin = cloudflared_path and (os.path.exists(cloudflared_path) or shutil.which(cloudflared_path))
    if is_valid_bin:
        try:
            print("[INFO] Đang khởi tạo Cloudflare HTTPS Tunnel cho camera di động...")
            cmd = [cloudflared_path, "tunnel", "--url", f"http://localhost:{port}"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            
            start_t = time.time()
            while time.time() - start_t < 15:
                line = proc.stderr.readline()
                if line:
                    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                    if m:
                        public_url = m.group(0)
                        break

            # Chạy thread đọc xả pipe để chống deadlock khi lưu lượng truy cập cao
            if proc.stderr:
                threading.Thread(target=_drain_pipe, args=(proc.stderr,), daemon=True).start()
            if proc.stdout:
                threading.Thread(target=_drain_pipe, args=(proc.stdout,), daemon=True).start()
        except Exception as e:
            print(f"[WARN] Không khởi tạo được Cloudflare tunnel: {e}")
        except Exception as e:
            print(f"[WARN] Không khởi tạo được Cloudflare tunnel: {e}")

    # 2. Fallback sang ngrok
    if not public_url.startswith("https://"):
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url.replace("http://", "https://")
        except Exception:
            pass

    # 3. Fallback sang IP LAN nội bộ
    if not public_url.startswith("https://"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            public_url = f"http://{local_ip}:{port}"
        except Exception:
            pass

    return public_url, proc
