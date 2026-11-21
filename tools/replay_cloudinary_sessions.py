"""
tools/replay_cloudinary_sessions.py
===================================
Download the latest real customer sessions from Cloudinary (damage_detection/)
and run ReferencePipeline inspection on them to evaluate real-world behavior.
"""

import os
import sys
import urllib.request
import cv2
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cloudinary
import cloudinary.api

from config import (
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    PROFILES_DIR,
    UPLOADS_DIR,
    VIEWS,
)
from core.pipeline import ReferencePipeline, resolve_profile_id


def load_ref_vectors(profiles_dir: str):
    ref_vecs = {}
    for f in os.listdir(profiles_dir):
        if f.endswith("_01_front_profile.pt"):
            pid = f.replace("_01_front_profile.pt", "")
            data = torch.load(os.path.join(profiles_dir, f), map_location="cpu", weights_only=False)
            v = data["dino_features"][0].mean(dim=0)
            v = v / (v.norm() + 1e-8)
            ref_vecs[pid] = v
    return ref_vecs


def identify_product(pipeline: ReferencePipeline, ref_vecs: dict, img_bgr: np.ndarray):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
    t_img = pipeline.transform(resized).unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        feat = pipeline.backbone.get_intermediate_layers(t_img, n=1)[0].squeeze(0)
        q = feat.mean(dim=0)
        q = q / (q.norm() + 1e-8)
    best_pid, best_sim = None, -1.0
    for pid, rvec in ref_vecs.items():
        sim = float((q @ rvec).item())
        if sim > best_sim:
            best_sim = sim
            best_pid = pid
    return best_pid, best_sim


def main():
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
    )
    target_sessions = ["a6221fba", "66379f44", "391b5fd1", "01d95c31", "7129b13e"]

    pipeline = ReferencePipeline(profiles_dir=PROFILES_DIR)
    ref_vecs = load_ref_vectors(PROFILES_DIR)

    replay_root = os.path.join(UPLOADS_DIR, "replay_sessions")
    os.makedirs(replay_root, exist_ok=True)

    for sid in target_sessions:
        res = cloudinary.api.resources(type="upload", prefix=f"damage_detection/{sid}/", max_results=20)
        resources = res.get("resources", [])
        if not resources:
            print(f"[SKIP] {sid}: no images found")
            continue

        s_dir = os.path.join(replay_root, sid)
        os.makedirs(s_dir, exist_ok=True)
        view_paths = {}
        for r in resources:
            v_name = r["public_id"].rsplit("/", 1)[-1]
            ext = r.get("format", "jpg")
            out_p = os.path.join(s_dir, f"{v_name}.{ext}")
            if not os.path.exists(out_p):
                with urllib.request.urlopen(r["secure_url"], timeout=30) as resp:
                    with open(out_p, "wb") as f:
                        f.write(resp.read())
            view_paths[v_name] = out_p

        probe_path = view_paths.get("front") or next(iter(view_paths.values()))
        probe_bgr = cv2.imread(probe_path)
        pid, sim = identify_product(pipeline, ref_vecs, probe_bgr)
        print(f"\n=== SESSION {sid} | Matched Product: {pid} (sim={sim*100:.1f}%) ===")

        for v in VIEWS:
            if v not in view_paths:
                continue
            prof_id = resolve_profile_id(pid, v, PROFILES_DIR)
            if not prof_id:
                print(f"  [{v}] missing profile for {pid}")
                continue
            report = pipeline.inspect_return(view_paths[v], prof_id, output_dir=s_dir)
            labels = [d["label"] for d in report.get("defects", [])]
            print(
                f"  [{v}] {report['severity']} | count={report['defect_count']} | "
                f"damage={report['damage_percentage']}% | labels={labels}"
            )


if __name__ == "__main__":
    main()
