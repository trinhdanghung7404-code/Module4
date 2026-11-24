"""
app.py
======
FastAPI Application for Ceramic Defect Inspection & Return Verification.
Entrypoint for Module4_AIReturn.

Run:
    python app.py
or:
    run_server.bat
"""

import os
import sys
import uuid

if "TORCH_HOME" not in os.environ and os.path.exists(r"E:\Ceramic_Defect_Benchmark\cache\torch"):
    os.environ["TORCH_HOME"] = r"E:\Ceramic_Defect_Benchmark\cache\torch"
import time
import json
import base64
import asyncio
from contextlib import asynccontextmanager
import cv2
import numpy as np
from typing import Optional, Union, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    SERVER_HOST, SERVER_PORT,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET,
    DATA_DIR, REFERENCE_IMAGES_DIR, PROFILES_DIR, GHOST_IMAGES_DIR,
    TEMPLATES_DIR, UPLOADS_DIR, CLOUDFLARED_BIN,
    SESSION_EXPIRE_MINUTES, VIEWS, VIEW_LABELS, AUTO_CAPTURE, LEGACY_VIEW_MAP,
)
from core.pipeline import ReferencePipeline, resolve_profile_id
from core.preprocessor import process_and_detect
from core.matcher import (
    process_single_ghost_image,
    compute_similarity_from_bgr,
    validate_upload_match,
)
from core.defect_simulator import simulate_defect_on_session, reset_simulation_on_session
from utils.qr_code import generate_qr_base64, generate_qr_bytes
from utils.tunnel import start_tunnel, get_public_url, set_url_change_callback, stop_tunnel

# ── App Initialization ────────────────────────────────────────────────────────
# ── Automated Inspection Job Queue ───────────────────────────────────────────
inspection_queue: asyncio.Queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(inspection_worker())
    print("[INIT] Automated inspection queue worker started successfully.")
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    print("[SHUTDOWN] Inspection queue worker stopped.")


app = FastAPI(
    title="Ceramic Defect Inspection Server - Module4_AIReturn",
    lifespan=lifespan
)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def resolve_ghost_path(product_id: str, view: str) -> Optional[str]:
    """Tìm đường dẫn ảnh bóng mờ trong suốt (chuẩn hóa front/back/left/right.png)."""
    p_dir = os.path.join(GHOST_IMAGES_DIR, str(product_id))
    if not os.path.isdir(p_dir):
        if os.path.isdir(os.path.join(GHOST_IMAGES_DIR, f"B{product_id}")):
            p_dir = os.path.join(GHOST_IMAGES_DIR, f"B{product_id}")
        else:
            return None

    v = view.lower()
    std_file = os.path.join(p_dir, f"{v}.png")
    if os.path.isfile(std_file):
        return std_file
    return None

# Serve ghost overlay images with smart fallback resolution
@app.get("/ghost/{product_id}/{filename:path}")
async def serve_ghost_image(product_id: str, filename: str):
    """
    Phục vụ ảnh bóng mờ thông minh:
    - Tự động chuẩn hóa product_id: '1' -> 'B1', '22' -> 'B22'.
    - Tự động chuẩn hóa góc chụp: 'front' -> ['front.png', 'front_nb.png', 'truoc_nb.png', ...].
    - Đảm bảo camera di động luôn tải được ảnh 200 OK bất kể tên file trên đĩa cứng là gì.
    """
    cand_pids = [product_id]
    if product_id.isdigit():
        cand_pids.append(f"B{product_id}")
    elif product_id.upper().startswith("B"):
        cand_pids.append(product_id.upper())
    else:
        cand_pids.append(f"B{product_id}")

    target_pdir = None
    for pid in cand_pids:
        pdir = os.path.join(GHOST_IMAGES_DIR, pid)
        if os.path.isdir(pdir):
            target_pdir = pdir
            break

    if not target_pdir:
        for fallback in ["B22", "B1"]:
            pdir = os.path.join(GHOST_IMAGES_DIR, fallback)
            if os.path.isdir(pdir):
                target_pdir = pdir
                break

    if not target_pdir or not os.path.isdir(target_pdir):
        return JSONResponse({"error": f"Ghost directory not found for {product_id}"}, status_code=404)

    clean_name = os.path.basename(filename).lower()
    base_stem = clean_name.replace(".png", "").replace(".jpg", "").replace("_nb", "")

    view_aliases = {
        "front": ["front.png", "front_nb.png", "truoc_nb.png", "truoc.png", "truoc.jpg"],
        "back":  ["back.png", "back_nb.png", "sau_nb.png", "sau.png", "sau.jpg"],
        "left":  ["left.png", "left_nb.png", "trai_nb.png", "trai.png", "trai.jpg"],
        "right": ["right.png", "right_nb.png", "phai_nb.png", "phai.png", "phai.jpg"],
        "truoc": ["truoc_nb.png", "front.png", "front_nb.png", "truoc.png", "truoc.jpg"],
        "sau":   ["sau_nb.png", "back.png", "back_nb.png", "sau.png", "sau.jpg"],
        "trai":  ["trai_nb.png", "left.png", "left_nb.png", "trai.png", "trai.jpg"],
        "phai":  ["phai_nb.png", "right.png", "right_nb.png", "phai.png", "phai.jpg"],
    }

    cand_filenames = [clean_name]
    if base_stem in view_aliases:
        cand_filenames.extend(view_aliases[base_stem])
    else:
        for v, aliases in view_aliases.items():
            if v in base_stem:
                cand_filenames.extend(aliases)
                break

    for fn in cand_filenames:
        fp = os.path.join(target_pdir, fn)
        if os.path.isfile(fp):
            return FileResponse(fp, media_type="image/png")

    try:
        files_in_dir = os.listdir(target_pdir)
        for fn in files_in_dir:
            fn_lower = fn.lower()
            for v in [base_stem] + view_aliases.get(base_stem, []):
                key = v.replace(".png", "").replace(".jpg", "").replace("_nb", "")
                if key and key in fn_lower and (fn_lower.endswith(".png") or fn_lower.endswith(".jpg")):
                    return FileResponse(os.path.join(target_pdir, fn), media_type="image/png")
    except Exception:
        pass

    return JSONResponse({"error": f"Ghost file not found for {product_id}/{filename}"}, status_code=404)


# Serve ghost overlay images
if os.path.isdir(GHOST_IMAGES_DIR):
    app.mount("/ghost", StaticFiles(directory=GHOST_IMAGES_DIR), name="ghost")

# ── Cloudinary (Optional Cloud Storage) ───────────────────────────────────────
_cloudinary_ready = False
if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    try:
        import cloudinary
        import cloudinary.uploader
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET,
        )
        _cloudinary_ready = True
        print("[OK] Cloudinary kết nối thành công.")
    except ImportError:
        print("[WARN] Thư viện cloudinary chưa cài đặt. Ảnh lưu trữ local tại uploads/")
else:
    print("[INFO] Cloudinary chưa cấu hình. Ảnh lưu trữ local tại uploads/")

# ── Reference AI Pipeline (DINOv2 SOTA) ────────────────────────────────────────
_reference_pipeline: Optional[ReferencePipeline] = None


def get_reference_pipeline() -> ReferencePipeline:
    """Tải một lần hoặc tái sử dụng instance ReferencePipeline."""
    global _reference_pipeline
    if _reference_pipeline is None:
        _reference_pipeline = ReferencePipeline(profiles_dir=PROFILES_DIR)
    return _reference_pipeline


# ── In-Memory Session Store ───────────────────────────────────────────────────
sessions: Dict[str, Dict[str, Any]] = {}


def _clean_expired_sessions():
    """Xóa các phiên làm việc đã quá hạn."""
    now = time.time()
    expired = [
        sid for sid, s in sessions.items()
        if now - s.get("created_at", now) > SESSION_EXPIRE_MINUTES * 60
    ]
    for sid in expired:
        del sessions[sid]


# ── Core Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def desktop_page(request: Request):
    """Giao diện máy tính: Giám sát quy trình chụp & hiển thị QR."""
    return templates.TemplateResponse(request, "desktop.html")


@app.get("/api/products")
async def list_products():
    """Liệt kê danh sách các sản phẩm có ảnh bóng mờ trong kho dữ liệu."""
    prods = []
    seen_ids = set()

    if os.path.isdir(GHOST_IMAGES_DIR):
        for name in os.listdir(GHOST_IMAGES_DIR):
            p_dir = os.path.join(GHOST_IMAGES_DIR, name)
            if os.path.isdir(p_dir):
                files = os.listdir(p_dir)
                valid_views = []
                for v in VIEWS:
                    vn = {"front": "truoc", "back": "sau", "left": "trai", "right": "phai"}.get(v, v)
                    if any(f in files for f in [f"{v}.png", f"{v}_nb.png", f"{vn}_nb.png", f"{vn}.png", f"{vn}.jpg"]):
                        valid_views.append(v)
                if valid_views:
                    seen_ids.add(name)
                    prods.append({
                        "id": name,
                        "views": valid_views,
                        "total_views": len(valid_views),
                    })

    if os.path.isdir(PROFILES_DIR):
        for f in os.listdir(PROFILES_DIR):
            if f.endswith("_profile.pt"):
                parts = f.split("_")
                pid = parts[0]
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    prods.append({
                        "id": pid,
                        "views": VIEWS,
                        "total_views": len(VIEWS),
                    })

    def natural_key(p):
        pid = str(p["id"])
        num_part = "".join(filter(str.isdigit, pid))
        return (0 if pid.startswith("B") else 1, int(num_part) if num_part else 9999, pid)

    prods.sort(key=natural_key)
    return {"products": prods}


@app.post("/api/product/upload")
async def upload_product_images(
    product_id: str = Form(...),
    truoc: UploadFile = File(None),
    sau: UploadFile = File(None),
    trai: UploadFile = File(None),
    phai: UploadFile = File(None),
):
    """
    Tải lên ảnh gốc sản phẩm -> tự động tạo bóng mờ và lưu vào kho mẫu.
    """
    product_id = product_id.strip()
    if not product_id:
        return JSONResponse({"error": "Vui lòng nhập Product ID"}, status_code=400)

    target_dir = os.path.join(GHOST_IMAGES_DIR, product_id)
    os.makedirs(target_dir, exist_ok=True)

    uploaded_files = {
        "truoc": truoc,
        "sau": sau,
        "trai": trai,
        "phai": phai,
    }

    results = {}
    processed_count = 0

    for view, file_obj in uploaded_files.items():
        if not file_obj or not file_obj.filename:
            continue

        file_bytes = await file_obj.read()
        if len(file_bytes) == 0:
            continue

        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            results[view] = {"status": "error", "message": "Không thể đọc định dạng ảnh"}
            continue

        # 1. Lưu ảnh gốc
        orig_filename = f"{view}.jpg"
        orig_path = os.path.join(target_dir, orig_filename)
        cv2.imwrite(orig_path, img)

        # 2. Tạo ghost overlay
        try:
            ghost_img = process_single_ghost_image(img)
            ghost_filename = f"{view}_nb.png"
            ghost_path = os.path.join(target_dir, ghost_filename)
            cv2.imwrite(ghost_path, ghost_img)
        except Exception as e:
            print(f"[ERROR] Xử lý ghost cho {view} thất bại: {e}")
            results[view] = {"status": "error", "message": f"Lỗi ghost: {e}"}
            continue

        # 3. Đồng bộ Cloudinary nếu có cấu hình
        c_orig_url = None
        c_ghost_url = None
        if _cloudinary_ready:
            try:
                import cloudinary.uploader
                r_orig = cloudinary.uploader.upload(
                    file_bytes,
                    folder=f"damage_detection/products/{product_id}",
                    public_id=f"{view}_orig",
                    overwrite=True,
                )
                c_orig_url = r_orig.get("secure_url")

                with open(ghost_path, "rb") as gf:
                    r_ghost = cloudinary.uploader.upload(
                        gf.read(),
                        folder=f"damage_detection/products/{product_id}",
                        public_id=f"{view}_nb",
                        overwrite=True,
                    )
                c_ghost_url = r_ghost.get("secure_url")
            except Exception as ce:
                print(f"[WARN] Upload Cloudinary thất bại cho {view}: {ce}")

        processed_count += 1
        results[view] = {
            "status": "ok",
            "ghost_local": f"/ghost/{product_id}/{ghost_filename}",
            "ghost_cloud": c_ghost_url,
            "orig_local": f"/ghost/{product_id}/{orig_filename}",
            "orig_cloud": c_orig_url,
        }

    return {
        "status": "ok" if processed_count > 0 else "empty",
        "product_id": product_id,
        "processed_count": processed_count,
        "results": results,
    }



@app.get("/api/tunnel")
async def get_tunnel_status():
    """Return active public URL for dynamic QR generation."""
    url = get_public_url() or getattr(app.state, "public_url", f"http://localhost:{SERVER_PORT}")
    return {"public_url": url, "is_https": url.startswith("https://")}

@app.post("/api/session")
async def create_session(product_id: str = Form(...)):
    """Tạo phiên kiểm định mới và tạo mã QR để quét trên di động."""
    _clean_expired_sessions()

    product_id = product_id.strip()
    if product_id.isdigit():
        product_id = f"B{product_id}"
    elif not product_id.upper().startswith("B") and os.path.isdir(os.path.join(GHOST_IMAGES_DIR, f"B{product_id}")):
        product_id = f"B{product_id}"

    session_id = uuid.uuid4().hex[:8]
    sessions[session_id] = {
        "product_id": product_id,
        "created_at": time.time(),
        "views": {v: {"status": "pending", "url": None} for v in VIEWS},
        "status": "waiting",
    }

    base_url = get_public_url() or getattr(app.state, "public_url", f"http://localhost:{SERVER_PORT}")
    capture_url = f"{base_url}/capture/{session_id}"
    qr_data = generate_qr_base64(capture_url)

    return {
        "session_id": session_id,
        "capture_url": capture_url,
        "qr_base64": qr_data,
    }


@app.get("/capture/{session_id}", response_class=HTMLResponse)
async def capture_page(
    request: Request,
    session_id: str,
    productCode: Optional[str] = Query(None),
    returnId: Optional[Union[str, int]] = Query(None)
):
    """Trang chụp ảnh di động (auto-capture)."""
    session = sessions.get(session_id)

    # Tự khởi tạo session nếu mở trực tiếp từ liên kết có params
    if not session:
        p_code = productCode or "B1"
        sessions[session_id] = {
            "product_id": p_code,
            "return_id": returnId,
            "created_at": time.time(),
            "views": {v: {"status": "pending", "url": None} for v in VIEWS},
            "status": "waiting",
        }
        session = sessions[session_id]
    else:
        if returnId is not None:
            session["return_id"] = returnId
        if productCode:
            session["product_id"] = productCode

    product_id = session.get("product_id", "B1")
    ghost_available = os.path.isdir(os.path.join(GHOST_IMAGES_DIR, str(product_id)))
    if not ghost_available and (bool(CLOUDINARY_CLOUD_NAME) or os.path.isdir(os.path.join(GHOST_IMAGES_DIR, "B1"))):
        ghost_available = True

    return templates.TemplateResponse(request, "capture.html", {
        "session_id": session_id,
        "product_id": product_id,
        "return_id": session.get("return_id"),
        "views": VIEWS,
        "view_labels": VIEW_LABELS,
        "config": AUTO_CAPTURE,
        "ghost_available": ghost_available,
        "cloudinary_cloud_name": CLOUDINARY_CLOUD_NAME,
    })


@app.post("/api/upload/{session_id}")
async def upload_image(
    session_id: str,
    view: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Tiếp nhận ảnh chụp từ điện thoại:
    - Kiểm tra độ sắc nét quang học và chống chụp vật thể lạ/bàn phím.
    - Lưu trữ cục bộ tại uploads/{session_id}/ và đồng bộ lên Cloudinary (nếu bật).
    """
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session không tồn tại hoặc đã hết hạn"}, status_code=404)

    view = LEGACY_VIEW_MAP.get(view.lower(), view.lower())
    if view not in VIEWS:
        return JSONResponse({"error": f"Invalid view: {view}. Allowed: {VIEWS}"}, status_code=400)

    image_bytes = await image.read()

    # Cửa soát chất lượng tải lên
    product_id = session.get("product_id", "B1")
    ghost_path = resolve_ghost_path(product_id, view) or os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}.png")
    ref_path = os.path.join(REFERENCE_IMAGES_DIR, str(product_id), f"{view}.png")
    val_res = validate_upload_match(ghost_path, image_bytes, ref_path=ref_path)
    if not val_res.get("valid", True):
        msg = val_res.get("message", "Ảnh không khớp sản phẩm mẫu")
        print(f"[REJECTED] {session_id} view={view}: {msg}")
        return JSONResponse({
            "status": "error",
            "message": msg
        }, status_code=400)

    print(f"[ACCEPTED] {session_id} view={view}: ảnh hợp lệ, lưu vào hệ thống")

    # Lưu file gốc
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    local_path = os.path.join(session_dir, f"{view}.jpg")
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    local_url = f"/uploads/{session_id}/{view}.jpg"

    # Invalidate stale preprocessed artifacts for this view
    for stale_file in [
        f"customer_{view}_processed.png",
        f"customer_{view}_specular_mask.png",
        f"customer_{view}_saturated_mask.png",
        f"customer_{view}_ignore_mask.png",
        f"customer_{view}_overlay_contour.png",
        f"reference_{view}_processed.png",
    ]:
        sp = os.path.join(session_dir, stale_file)
        if os.path.exists(sp):
            try:
                os.remove(sp)
            except Exception:
                pass

    url = local_url
    if _cloudinary_ready:
        try:
            import cloudinary.uploader
            result = cloudinary.uploader.upload(
                image_bytes,
                folder=f"damage_detection/{session_id}",
                public_id=view,
                overwrite=True,
                resource_type="image",
            )
            url = result["secure_url"]
        except Exception as e:
            print(f"[WARN] Cloudinary error: {e}, fallback local")

    session["views"][view] = {"status": "uploaded", "url": url}

    all_done = all(v["status"] == "uploaded" for v in session["views"].values())
    if all_done:
        session["status"] = "complete"
        curr_pipe = session.get("pipeline_status")
        if curr_pipe not in ("queued", "preprocessing", "inspecting", "ready"):
            session["pipeline_status"] = "queued"
            session["pipeline_step"] = "Đã xếp vào hàng đợi phân tích tự động..."
            _save_session_disk(session_id, session)
            await inspection_queue.put(session_id)
            print(f"[QUEUE] Tự động xếp hàng kiểm định session {session_id}. Queue size: {inspection_queue.qsize()}")

    uploaded_count = sum(1 for v in session["views"].values() if v["status"] == "uploaded")

    return {
        "status": "ok",
        "view": view,
        "url": url,
        "uploaded": uploaded_count,
        "total": len(VIEWS),
        "all_done": all_done,
    }


def _save_session_disk(session_id: str, session: Dict[str, Any]):
    """Ghi dữ liệu phiên làm việc xuống đĩa để bảo toàn trạng thái khi khởi động lại server."""
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    sess_json_path = os.path.join(session_dir, "session.json")
    try:
        with open(sess_json_path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Không thể ghi session.json cho {session_id}: {e}")

def _trigger_shop_webhook(session_id: str, return_id: str, results: Dict[str, Any], session: Dict[str, Any]):
    """Gửi Webhook callback sang ceramic_shop_server (cổng 5000) để cập nhật PostgreSQL."""
    try:
        import urllib.request
        import json

        is_damaged = any(v.get("is_damaged", False) for v in results.values() if isinstance(v, dict))
        max_pct = max([v.get("damage_percentage", 0.0) for v in results.values() if isinstance(v, dict)] or [0.0])
        score = round(min(max(max_pct / 100.0, 0.0), 1.0), 4)

        categories = set()
        all_defects = []
        for view_name, r in results.items():
            if isinstance(r, dict):
                for d in r.get("defects", []):
                    cat = d.get("category", "CRACK").upper()
                    categories.add(cat)
                    d_copy = dict(d)
                    d_copy["view"] = view_name
                    all_defects.append(d_copy)

        defect_type = ", ".join(sorted(categories)) if categories else ("DEFECTIVE" if is_damaged else "INTACT")

        views_info = session.get("views", {})
        captured_imgs = {
            "truoc": views_info.get("front", {}).get("url") or views_info.get("truoc", {}).get("url") or f"/uploads/{session_id}/front.jpg",
            "sau": views_info.get("back", {}).get("url") or views_info.get("sau", {}).get("url") or f"/uploads/{session_id}/back.jpg",
            "trai": views_info.get("left", {}).get("url") or views_info.get("trai", {}).get("url") or f"/uploads/{session_id}/left.jpg",
            "phai": views_info.get("right", {}).get("url") or views_info.get("phai", {}).get("url") or f"/uploads/{session_id}/right.jpg",
        }

        result_imgs = {
            k: v.get("result_image_url") for k, v in results.items() if isinstance(v, dict)
        }

        payload = {
            "captureSessionId": session_id,
            "capturedImages": captured_imgs,
            "aiDetectedDamage": is_damaged,
            "aiDamageScore": score,
            "aiDefectType": defect_type,
            "customerAgreedWithAi": True,
            "defectDetails": all_defects,
            "resultImages": result_imgs,
        }

        shop_url = f"http://localhost:5000/api/v1/returns/{return_id}/ai-inspection"
        req = urllib.request.Request(
            shop_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-ai-service-key": "module4_secret"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[WEBHOOK] Đã đồng bộ kết quả AI sang Shop Server ({return_id}): Status {resp.status}")
    except Exception as ex:
        print(f"[WEBHOOK ERROR] Không thể gửi webhook đồng bộ sang Shop Server: {ex}")



def _get_or_recover_session(session_id: str) -> Optional[Dict[str, Any]]:
    session = sessions.get(session_id)
    if session:
        return session
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return None

    sess_json_path = os.path.join(session_dir, "session.json")
    if os.path.exists(sess_json_path):
        try:
            with open(sess_json_path, "r", encoding="utf-8") as f:
                session = json.load(f)
                sessions[session_id] = session
                return session
        except Exception:
            pass

    resolved_pid = "B22"
    det_p = os.path.join(session_dir, "detect_meta.json")
    if os.path.exists(det_p):
        try:
            with open(det_p, "r", encoding="utf-8") as df:
                dm = json.load(df)
                for d_val in dm.values():
                    if isinstance(d_val, dict) and "product_id" in d_val:
                        resolved_pid = d_val["product_id"].split("_")[0]
                        break
        except Exception:
            pass

    views_info = {}
    legacy_rev = {"front": "truoc", "back": "sau", "left": "trai", "right": "phai"}
    for v in VIEWS:
        candidates = [
            f"customer_{v}_processed.png",
            f"{v}.jpg",
            f"{v}.png",
            f"{legacy_rev.get(v, v)}.jpg",
            f"{legacy_rev.get(v, v)}.png"
        ]
        found_url = None
        for c in candidates:
            if os.path.exists(os.path.join(session_dir, c)):
                found_url = f"/uploads/{session_id}/{c}"
                break
        views_info[v] = {"status": "uploaded" if found_url else "pending", "url": found_url}

    session = {
        "product_id": resolved_pid,
        "created_at": time.time(),
        "views": views_info,
        "status": "complete" if any(v["status"] == "uploaded" for v in views_info.values()) else "in_progress",
    }
    sessions[session_id] = session
    return session


@app.get("/api/session/{session_id}/status")
async def session_status(session_id: str):
    """Desktop polling kiểm tra tiến độ chụp."""
    session = _get_or_recover_session(session_id)
    if not session:
        return JSONResponse({"error": "Session không tồn tại"}, status_code=404)

    return {
        "status": session["status"],
        "product_id": session["product_id"],
        "return_id": session.get("return_id"),
        "pipeline_status": session.get("pipeline_status", "idle"),
        "pipeline_step": session.get("pipeline_step", ""),
        "views": session["views"],
        "is_damaged": session.get("is_damaged"),
        "severity": session.get("severity"),
        "queue_size": inspection_queue.qsize(),
    }


@app.get("/api/session/{session_id}/qr")
async def session_qr_image(session_id: str):
    """Trả về mã QR định dạng hình ảnh PNG."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session không tồn tại"}, status_code=404)

    base_url = getattr(app.state, "public_url", f"http://localhost:{SERVER_PORT}")
    capture_url = f"{base_url}/capture/{session_id}"
    qr_bytes = generate_qr_bytes(capture_url)
    return Response(content=qr_bytes, media_type="image/png")



def _execute_preprocess_sync(session_id: str) -> Dict[str, Any]:
    """
    Tiền xử lý ảnh (Pha 1: Tách nền xám 40 + Crop; Pha 2: Quét lóa sáng).
    """
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "Session directory không tồn tại"}, status_code=404)

    session = _get_or_recover_session(session_id)
    product_id = session.get("product_id", "B1") if session else "B1"

    results = {}
    for view in VIEWS:
        candidate_raw = [
            os.path.join(session_dir, f"{view}.jpg"),
            os.path.join(session_dir, f"{view}.png"),
        ]
        for leg_k, leg_v in LEGACY_VIEW_MAP.items():
            if leg_v == view and leg_k != view:
                candidate_raw.append(os.path.join(session_dir, f"{leg_k}.jpg"))
                candidate_raw.append(os.path.join(session_dir, f"{leg_k}.png"))

        raw_path = next((p for p in candidate_raw if os.path.exists(p)), None)
        if not raw_path:
            continue

        try:
            res_cust = process_and_detect(
                raw_path,
                label=f"customer_{view}",
                output_dir=session_dir,
            )
            ignore_pct = round(float(res_cust.get("ignore_ratio", 0.0)) * 100, 2)

            candidate_refs = [
                os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}.png"),
                os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}_nb.png"),
                os.path.join(REFERENCE_IMAGES_DIR, str(product_id), f"{view}.png"),
                os.path.join(DATA_DIR, "references", str(product_id), f"{view}.png"),
                os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}.jpg"),
            ]
            ref_path = next((p for p in candidate_refs if os.path.exists(p)), None)
            ref_processed_url = None
            if ref_path:
                try:
                    process_and_detect(
                        ref_path,
                        label=f"reference_{view}",
                        output_dir=session_dir,
                    )
                    ref_processed_url = f"/uploads/{session_id}/reference_{view}_processed.png"
                except Exception as e_ref:
                    print(f"[WARN] Failed to preprocess reference image {view}: {e_ref}")

            results[view] = {
                "status": "ok",
                "ignore_ratio": ignore_pct,
                "should_reshoot": bool(res_cust.get("should_reshoot", False)),
                "processed_url": f"/uploads/{session_id}/customer_{view}_processed.png",
                "specular_mask_url": f"/uploads/{session_id}/customer_{view}_specular_mask.png",
                "saturated_mask_url": f"/uploads/{session_id}/customer_{view}_saturated_mask.png",
                "ignore_mask_url": f"/uploads/{session_id}/customer_{view}_ignore_mask.png",
                "overlay_contour_url": f"/uploads/{session_id}/customer_{view}_overlay_contour.png",
                "reference_processed_url": ref_processed_url,
                "goc_processed_url": ref_processed_url,
            }
        except Exception as e:
            print(f"[ERROR] Preprocessing error for {view}: {e}")
            results[view] = {"status": "error", "message": str(e)}

    # Lưu metadata tiền xử lý
    meta_path = os.path.join(session_dir, "preprocess_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Không thể lưu preprocess_meta.json: {e}")

    if session:
        session["preprocess_results"] = results
        _save_session_disk(session_id, session)

    return results


@app.post("/api/session/{session_id}/preprocess")
async def preprocess_session(session_id: str):
    """Tiền xử lý ảnh thủ công (fallback)."""
    try:
        results = await asyncio.to_thread(_execute_preprocess_sync, session_id)
        return {"status": "ok", "session_id": session_id, "results": results}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def _execute_detect_sync(session_id: str) -> Dict[str, Any]:
    """Hàm đồng bộ thực thi Soi lỗi đối chiếu vi sai ReferencePipeline."""
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "Session directory không tồn tại"}, status_code=404)

    results = {}
    try:
        pipeline = get_reference_pipeline()
        session = _get_or_recover_session(session_id)
        product_id = session.get("product_id", "B1") if session else "B1"

        view_images = {}
        for view in VIEWS:
            candidate_proc = [
                os.path.join(session_dir, f"customer_{view}_processed.png"),
                os.path.join(session_dir, f"khach_{view}_processed.png"),
                os.path.join(session_dir, f"{view}.jpg"),
                os.path.join(session_dir, f"{view}.png"),
            ]
            for leg_k, leg_v in LEGACY_VIEW_MAP.items():
                if leg_v == view and leg_k != view:
                    candidate_proc.append(os.path.join(session_dir, f"customer_{leg_k}_processed.png"))
                    candidate_proc.append(os.path.join(session_dir, f"khach_{leg_k}_processed.png"))
                    candidate_proc.append(os.path.join(session_dir, f"{leg_k}.jpg"))
            r_path = next((p for p in candidate_proc if os.path.exists(p)), None)
            if r_path:
                view_images[view] = r_path

        # Thực thi kiểm định đa góc tổng hợp có thuật toán Khử trùng lặp 3D
        session_summary = pipeline.inspect_session(
            product_id=product_id,
            view_images=view_images,
            output_dir=session_dir
        )

        view_reports = session_summary.get("views", {})
        results_views = {}
        for view in VIEWS:
            report = view_reports.get(view)
            if report:
                res_img_name = os.path.basename(report.get("result_image", ""))
                res_img_url = f"/uploads/{session_id}/{res_img_name}" if res_img_name else None
                view_res = {
                    "status": "ok",
                    "product_id": report.get("product_id"),
                    "is_damaged": report.get("is_damaged", False),
                    "defect_count": report.get("defect_count", 0),
                    "total_defects": report.get("defect_count", 0),
                    "damage_percentage": report.get("damage_percentage", 0.0),
                    "severity": report.get("severity", "INTACT"),
                    "needs_review": report.get("needs_review", False),
                    "review_reason": report.get("review_reason", ""),
                    "result_image_url": res_img_url,
                    "side_by_side_url": res_img_url,
                    "defects": report.get("defects", []),
                }
            else:
                view_res = {"status": "pending", "message": f"View {view} not captured or processed."}

            results_views[view] = view_res
            results[view] = view_res  # Giữ backward-compatibility cho detect_meta.get(view)

        results["status"] = session_summary.get("status", "ok")
        results["product_id"] = product_id
        results["views"] = results_views
        results["unique_defects"] = session_summary.get("unique_defects", [])
        results["dedup_pairs"] = session_summary.get("dedup_pairs", [])
        results["dedup_candidates_rejected"] = session_summary.get("dedup_candidates_rejected", [])
        results["total_defects"] = session_summary.get("total_defects", 0)
        results["raw_defect_count"] = session_summary.get("raw_defect_count", 0)
        results["is_damaged"] = session_summary.get("is_damaged", False)
        results["damage_percentage"] = session_summary.get("damage_percentage", 0.0)
        results["severity"] = session_summary.get("severity", "INTACT")
        results["verdict"] = session_summary.get("verdict", "INTACT")
        results["classification_summary"] = session_summary.get("classification_summary", "INTACT")

    except Exception as ex_global:
        print(f"[ERROR] Global detect error: {ex_global}")
        return JSONResponse({"status": "error", "message": str(ex_global)}, status_code=500)

    # Lưu metadata detect
    detect_meta_path = os.path.join(session_dir, "detect_meta.json")
    try:
        with open(detect_meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Không thể lưu detect_meta.json: {e}")

    if session:
        session["detect_results"] = results
        session["is_damaged"] = results.get("is_damaged", False)
        session["severity"] = results.get("severity", "INTACT")
        session["verdict"] = results.get("verdict", "INTACT")
        session["total_defects"] = results.get("total_defects", 0)
        _save_session_disk(session_id, session)
        return_id = session.get("return_id")
        if return_id:
            _trigger_shop_webhook(session_id, return_id, results, session)

    return results


@app.post("/api/session/{session_id}/detect")
async def detect_damage_session(session_id: str):
    """Phát hiện khuyết tật thủ công (fallback)."""
    try:
        results = await asyncio.to_thread(_execute_detect_sync, session_id)
        return {"status": "ok", "session_id": session_id, "results": results}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/session/{session_id}/simulate-defect")
async def api_simulate_defect(
    session_id: str,
    defect_type: str = Form("random"),
    view: str = Form("random"),
):
    """Giả lập cấy khuyết tật nhân tạo (Crack/Stain/Chip) phục vụ tester kiểm tra độ nhạy AI."""
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"status": "error", "message": "Session không tồn tại"}, status_code=404)
    session = _get_or_recover_session(session_id)
    product_id = session.get("product_id", "B1") if session else "B1"
    pipeline = get_reference_pipeline()
    try:
        res = await asyncio.to_thread(
            simulate_defect_on_session,
            session_dir,
            product_id,
            pipeline,
            view,
            defect_type,
            VIEWS,
        )
        return res
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/session/{session_id}/reset-simulation")
async def api_reset_simulation(session_id: str):
    """Khôi phục ảnh gốc sạch và kiểm định lại đưa session về trạng thái nguyên bản."""
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"status": "error", "message": "Session không tồn tại"}, status_code=404)
    session = _get_or_recover_session(session_id)
    product_id = session.get("product_id", "B1") if session else "B1"
    pipeline = get_reference_pipeline()
    try:
        res = await asyncio.to_thread(
            reset_simulation_on_session,
            session_dir,
            product_id,
            pipeline,
            VIEWS,
        )
        return res
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def inspection_worker():
    """Worker chạy nền tuần tự lấy session từ inspection_queue để tự động chạy Preprocess -> Detect."""
    print("[QUEUE] Background inspection worker initialized and listening...")
    while True:
        try:
            session_id = await inspection_queue.get()
            print(f"\n[QUEUE] Bắt đầu tự động thẩm định cho session {session_id}...")

            session = _get_or_recover_session(session_id)
            if not session:
                print(f"[QUEUE ERROR] Không tìm thấy session {session_id}.")
                inspection_queue.task_done()
                continue

            # Bước 1: Tiền xử lý
            session["pipeline_status"] = "preprocessing"
            session["pipeline_step"] = "Đang tự động tách nền và cân chỉnh 4 góc ảnh..."
            _save_session_disk(session_id, session)

            try:
                await asyncio.to_thread(_execute_preprocess_sync, session_id)
            except Exception as e_prep:
                print(f"[QUEUE ERROR] Tiền xử lý lỗi cho {session_id}: {e_prep}")
                session["pipeline_status"] = "error"
                session["pipeline_step"] = f"Lỗi tiền xử lý: {e_prep}"
                _save_session_disk(session_id, session)
                inspection_queue.task_done()
                continue

            # Bước 2: Soi lỗi vi sai DINOv2
            session["pipeline_status"] = "inspecting"
            session["pipeline_step"] = "Đang đối chiếu vi sai quang học DINOv2..."
            _save_session_disk(session_id, session)

            try:
                await asyncio.to_thread(_execute_detect_sync, session_id)
            except Exception as e_det:
                print(f"[QUEUE ERROR] Soi lỗi thất bại cho {session_id}: {e_det}")
                session["pipeline_status"] = "error"
                session["pipeline_step"] = f"Lỗi kiểm định: {e_det}"
                _save_session_disk(session_id, session)
                inspection_queue.task_done()
                continue

            # Bước 3: Hoàn tất
            session["pipeline_status"] = "ready"
            session["pipeline_step"] = "Kiểm định hoàn tất! Đang chuyển đến trang Kết quả..."
            session["completed_at"] = time.time()
            _save_session_disk(session_id, session)
            print(f"[QUEUE] Hoàn tất tự động toàn trình cho session {session_id}!\n")

            inspection_queue.task_done()
        except asyncio.CancelledError:
            print("[QUEUE] Worker cancelled on shutdown.")
            break
        except Exception as ex_global:
            print(f"[QUEUE UNEXPECTED ERROR] {ex_global}")
            await asyncio.sleep(1)


def _load_image_for_similarity(captured_url: str, session_id: str, view: str) -> Optional[np.ndarray]:
    """Tải ma trận ảnh BGR từ URL local hoặc cloud."""
    try:
        if captured_url.startswith("/uploads/"):
            cap_local = os.path.join(UPLOADS_DIR, session_id, f"{view}.jpg")
            return cv2.imread(cap_local)
        else:
            import urllib.request
            with urllib.request.urlopen(captured_url, timeout=8) as resp:
                buf = np.frombuffer(resp.read(), np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[WARN] Không thể tải ảnh {view} để tính tương đồng: {e}")
        return None


@app.get("/result/{session_id}", response_class=HTMLResponse)
async def result_page(request: Request, session_id: str):
    """Trang kết quả: Hiển thị 4 cặp ảnh mẫu vs ảnh chụp thực tế cùng kết quả kiểm định."""
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    session = _get_or_recover_session(session_id)
    if not session:
        return HTMLResponse("<h2 style='color:red'>Session không tồn tại hoặc đã hết hạn</h2>", status_code=404)

    product_id = session["product_id"]
    views_data = session["views"]

    # Đọc metadata tiền xử lý
    meta_path = os.path.join(session_dir, "preprocess_meta.json")
    preprocess_meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                preprocess_meta = json.load(f)
        except Exception:
            pass

    # Đọc metadata detect
    detect_meta_path = os.path.join(session_dir, "detect_meta.json")
    detect_meta = {}
    if os.path.exists(detect_meta_path):
        try:
            with open(detect_meta_path, "r", encoding="utf-8") as f:
                detect_meta = json.load(f)
        except Exception:
            pass

    pairs = []
    for view in VIEWS:
        info = views_data.get(view, {})
        captured_url = info.get("url") if info.get("status") == "uploaded" else None

        ghost_path = resolve_ghost_path(product_id, view)
        ghost_url = f"/ghost/{product_id}/{os.path.basename(ghost_path)}" if ghost_path else None

        sim_res = None
        if captured_url and ghost_url and os.path.exists(ghost_path):
            cap_bgr = _load_image_for_similarity(captured_url, session_id, view)
            if cap_bgr is not None:
                sim_res = compute_similarity_from_bgr(ghost_path, cap_bgr)

        if isinstance(sim_res, dict):
            sim_score = sim_res.get("score")
            ncc_score = sim_res.get("ncc")
            edge_score = sim_res.get("edge")
            corr_score = sim_res.get("corr")
            color_score = sim_res.get("color")
        else:
            sim_score = sim_res
            ncc_score = edge_score = corr_score = color_score = None

        pairs.append({
            "view": view,
            "label": VIEW_LABELS[view],
            "ghost_url": ghost_url,
            "captured_url": captured_url,
            "similarity": sim_score,
            "ncc": ncc_score,
            "edge": edge_score,
            "corr": corr_score,
            "color": color_score,
            "prep": preprocess_meta.get(view),
            "detect": detect_meta.get(view),
        })

    has_preprocessed = bool(preprocess_meta)
    has_detected = bool(detect_meta)

    pipe_st = session.get("pipeline_status", "ready" if (has_preprocessed and has_detected) else "idle")
    pipe_msg = session.get("pipeline_step", "")

    # Thống kê tổng hợp kết quả kiểm định hoàn hàng (Executive Return Inspection Summary)
    uploaded_views_count = sum(1 for v in VIEWS if views_data.get(v, {}).get("status") == "uploaded")

    if "unique_defects" in detect_meta:
        total_defects = detect_meta.get("total_defects", len(detect_meta.get("unique_defects", [])))
        damaged_views_count = sum(1 for v in VIEWS if detect_meta.get(v, {}).get("is_damaged", False))
        inspected_views_count = sum(1 for v in VIEWS if detect_meta.get(v, {}).get("status") == "ok")
        verdict = detect_meta.get("verdict", "INTACT" if total_defects == 0 else "DEFECTIVE")
        all_intact = (verdict == "INTACT")
    else:
        total_defects = 0
        damaged_views_count = 0
        inspected_views_count = 0
        all_intact = True

        for v in VIEWS:
            dm = detect_meta.get(v)
            if dm and isinstance(dm, dict) and dm.get("status") == "ok":
                inspected_views_count += 1
                cnt = dm.get("defect_count", 0)
                total_defects += cnt
                if dm.get("is_damaged", False) or cnt > 0:
                    damaged_views_count += 1
                    all_intact = False

        verdict = "INTACT" if (inspected_views_count > 0 and all_intact) else ("DEFECTIVE" if damaged_views_count > 0 else "PENDING")

    inspection_summary = {
        "uploaded_count": uploaded_views_count,
        "is_complete_capture": uploaded_views_count == len(VIEWS),
        "inspected_count": inspected_views_count,
        "is_fully_inspected": inspected_views_count == len(VIEWS),
        "is_damaged": (not all_intact) if inspected_views_count > 0 else False,
        "total_defects": total_defects,
        "damaged_views_count": damaged_views_count,
        "verdict": verdict,
        "dedup_pairs": detect_meta.get("dedup_pairs", []),
        "dedup_rejected": detect_meta.get("dedup_candidates_rejected", []),
        "unique_defects": detect_meta.get("unique_defects", [])
    }

    is_simulated = any(
        isinstance(dm, dict) and dm.get("is_simulated", False)
        for dm in detect_meta.values()
    ) or any(
        os.path.exists(os.path.join(session_dir, f"customer_{v}_processed_clean.png"))
        for v in VIEWS
    )

    return templates.TemplateResponse(request, "result.html", {
        "session_id": session_id,
        "product_id": product_id,
        "return_id": session.get("return_id"),
        "pairs": pairs,
        "has_preprocessed": has_preprocessed,
        "has_detected": has_detected,
        "pipeline_status": pipe_st,
        "pipeline_step": pipe_msg,
        "inspection_summary": inspection_summary,
        "is_simulated": is_simulated,
    })


# ── Static Uploads Mount (Phải đặt sau các route API) ─────────────────────────
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# ── Server Main Runner ────────────────────────────────────────────────────────
def main():
    import uvicorn

    # Khởi động HTTPS tunnel nếu có cloudflared
    def _sync_url(new_url: str):
        app.state.public_url = new_url
        print(f"[TUNNEL-SYNC] Public URL updated: {new_url}")
    set_url_change_callback(_sync_url)
    public_url, tunnel_proc = start_tunnel(SERVER_PORT, CLOUDFLARED_BIN)
    app.state.public_url = public_url

    print(f"\n{'='*65}")
    if public_url.startswith("https://"):
        print(f"  [HTTPS] PUBLIC URL: {public_url}")
        print(f"  [IPHONE] Camera tự động hoạt động 100% trên Safari iOS!")
    else:
        print(f"  [HTTP] Local URL:   {public_url}")
    print(f"{'='*65}")

    print(f"\n  [DESKTOP] Mở trình duyệt máy tính: {public_url}/")
    print(f"  [SERVER]                           http://localhost:{SERVER_PORT}\n")

    try:
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
