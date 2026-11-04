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
import time
import json
import base64
import cv2
import numpy as np
from typing import Optional, Union, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    SERVER_HOST, SERVER_PORT,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET,
    PROFILES_DIR, GHOST_IMAGES_DIR, TEMPLATES_DIR, UPLOADS_DIR, CLOUDFLARED_BIN,
    SESSION_EXPIRE_MINUTES, VIEWS, VIEW_LABELS, AUTO_CAPTURE,
)
from core.pipeline import ReferencePipeline, resolve_profile_id
from core.preprocessor import process_and_detect
from core.matcher import (
    process_single_ghost_image,
    compute_similarity_from_bgr,
    validate_upload_match,
)
from utils.qr_code import generate_qr_base64, generate_qr_bytes
from utils.tunnel import start_tunnel

# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(title="Ceramic Defect Inspection Server - Module4_AIReturn")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

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
        if now - s["created_at"] > SESSION_EXPIRE_MINUTES * 60
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
    if os.path.isdir(GHOST_IMAGES_DIR):
        for name in os.listdir(GHOST_IMAGES_DIR):
            p_dir = os.path.join(GHOST_IMAGES_DIR, name)
            if os.path.isdir(p_dir):
                files = os.listdir(p_dir)
                nb_views = [v for v in VIEWS if f"{v}_nb.png" in files]
                if nb_views:
                    prods.append({
                        "id": name,
                        "views": nb_views,
                        "total_views": len(nb_views),
                    })
    prods.sort(key=lambda x: str(x["id"]))
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


@app.post("/api/session")
async def create_session(product_id: str = Form(...)):
    """Tạo phiên kiểm định mới và tạo mã QR để quét trên di động."""
    _clean_expired_sessions()

    session_id = uuid.uuid4().hex[:8]
    sessions[session_id] = {
        "product_id": product_id,
        "created_at": time.time(),
        "views": {v: {"status": "pending", "url": None} for v in VIEWS},
        "status": "waiting",
    }

    base_url = getattr(app.state, "public_url", f"http://localhost:{SERVER_PORT}")
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

    if view not in VIEWS:
        return JSONResponse({"error": f"Góc chụp không hợp lệ: {view}"}, status_code=400)

    image_bytes = await image.read()

    # Cửa soát chất lượng tải lên
    product_id = session.get("product_id", "B1")
    ghost_path = os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}_nb.png")
    val_res = validate_upload_match(ghost_path, image_bytes)
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

    uploaded_count = sum(1 for v in session["views"].values() if v["status"] == "uploaded")

    return {
        "status": "ok",
        "view": view,
        "url": url,
        "uploaded": uploaded_count,
        "total": len(VIEWS),
        "all_done": all_done,
    }


@app.get("/api/session/{session_id}/status")
async def session_status(session_id: str):
    """Desktop polling kiểm tra tiến độ chụp."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session không tồn tại"}, status_code=404)

    return {
        "status": session["status"],
        "product_id": session["product_id"],
        "return_id": session.get("return_id"),
        "views": session["views"],
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


@app.post("/api/session/{session_id}/preprocess")
async def preprocess_session(session_id: str):
    """
    Tiền xử lý ảnh (Pha 1: Tách nền xám 40 + Crop; Pha 2: Quét lóa sáng).
    """
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "Session directory không tồn tại"}, status_code=404)

    session = sessions.get(session_id)
    product_id = session.get("product_id", "B1") if session else "B1"

    results = {}
    for view in VIEWS:
        raw_path = os.path.join(session_dir, f"{view}.jpg")
        if not os.path.exists(raw_path):
            continue

        try:
            res_khach = process_and_detect(
                raw_path,
                label=f"khach_{view}",
                output_dir=session_dir,
            )
            ignore_pct = round(float(res_khach.get("ignore_ratio", 0.0)) * 100, 2)

            ref_path = os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}.jpg")
            goc_processed_url = None
            if os.path.exists(ref_path):
                try:
                    process_and_detect(
                        ref_path,
                        label=f"goc_{view}",
                        output_dir=session_dir,
                    )
                    goc_processed_url = f"/uploads/{session_id}/goc_{view}_processed.png"
                except Exception as e_goc:
                    print(f"[WARN] Không thể xử lý ảnh gốc {view}: {e_goc}")

            results[view] = {
                "status": "ok",
                "ignore_ratio": ignore_pct,
                "should_reshoot": bool(res_khach.get("should_reshoot", False)),
                "processed_url": f"/uploads/{session_id}/khach_{view}_processed.png",
                "specular_mask_url": f"/uploads/{session_id}/khach_{view}_specular_mask.png",
                "saturated_mask_url": f"/uploads/{session_id}/khach_{view}_saturated_mask.png",
                "ignore_mask_url": f"/uploads/{session_id}/khach_{view}_ignore_mask.png",
                "overlay_contour_url": f"/uploads/{session_id}/khach_{view}_overlay_contour.png",
                "goc_processed_url": goc_processed_url,
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

    return {
        "status": "ok",
        "session_id": session_id,
        "results": results,
    }


@app.post("/api/session/{session_id}/detect")
async def detect_damage_session(session_id: str):
    """
    Phát hiện khuyết tật qua ReferencePipeline (DINOv2 + Cổng kiểm chứng đối chiếu vi sai).
    """
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "Session directory không tồn tại"}, status_code=404)

    results = {}
    try:
        pipeline = get_reference_pipeline()
        session = sessions.get(session_id)
        product_id = session.get("product_id", "B1") if session else "B1"

        for view in VIEWS:
            r_path = os.path.join(session_dir, f"khach_{view}_processed.png")
            if not os.path.exists(r_path):
                r_path = os.path.join(session_dir, f"{view}.jpg")
            if not os.path.exists(r_path):
                continue

            profile_id = resolve_profile_id(product_id, view, pipeline.profiles_dir)
            if not profile_id:
                results[view] = {"status": "error", "message": f"Không tìm thấy hồ sơ vân tay cho {product_id} góc {view}"}
                continue

            try:
                report = pipeline.inspect_return(
                    return_img_path=r_path,
                    product_id=profile_id,
                    output_dir=session_dir
                )

                res_img_name = os.path.basename(report.get("result_image", ""))
                res_img_url = f"/uploads/{session_id}/{res_img_name}" if res_img_name else None

                results[view] = {
                    "status": "ok",
                    "product_id": profile_id,
                    "is_damaged": report.get("is_damaged", False),
                    "defect_count": report.get("defect_count", 0),
                    "total_defects": report.get("defect_count", 0),
                    "damage_percentage": report.get("damage_percentage", 0.0),
                    "severity": report.get("severity", "Nguyên vẹn"),
                    "result_image_url": res_img_url,
                    "side_by_side_url": res_img_url,
                    "defects": report.get("defects", []),
                }
            except Exception as e:
                print(f"[ERROR] Damage detection error for {view}: {e}")
                results[view] = {"status": "error", "message": str(e)}

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

    return {
        "status": "ok",
        "session_id": session_id,
        "results": results,
    }


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
    session = sessions.get(session_id)
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if not session:
        if os.path.isdir(session_dir):
            views_info = {}
            for v in VIEWS:
                if os.path.exists(os.path.join(session_dir, f"{v}.jpg")):
                    views_info[v] = {"status": "uploaded", "url": f"/uploads/{session_id}/{v}.jpg"}
                else:
                    views_info[v] = {"status": "pending", "url": None}
            session = {
                "product_id": "B1",
                "views": views_info,
                "status": "complete" if all(v["status"] == "uploaded" for v in views_info.values()) else "in_progress",
            }
        else:
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

        ghost_path = os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}_nb.png")
        ghost_url = f"/ghost/{product_id}/{view}_nb.png" if os.path.exists(ghost_path) else None

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

    return templates.TemplateResponse(request, "result.html", {
        "session_id": session_id,
        "product_id": product_id,
        "return_id": session.get("return_id"),
        "pairs": pairs,
        "has_preprocessed": has_preprocessed,
        "has_detected": has_detected,
    })


# ── Static Uploads Mount (Phải đặt sau các route API) ─────────────────────────
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# ── Server Main Runner ────────────────────────────────────────────────────────
def main():
    import uvicorn

    # Khởi động HTTPS tunnel nếu có cloudflared
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
