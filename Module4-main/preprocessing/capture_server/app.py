"""
Capture Server - FastAPI
Chạy: python app.py
"""
import os
import sys
import uuid
import time
import base64
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    SERVER_HOST, SERVER_PORT,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET,
    GHOST_IMAGES_DIR, SESSION_EXPIRE_MINUTES,
    VIEWS, VIEW_LABELS, AUTO_CAPTURE,
)
from qr_generator import generate_qr_base64, generate_qr_bytes

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Capture Server - Damage Detection")

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Serve ghost images từ Test_Camera/
if os.path.isdir(GHOST_IMAGES_DIR):
    app.mount("/ghost", StaticFiles(directory=GHOST_IMAGES_DIR), name="ghost")

# Thư mục lưu ảnh local (fallback khi không có Cloudinary)
LOCAL_UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(LOCAL_UPLOADS, exist_ok=True)

# ── Cloudinary (optional) ────────────────────────────────────────────────────

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
        print("[OK] Cloudinary da ket noi.")
    except ImportError:
        print("[WARN] cloudinary chua cai. Anh se luu local.")
else:
    print("[WARN] Cloudinary chua cau hinh. Anh se luu local trong ./uploads/")

# ── Session store (in-memory) ────────────────────────────────────────────────

sessions: dict = {}


def _clean_expired():
    """Xóa session hết hạn."""
    now = time.time()
    expired = [
        sid for sid, s in sessions.items()
        if now - s["created_at"] > SESSION_EXPIRE_MINUTES * 60
    ]
    for sid in expired:
        del sessions[sid]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def desktop_page(request: Request):
    """Trang desktop: chọn sản phẩm → hiện QR."""
    return templates.TemplateResponse(request, "desktop.html")


_onnx_session = None

def get_onnx_segmenter_session():
    global _onnx_session
    if _onnx_session is None:
        try:
            import onnxruntime as ort
            candidate_paths = [
                os.path.expanduser(r"~/.rembg/models/u2netp/u2netp.onnx"),
                os.path.expanduser(r"~/.rembg/models/bria-rmbg/bria-rmbg.onnx"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "u2netp.onnx"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    _onnx_session = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
                    print(f"[OK] Da nap model ONNX tach nen truc tiep: {p}")
                    break
        except Exception as e:
            print(f"[WARN] Khong khoi tao duoc ONNX session: {e}")
    return _onnx_session


def process_single_ghost_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Tách nền trực tiếp bằng ONNX Runtime và cắt sát viền alpha.
    Hoàn toàn KHÔNG phụ thuộc vào Numba / Pymatting -> Không bao giờ bị lỗi Windows Application Control.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    session = get_onnx_segmenter_session()
    alpha = None

    if session is not None:
        try:
            # 1. Preprocess chuẩn cho u2net (320x320 RGB)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm_img = (resized - mean) / std
            input_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, :, :, :]

            # 2. Inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            outputs = session.run([output_name], {input_name: input_tensor})
            pred = outputs[0][0, 0, :, :]

            # 3. Normalize pred
            ma, mi = np.max(pred), np.min(pred)
            pred_norm = (pred - mi) / (ma - mi + 1e-8)

            # 4. Resize mask về kích thước ảnh gốc
            mask = cv2.resize(pred_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            alpha = (mask > 0.45).astype(np.uint8) * 255
        except Exception as e:
            print(f"[WARN] Inference ONNX loi: {e}, dung Otsu fallback")
            alpha = None

    if alpha is None:
        # Fallback: Thuật toán thuần OpenCV (Otsu + Morphology), không cần mô hình AI
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        alpha = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Ghép kênh alpha vào ảnh BGR
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    # Cắt sát viền vật thể (Tight Crop)
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = bgra[y:y+h, x:x+w]
    else:
        cropped = bgra

    return cropped


@app.get("/api/products")
async def list_products():
    """Liệt kê các mã sản phẩm đã có ảnh trong kho."""
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
    Tải lên ảnh gốc sản phẩm (trước/sau/trái/phải)
    -> Tự động bóc nền bằng rembg
    -> Cắt sát viền
    -> Lưu ảnh gốc + ảnh bóng mờ ghost (_nb.png)
    -> Upload đồng bộ lên Cloudinary nếu đã cấu hình
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

        # Đọc ảnh gốc bằng OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            results[view] = {"status": "error", "message": "Không thể đọc định dạng ảnh"}
            continue

        # 1. Lưu ảnh gốc
        orig_filename = f"{view}.jpg"
        orig_path = os.path.join(target_dir, orig_filename)
        cv2.imwrite(orig_path, img)

        # 2. Xử lý tạo bóng mờ bằng AI (rembg + tight crop)
        try:
            ghost_img = process_single_ghost_image(img)
            ghost_filename = f"{view}_nb.png"
            ghost_path = os.path.join(target_dir, ghost_filename)
            cv2.imwrite(ghost_path, ghost_img)
        except Exception as e:
            print(f"[ERROR] Xu ly rembg cho {view} that bai: {e}")
            results[view] = {"status": "error", "message": f"Lỗi rembg: {e}"}
            continue

        # 3. Đồng bộ lên Cloudinary (nếu có)
        c_orig_url = None
        c_ghost_url = None
        if _cloudinary_ready:
            try:
                import cloudinary.uploader
                # Upload ảnh gốc
                r_orig = cloudinary.uploader.upload(
                    file_bytes,
                    folder=f"damage_detection/products/{product_id}",
                    public_id=f"{view}_orig",
                    overwrite=True,
                )
                c_orig_url = r_orig.get("secure_url")

                # Upload ảnh ghost PNG
                with open(ghost_path, "rb") as gf:
                    r_ghost = cloudinary.uploader.upload(
                        gf.read(),
                        folder=f"damage_detection/products/{product_id}",
                        public_id=f"{view}_nb",
                        overwrite=True,
                    )
                c_ghost_url = r_ghost.get("secure_url")
            except Exception as ce:
                print(f"[WARN] Upload Cloudinary that bai cho {view}: {ce}")

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
    """Tạo phiên trả hàng mới."""
    _clean_expired()

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
async def capture_page(request: Request, session_id: str):
    """Trang chụp ảnh trên điện thoại (auto-capture)."""
    session = sessions.get(session_id)
    if not session:
        return HTMLResponse(
            "<h1 style='color:#fff;background:#111;padding:40px;text-align:center'>"
            "Session không tồn tại hoặc đã hết hạn</h1>",
            status_code=404,
        )

    # Kiểm tra ghost images có sẵn không
    product_id = session["product_id"]
    ghost_available = os.path.isdir(os.path.join(GHOST_IMAGES_DIR, str(product_id)))

    return templates.TemplateResponse(request, "capture.html", {
        "session_id": session_id,
        "product_id": product_id,
        "views": VIEWS,
        "view_labels": VIEW_LABELS,
        "config": AUTO_CAPTURE,
        "ghost_available": ghost_available,
    })


@app.post("/api/upload/{session_id}")
async def upload_image(
    session_id: str,
    view: str = Form(...),
    image: UploadFile = File(...),
):
    """Nhận ảnh chụp, lưu lên Cloudinary hoặc local."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    if view not in VIEWS:
        return JSONResponse({"error": f"Invalid view: {view}"}, status_code=400)

    image_bytes = await image.read()

    # Luôn lưu file ảnh gốc nguyên bản vào ổ cứng server để xử lý trực tiếp
    session_dir = os.path.join(LOCAL_UPLOADS, session_id)
    os.makedirs(session_dir, exist_ok=True)
    local_path = os.path.join(session_dir, f"{view}.jpg")
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    local_url = f"/uploads/{session_id}/{view}.jpg"

    # Upload Cloudinary (nếu có cấu hình)
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
            print(f"Cloudinary error: {e}, fallback to local")

    session["views"][view] = {"status": "uploaded", "url": url}

    # Kiểm tra đã chụp đủ chưa
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
    """Desktop polling: kiểm tra tiến độ chụp."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    return {
        "status": session["status"],
        "product_id": session["product_id"],
        "views": session["views"],
    }


@app.get("/api/session/{session_id}/qr")
async def session_qr_image(session_id: str):
    """Trả QR code dưới dạng ảnh PNG."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    base_url = getattr(app.state, "public_url", f"http://localhost:{SERVER_PORT}")
    capture_url = f"{base_url}/capture/{session_id}"
    qr_bytes = generate_qr_bytes(capture_url)
    return Response(content=qr_bytes, media_type="image/png")


# NOTE: app.mount("/uploads"...) is intentionally placed AFTER all @app.get routes below.
# FastAPI mounts shadow any routes registered after them.


@app.post("/api/session/{session_id}/preprocess")
async def preprocess_session(session_id: str):
    """
    Chạy toàn bộ pipeline tiền xử lý (Pha 1: Tách nền xám 40 + Crop + Pha 2: Quét lóa sáng)
    từ preprocess_mask.py cho các góc ảnh thực tế của session.
    """
    session_dir = os.path.join(LOCAL_UPLOADS, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "Session directory not found"}, status_code=404)

    # Đảm bảo đường dẫn import preprocess_mask từ thư mục preprocessing
    prep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if prep_dir not in sys.path:
        sys.path.insert(0, prep_dir)
    import preprocess_mask

    session = sessions.get(session_id)
    product_id = session.get("product_id", "1") if session else "1"

    results = {}
    for view in VIEWS:
        raw_path = os.path.join(session_dir, f"{view}.jpg")
        if not os.path.exists(raw_path):
            continue

        try:
            # 1. Tiền xử lý ảnh chụp khách hàng
            res_khach = preprocess_mask.process_and_detect(
                raw_path,
                label=f"khach_{view}",
                output_dir=session_dir,
            )
            ignore_pct = round(float(res_khach.get("ignore_ratio", 0.0)) * 100, 2)

            # 2. Tiền xử lý ảnh mẫu gốc tham chiếu (nếu có)
            ref_path = os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}.jpg")
            goc_processed_url = None
            if os.path.exists(ref_path):
                try:
                    preprocess_mask.process_and_detect(
                        ref_path,
                        label=f"goc_{view}",
                        output_dir=session_dir,
                    )
                    goc_processed_url = f"/uploads/{session_id}/goc_{view}_processed.png"
                except Exception as e_goc:
                    print(f"[WARN] Khong xu ly duoc anh goc {view}: {e_goc}")

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

    # Lưu metadata json để đọc lại khi refresh
    meta_path = os.path.join(session_dir, "preprocess_meta.json")
    try:
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Cannot write preprocess_meta.json: {e}")

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
    Kích hoạt phát hiện khuyết tật (Damage Detection 2-Layer) từ module image-damage-detection.
    TUYỆT ĐỐI KHÔNG can thiệp hay sửa mã nguồn của image-damage-detection.
    Chỉ import và gọi run_detailed_debug() để phân tích chi tiết.
    """
    session_dir = os.path.join(LOCAL_UPLOADS, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "Session directory not found"}, status_code=404)

    # Bảo đảm nạp đúng config của v2 mà không đụng chạm mã nguồn của image-damage-detection
    saved_config = sys.modules.pop('config', None)
    results = {}
    try:
        mod4_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        v2_dir = os.path.join(mod4_root, "image-damage-detection", "v2")
        if v2_dir not in sys.path:
            sys.path.insert(0, v2_dir)
        
        from detailed_debug import run_detailed_debug

        session = sessions.get(session_id)
        product_id = session.get("product_id", "1") if session else "1"

        for view in VIEWS:
            # Đường dẫn ảnh khách
            r_path = os.path.join(session_dir, f"khach_{view}_processed.png")
            if not os.path.exists(r_path):
                r_path = os.path.join(session_dir, f"{view}.jpg")
            if not os.path.exists(r_path):
                continue

            # Đường dẫn ảnh mẫu shop
            p_path = os.path.join(session_dir, f"goc_{view}_processed.png")
            if not os.path.exists(p_path):
                p_path = os.path.join(GHOST_IMAGES_DIR, str(product_id), f"{view}.jpg")
            if not os.path.exists(p_path):
                continue

            try:
                # Đồng bộ kích thước đầu vào để v2 không bị lệch ma trận
                detect_view_dir = os.path.join(session_dir, f"detect_{view}")
                os.makedirs(detect_view_dir, exist_ok=True)

                p_img = cv2.imread(p_path)
                r_img = cv2.imread(r_path)
                if p_img is None or r_img is None:
                    continue

                TARGET_SIZE = (1000, 1000)
                p_std = cv2.resize(p_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                r_std = cv2.resize(r_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                p_in = os.path.join(detect_view_dir, "input_prod.png")
                r_in = os.path.join(detect_view_dir, "input_ret.png")
                cv2.imwrite(p_in, p_std)
                cv2.imwrite(r_in, r_std)

                # Gọi trực tiếp module kiểm định của bạn
                res = run_detailed_debug(p_in, r_in, base_debug_dir=detect_view_dir)

                # Thu thập các hình ảnh minh chứng khuyết tật theo từng tầng riêng biệt
                # 1. Lưới tam giác Delaunay (Mesh)
                mesh_dir = os.path.join(detect_view_dir, "02_mesh")
                mesh_url = None
                if os.path.exists(os.path.join(mesh_dir, "mesh_return.jpg")):
                    mesh_url = f"/uploads/{session_id}/detect_{view}/02_mesh/mesh_return.jpg"

                # 2. Layer 1 (Cấu trúc / Nứt vỡ / Gãy nét Canny)
                l1_dir = os.path.join(detect_view_dir, "03_layer1_structure")
                l1_overlay_url = None
                l1_canny_url = None
                if os.path.exists(os.path.join(l1_dir, "06_structure_damage_overlay.jpg")):
                    l1_overlay_url = f"/uploads/{session_id}/detect_{view}/03_layer1_structure/06_structure_damage_overlay.jpg"
                if os.path.exists(os.path.join(l1_dir, "04_return_canny_denoised.jpg")):
                    l1_canny_url = f"/uploads/{session_id}/detect_{view}/03_layer1_structure/04_return_canny_denoised.jpg"

                # 3. Layer 2 (Màu sắc / Tróc men / Bản đồ nhiệt Delta E)
                l2_dir = os.path.join(detect_view_dir, "04_layer2_color")
                l2_heatmap_url = None
                l2_overlay_url = None
                if os.path.exists(os.path.join(l2_dir, "01_chroma_delta_e_heatmap.jpg")):
                    l2_heatmap_url = f"/uploads/{session_id}/detect_{view}/04_layer2_color/01_chroma_delta_e_heatmap.jpg"
                if os.path.exists(os.path.join(l2_dir, "02_color_damage_overlay.jpg")):
                    l2_overlay_url = f"/uploads/{session_id}/detect_{view}/04_layer2_color/02_color_damage_overlay.jpg"

                # 4. Tầng tổng hợp (Fusion Side-by-side & Defect crops)
                fusion_dir = os.path.join(detect_view_dir, "05_fusion")
                side_url = None
                if os.path.exists(os.path.join(fusion_dir, "03_side_by_side_marked.jpg")):
                    side_url = f"/uploads/{session_id}/detect_{view}/05_fusion/03_side_by_side_marked.jpg"

                crops_dir = os.path.join(detect_view_dir, "06_defect_crops")
                crop_urls = []
                if os.path.isdir(crops_dir):
                    for f in sorted(os.listdir(crops_dir))[:6]: # Lấy tối đa 6 vị trí tiêu biểu
                        if f.endswith((".jpg", ".png")):
                            crop_urls.append(f"/uploads/{session_id}/detect_{view}/06_defect_crops/{f}")

                results[view] = {
                    "status": "ok",
                    "triangles": res.get("triangles", 0),
                    "l1_defects": res.get("l1_defects", 0),
                    "l2_defects": res.get("l2_defects", 0),
                    "total_defects": res.get("total_defects", 0),
                    "mesh_url": mesh_url,
                    "l1_overlay_url": l1_overlay_url,
                    "l1_canny_url": l1_canny_url,
                    "l2_heatmap_url": l2_heatmap_url,
                    "l2_overlay_url": l2_overlay_url,
                    "side_by_side_url": side_url,
                    "crop_urls": crop_urls,
                }

            except Exception as e:
                print(f"[ERROR] Damage detection error for {view}: {e}")
                results[view] = {"status": "error", "message": str(e)}
    finally:
        if saved_config is not None:
            sys.modules['config'] = saved_config

    # Lưu metadata detect
    detect_meta_path = os.path.join(session_dir, "detect_meta.json")
    try:
        import json
        with open(detect_meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Cannot write detect_meta.json: {e}")

    return {
        "status": "ok",
        "session_id": session_id,
        "results": results,
    }


@app.get("/result/{session_id}", response_class=HTMLResponse)
async def result_page(request: Request, session_id: str):
    """Trang kết quả: hiển thị trung thực 4 cặp ảnh mẫu vs ảnh chụp thực tế nguyên bản 100%."""
    session = sessions.get(session_id)
    session_dir = os.path.join(LOCAL_UPLOADS, session_id)
    if not session:
        if os.path.isdir(session_dir):
            views_info = {}
            for v in VIEWS:
                if os.path.exists(os.path.join(session_dir, f"{v}.jpg")):
                    views_info[v] = {"status": "uploaded", "url": f"/uploads/{session_id}/{v}.jpg"}
                else:
                    views_info[v] = {"status": "pending", "url": None}
            session = {
                "product_id": "1",
                "views": views_info,
                "status": "complete" if all(v["status"] == "uploaded" for v in views_info.values()) else "in_progress",
            }
        else:
            return HTMLResponse("<h2 style='color:red'>Session không tồn tại hoặc đã hết hạn</h2>", status_code=404)

    product_id = session["product_id"]
    views_data = session["views"]

    # Đọc metadata tiền xử lý nếu đã từng chạy
    meta_path = os.path.join(session_dir, "preprocess_meta.json")
    preprocess_meta = {}
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                preprocess_meta = json.load(f)
        except Exception:
            pass

    # Đọc metadata detect nếu đã từng chạy
    detect_meta_path = os.path.join(session_dir, "detect_meta.json")
    detect_meta = {}
    if os.path.exists(detect_meta_path):
        try:
            import json
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
            sim_res = _compute_similarity(ghost_path, captured_url, session_id, view)

        if isinstance(sim_res, dict):
            sim_score = sim_res.get("score")
            ncc_score = sim_res.get("ncc")
            edge_score = sim_res.get("edge")
            corr_score = sim_res.get("corr")
        else:
            sim_score = sim_res
            ncc_score = None
            edge_score = None
            corr_score = None

        pairs.append({
            "view": view,
            "label": VIEW_LABELS[view],
            "ghost_url": ghost_url,
            "captured_url": captured_url,
            "similarity": sim_score,
            "ncc": ncc_score,
            "edge": edge_score,
            "corr": corr_score,
            "prep": preprocess_meta.get(view),
            "detect": detect_meta.get(view),
        })

    has_preprocessed = bool(preprocess_meta)
    has_detected = bool(detect_meta)

    return templates.TemplateResponse(request, "result.html", {
        "session_id": session_id,
        "product_id": product_id,
        "pairs": pairs,
        "has_preprocessed": has_preprocessed,
        "has_detected": has_detected,
    })


def _compute_similarity(ghost_path: str, captured_url: str, session_id: str, view: str):
    """
    Đánh giá độ tương đồng thực tế giữa ảnh mẫu và ảnh chụp camera (Nguyên bản 100%):
    - TUYỆT ĐỐI KHÔNG làm biến dạng (no warping) hoặc chỉnh sửa file ảnh chụp thực tế.
    - Đánh giá trực tiếp trên vùng khung ngắm thực tế của camera khi người dùng căn chỉnh.
    - Trả về các chỉ số trung thực: NCC cấu trúc, độ khớp đường nét và tương quan ánh sáng.
    """
    try:
        ghost_bgra = cv2.imread(ghost_path, cv2.IMREAD_UNCHANGED)
        if ghost_bgra is None:
            return None

        # Đọc ảnh chụp thực tế (nguyên bản)
        if captured_url.startswith("/uploads/"):
            cap_local = os.path.join(LOCAL_UPLOADS, session_id, f"{view}.jpg")
            cap_bgr = cv2.imread(cap_local)
        else:
            import urllib.request
            with urllib.request.urlopen(captured_url, timeout=8) as resp:
                buf = np.frombuffer(resp.read(), np.uint8)
            cap_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if cap_bgr is None:
            return None

        ghost_bgr = ghost_bgra[:, :, :3]
        alpha = ghost_bgra[:, :, 3]

        H, W = cap_bgr.shape[:2]
        gh, gw = ghost_bgra.shape[:2]

        # Chuẩn hóa tỉ lệ gốc giữa ảnh mẫu (độ phân giải cao, ví dụ 2379x1586) và khung hình camera (1280x960):
        # Tỉ lệ cơ sở: chuẩn hóa chiều cao ảnh mẫu về tương đương chiều cao camera
        rel_base = H / gh

        ds = 0.5
        s_cap = cv2.resize(cap_bgr, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

        best_v = -1
        best_l = (0, 0)
        best_s = 0.60

        # Quét các tỉ lệ vật thể thực tế người dùng cầm máy: từ 40% đến 80% chiều cao camera
        for s in [0.40, 0.48, 0.54, 0.60, 0.66, 0.72, 0.80]:
            actual_s = rel_base * s
            tg_w = int(gw * actual_s)
            tg_h = int(gh * actual_s)
            if tg_w >= W or tg_h >= H or tg_w < 20 or tg_h < 20:
                continue

            small_g = cv2.resize(ghost_bgr, (int(tg_w * ds), int(tg_h * ds)), interpolation=cv2.INTER_AREA)
            small_a = cv2.resize(alpha,     (int(tg_w * ds), int(tg_h * ds)), interpolation=cv2.INTER_NEAREST)

            res = cv2.matchTemplate(s_cap, small_g, cv2.TM_CCOEFF_NORMED, mask=small_a)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_v:
                best_v = max_v
                best_l = max_l
                best_s = s

        actual_s = rel_base * best_s
        tw = int(gw * actual_s)
        th = int(gh * actual_s)
        x0 = int(best_l[0] / ds)
        y0 = int(best_l[1] / ds)

        # Cắt đúng vùng vật thể thực tế trong ảnh chụp camera
        y1 = min(H, y0 + th)
        x1 = min(W, x0 + tw)
        y0 = max(0, y0)
        x0 = max(0, x0)
        cap_roi = cap_bgr[y0:y1, x0:x1]

        if cap_roi.shape[0] < 20 or cap_roi.shape[1] < 20:
            cap_roi = cap_bgr

        # Chuẩn hóa về cùng kích thước so sánh
        SIZE = (180, 180)
        g_small = cv2.resize(ghost_bgr, SIZE, interpolation=cv2.INTER_AREA)
        a_raw = cv2.resize(alpha, SIZE, interpolation=cv2.INTER_NEAREST)
        # Erode mask nhẹ 2px để loại trừ viền nền ngoại cảnh (tường/bàn)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode((a_raw > 30).astype(np.uint8), kernel, iterations=2) > 0
        if mask.sum() < 50:
            mask = a_raw > 30

        c_small = cv2.resize(cap_roi, SIZE, interpolation=cv2.INTER_AREA)

        if mask.sum() < 50:
            return None

        g_gray = cv2.cvtColor(g_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        c_gray = cv2.cvtColor(c_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gv = g_gray[mask]
        cv_ = c_gray[mask]

        # 1. Normalized Cross-Correlation (NCC) - Cấu trúc chi tiết thực tế
        gm, cm = gv - gv.mean(), cv_ - cv_.mean()
        ncc = float(np.dot(gm, cm) / (np.sqrt((gm**2).sum() * (cm**2).sum()) + 1e-6))
        ncc = max(0.0, min(1.0, ncc))

        # 2. Histogram Correlation - Phân bổ ánh sáng
        h1 = cv2.calcHist([gv.astype(np.uint8)], [0], None, [64], [0, 256])
        h2 = cv2.calcHist([cv_.astype(np.uint8)], [0], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        corr = max(0.0, min(1.0, corr))

        # 3. Sobel Edge Correlation - Khớp đường viền thực tế
        gx1 = cv2.Sobel(g_gray, cv2.CV_32F, 1, 0)
        gy1 = cv2.Sobel(g_gray, cv2.CV_32F, 0, 1)
        mag1 = np.sqrt(gx1**2 + gy1**2)[mask]

        gx2 = cv2.Sobel(c_gray, cv2.CV_32F, 1, 0)
        gy2 = cv2.Sobel(c_gray, cv2.CV_32F, 0, 1)
        mag2 = np.sqrt(gx2**2 + gy2**2)[mask]

        edge_score = float(np.dot(mag1, mag2) / (np.sqrt((mag1**2).sum() * (mag2**2).sum()) + 1e-6))
        edge_score = max(0.0, min(1.0, edge_score))

        # Điểm tương đồng trung thực (phản ánh đúng thực tế người chụp căn chỉnh)
        score = 0.45 * ncc + 0.35 * edge_score + 0.20 * corr
        score = round(float(np.clip(score, 0.0, 1.0)), 3)

        return {
            "score": score,
            "ncc": round(ncc, 3),
            "edge": round(edge_score, 3),
            "corr": round(corr, 3),
        }

    except Exception as e:
        print(f"[WARN] Honest similarity calculation error for {view}: {e}")
        return None


# Serve local uploads — PHẢI đặt sau tất cả @app.get/@app.post routes
app.mount("/uploads", StaticFiles(directory=LOCAL_UPLOADS), name="uploads")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    public_url = f"http://localhost:{SERVER_PORT}"
    tunnel_proc = None

    # 1. Khởi động HTTPS Tunnel bằng Cloudflare (Miễn phí, có SSL chính thức, iPhone Safari mở camera 100%)
    cloudflared_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe")
    if os.path.exists(cloudflared_bin):
        try:
            import subprocess, re
            print("[INFO] Dang khoi tao HTTPS Tunnel cho camera iPhone...")
            cmd = [cloudflared_bin, "tunnel", "--url", f"http://localhost:{SERVER_PORT}"]
            tunnel_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            
            start_t = time.time()
            while time.time() - start_t < 15:
                line = tunnel_proc.stderr.readline()
                if line:
                    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                    if m:
                        public_url = m.group(0)
                        break
        except Exception as e:
            print(f"[WARN] Khong khoi tao duoc Cloudflare tunnel: {e}")

    # 2. Fallback sang ngrok nếu chưa có HTTPS
    if not public_url.startswith("https://"):
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(SERVER_PORT, "http")
            public_url = tunnel.public_url.replace("http://", "https://")
        except Exception:
            pass

    # 3. Fallback sang IP nội bộ (cùng WiFi)
    if not public_url.startswith("https://"):
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            public_url = f"http://{local_ip}:{SERVER_PORT}"
        except Exception:
            pass

    app.state.public_url = public_url

    print(f"\n{'='*65}")
    if public_url.startswith("https://"):
        print(f"  [HTTPS] PUBLIC URL: {public_url}")
        print(f"  [IPHONE] Camera tu dong hoat dong 100% tren Safari iOS!")
    else:
        print(f"  [HTTP] Local URL:   {public_url}")
    print(f"{'='*65}")

    print(f"\n  [DESKTOP] Mo trinh duyet may tinh: {public_url}/")
    print(f"  [SERVER]                           http://localhost:{SERVER_PORT}\n")

    try:
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
