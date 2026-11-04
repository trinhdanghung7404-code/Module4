"""
tools/inspect_image.py
======================
CLI tool to inspect an image for ceramic defects using ReferencePipeline.
Usage:
    python tools/inspect_image.py <path_to_image> [product_id]
or run interactively:
    python tools/inspect_image.py
"""

import os
import sys
import cv2
import json
import torch
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import PROFILES_DIR, UPLOADS_DIR
from core.pipeline import ReferencePipeline

INSPECT_OUTPUT_DIR = os.path.join(UPLOADS_DIR, "inspections")
os.makedirs(INSPECT_OUTPUT_DIR, exist_ok=True)


class SingleImageDetector:
    def __init__(self, profiles_dir: str = PROFILES_DIR):
        print("=" * 68)
        print("       HỆ THỐNG SOI KHUYẾT TẬT GỐM SỨ (AI DINOv2)")
        print("=" * 68)
        print("[1/2] Đang nạp mô hình DINOv2 backbone vào bộ nhớ...")
        self.pipeline = ReferencePipeline(profiles_dir=profiles_dir)

        print("[2/2] Đang nạp danh sách hồ sơ vân tay sản phẩm gốc...")
        self.profiles = {}
        self.ref_vectors = {}

        if os.path.isdir(profiles_dir):
            for f in os.listdir(profiles_dir):
                if f.endswith("_profile.pt"):
                    p_id = f.replace("_profile.pt", "")
                    p_path = os.path.join(profiles_dir, f)
                    data = torch.load(p_path, map_location="cpu", weights_only=False)
                    self.profiles[p_id] = data
                    vec = data["dino_features"][0].mean(dim=0)
                    vec = vec / (vec.norm() + 1e-8)
                    self.ref_vectors[p_id] = vec
                    print(f"  --> Đã nạp hồ sơ: [{p_id}]")

        print(">> KHỞI ĐỘNG HOÀN TẤT! SẴN SÀNG NHẬN ẢNH ĐỂ KIỂM ĐỊNH.")
        print("=" * 68 + "\n")

    def identify_product(self, img_bgr: np.ndarray):
        """Tự động xác định loại bình bằng DINOv2 cosine similarity."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
        t_img = self.pipeline.transform(resized).unsqueeze(0).to(self.pipeline.device)

        with torch.no_grad():
            f = self.pipeline.backbone.get_intermediate_layers(t_img, n=1)[0].squeeze(0)
            q_vec = f.mean(dim=0)
            q_vec = q_vec / (q_vec.norm() + 1e-8)

        best_id = None
        best_sim = -1.0

        for p_id, ref_vec in self.ref_vectors.items():
            sim = (q_vec @ ref_vec).item()
            if sim > best_sim:
                best_sim = sim
                best_id = p_id

        return best_id, best_sim

    def detect(self, img_path: str, product_id: str = None):
        """Kiểm tra 1 ảnh và in kết quả trực quan."""
        clean_path = img_path.strip().strip('"').strip("'").strip('&').strip()

        if not os.path.exists(clean_path):
            print(f"\n[LỖI] Không tìm thấy file ảnh tại: {clean_path}")
            return None

        img_bgr = cv2.imread(clean_path)
        if img_bgr is None:
            print(f"\n[LỖI] Không thể đọc được nội dung ảnh: {clean_path}\n")
            return None

        # 1. Tự động nhận diện loại bình nếu chưa được chỉ định
        if not product_id or product_id not in self.profiles:
            product_id, sim = self.identify_product(img_bgr)
            print(f"\n[*] Nhận diện loại bình: [{product_id}] (Độ tương đồng hoa văn: {sim*100:.1f}%)")
        else:
            print(f"\n[*] Kiểm tra theo loại bình: [{product_id}]")

        # 2. Chạy pipeline kiểm định
        prod_out_dir = os.path.join(INSPECT_OUTPUT_DIR, product_id or "unknown")
        os.makedirs(prod_out_dir, exist_ok=True)

        report = self.pipeline.inspect_return(
            return_img_path=clean_path,
            product_id=product_id,
            output_dir=prod_out_dir
        )

        # 3. Hiển thị kết quả
        print("\n" + "=" * 68)
        print("                    KẾT QUẢ KIỂM ĐỊNH KHUYẾT TẬT")
        print("=" * 68)
        print(f"🏺 Loại sản phẩm:       {product_id}")
        print(f"🖼️  File ảnh kiểm tra:   {os.path.basename(clean_path)}")

        if report["is_damaged"]:
            print("🚨 Tình trạng:          CÓ KHUYẾT TẬT / HƯ HỎNG!")
            print(f"🔢 Số vết khuyết tật:   {report['defect_count']} vết")
            print(f"📊 Tỷ lệ hư hại:        {report['damage_percentage']:.2f}% diện tích bề mặt")
            print(f"⚠️  Mức độ:              {report['severity']}")
            print("-" * 68)
            print("Chi tiết các vị trí khuyết tật:")
            for idx, d in enumerate(report["defects"], 1):
                box = d["bbox"]
                print(f"  [{idx}] {d['label']} - Tọa độ [X={box[0]}, Y={box[1]}, Rộng={box[2]}, Cao={box[3]}]")
        else:
            print("✅ Tình trạng:          NGUYÊN VẸN 100% (Không có khuyết tật)")
            print("🔢 Số vết khuyết tật:   0 vết")
            print("📊 Tỷ lệ hư hại:        0.00%")

        print("=" * 68)
        print(f"📁 Ảnh kết quả đã khoanh vùng: {report.get('result_image', '')}")
        print(f"📄 Báo cáo chi tiết JSON:      {os.path.splitext(report.get('result_image', ''))[0]}.json")
        print("=" * 68)

        # 4. Mở ảnh kết quả
        try:
            if os.name == 'nt' and os.path.exists(report.get('result_image', '')):
                os.startfile(report['result_image'])
        except Exception:
            pass

        return report


def main():
    detector = SingleImageDetector()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        p_id = sys.argv[2] if len(sys.argv) > 2 else None
        detector.detect(img_path, p_id)
        return

    while True:
        try:
            print("-" * 68)
            user_input = input("👉 Kéo thả file ảnh vào đây rồi ấn Enter (hoặc gõ 'q' để thoát): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['q', 'exit', 'quit']:
                print(">> Đã thoát chương trình.")
                break

            detector.detect(user_input)
            print("\n")
        except KeyboardInterrupt:
            print("\n>> Đã dừng chương trình.")
            break
        except Exception as e:
            print(f"\n[LỖI NGOẠI LỆ]: {e}\n")


if __name__ == "__main__":
    main()
