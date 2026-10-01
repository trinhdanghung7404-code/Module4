import cv2
import numpy as np
import os
from rembg import remove

def preprocess_gimage():
    """
    Bóc nền (rembg) -> Tìm viền vật thể -> Cắt bỏ khoảng trong suốt thừa -> Lưu PNG.
    """
    views = ["truoc", "sau", "trai", "phai"]
    
    print("\n🚀 Bắt đầu tạo bóng mờ sát viền...")
    
    for view in views:
        input_path = f"{view}.jpg"
        output_path = f"{view}_nb.png"
        
        if not os.path.exists(input_path):
            print(f"⚠️ Bỏ qua: Không tìm thấy '{input_path}'")
            continue
            
        print(f"⏳ Đang xử lý: {input_path}...")
        
        # 1. Đọc ảnh
        img = cv2.imread(input_path)
        if img is None:
            print(f"❌ Lỗi đọc ảnh '{input_path}'")
            continue
            
        # 2. Xóa nền bằng AI
        img_nobg = remove(img)
        
        # 3. THUẬT TOÁN CẮT SÁT VIỀN (TIGHT CROP)
        alpha_channel = img_nobg[:, :, 3]
        
        # Tìm tọa độ của tất cả các điểm ảnh CÓ MÀU (khác 0)
        coords = cv2.findNonZero(alpha_channel)
        
        if coords is not None:
            # Lấy ra khung chữ nhật nhỏ nhất bao quanh vật thể (x, y, rộng, cao)
            x, y, w, h = cv2.boundingRect(coords)
            
            # Cắt tấm ảnh dứt khoát theo khung chữ nhật đó
            img_cropped = img_nobg[y:y+h, x:x+w]
        else:
            # Đề phòng trường hợp ảnh bị lỗi bóc sạch bách không còn gì
            img_cropped = img_nobg 
        
        # 4. Lưu ảnh đã cắt gọt ra PNG
        cv2.imwrite(output_path, img_cropped)
        
        print(f"✅ Xong: Đã tạo '{output_path}' (Đã cắt sát viền)")
        
    print("🎉 Hoàn tất! Các ảnh bóng mờ bây giờ đã được tối ưu kích thước.")

if __name__ == "__main__":
    preprocess_gimage()