import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

def run_debug_mode(prod_path=None, ret_path=None, debug_subname=None):
    """Chạy kiểm tra chi tiết 2 layer và xuất toàn bộ folder debug vào v2/debug/"""
    from detailed_debug import run_detailed_debug
    import shutil
    
    if prod_path is None or ret_path is None:
        p_default = os.path.join(PROJECT_ROOT, "images", "1.jpg")
        r_default = os.path.join(PROJECT_ROOT, "images", "15.jpg")
        
        print("\n--- NHAP DUONG DAN ANH KIEM TRA ---")
        p_in = input(f"Nhap duong dan anh Product [{p_default}]: ").strip()
        r_in = input(f"Nhap duong dan anh Return  [{r_default}]: ").strip()
        prod_path = p_in if p_in else p_default
        ret_path = r_in if r_in else r_default

    # Tự động đặt tên folder theo tên ảnh nếu chưa chỉ định
    if not debug_subname:
        p_name = os.path.splitext(os.path.basename(prod_path))[0]
        r_name = os.path.splitext(os.path.basename(ret_path))[0]
        debug_subname = f"compare_{p_name}_vs_{r_name}"

    target_dir = os.path.join(V2_DIR, "debug", debug_subname)
    latest_dir = os.path.join(V2_DIR, "debug", "latest_run")

    print(f"\n[Main] Dang khoi chay kiem tra loi chi tiet...")
    print(f"  Product: {prod_path}")
    print(f"  Return:  {ret_path}")
    print(f"  Folder xuat: {target_dir}")
    
    res = run_detailed_debug(prod_path, ret_path, base_debug_dir=target_dir)

    # Cập nhật thư mục latest_run để người dùng luôn xem được kết quả mới nhất tại 1 chỗ
    try:
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(target_dir, latest_dir)
    except Exception as e:
        pass

    print("\n" + "=" * 68)
    print("         KET QUA KIEM TRA CHI TIET 2 LAYER (BUOC 3)")
    print("=" * 68)
    print(f"  - Tong so tam giac hoa van:          {res['triangles']}")
    print(f"  - Layer 1 (Khong mau - Nut/Gay net): {res['l1_defects']} tam giac")
    print(f"  - Layer 2 (Co mau - Troc men/Mau):   {res['l2_defects']} tam giac")
    print(f"  ==> TONG VI TRI XAC DINH BI LOI:     {res['total_defects']} tam giac")
    print("=" * 68)
    print(f"  KET QUA DEBUG DA DUOC LUU TAI 2 VI TRI:")
    print(f"  [1] Thu muc rieng theo cap anh: {target_dir}")
    print(f"  [2] Thu muc mac dinh moi nhat:  {latest_dir}")
    print("=" * 68)
    print(f"  Ban co the mo nhanh tren Windows bang lenh:")
    print(f"  explorer \"{latest_dir}\"")
    print("=" * 68)
    return res

def menu():
    while True:
        print("\n=======================================================")
        print("     HE THONG PHAT HIEN KHUYET TAT GOM SU (v2)")
        print("=======================================================")
        print("1. [DEBUG] Chay mau binh co vet rach/seo (test_nobg vs scar)")
        print("2. [DEBUG] Chay mau binh nguyen ven (1.jpg vs 2.jpg)")
        print("3. [DEBUG] Nhap 2 anh bat ky de kiem tra & xuat folder debug")
        print("4. Menu He Thong v2 (Database & Pipeline)")
        print("5. Menu Cu (Legacy)")
        print("0. Thoat")
        print("=======================================================")

        choice = input("Chon chuc nang (0-5): ").strip()

        if choice == "1":
            p = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
            r = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
            run_debug_mode(p, r, debug_subname="02_scar_defect")

        elif choice == "2":
            p = os.path.join(PROJECT_ROOT, "images", "1.jpg")
            r = os.path.join(PROJECT_ROOT, "images", "2.jpg")
            run_debug_mode(p, r, debug_subname="01_undamaged_vase")

        elif choice == "3":
            run_debug_mode()

        elif choice == "4":
            from v2.main import menu as v2_menu
            v2_menu()

        elif choice == "5":
            print("\n1. Add Product (Legacy)\n2. Analyze Return (Legacy)")
            sub = input("Chon: ").strip()
            if sub == "1":
                from product_pipeline import add_product
                add_product()
            elif sub == "2":
                from return_pipeline import analyze_return_product
                analyze_return_product()

        elif choice == "0":
            print("Thoat chuong trinh.")
            break

        else:
            print("Lua chon khong hop le, vui long nhap lai.")

if __name__ == "__main__":
    menu()