import sys
import os

# Ensure v2 directory is at the root of sys.path
V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)


def quick_compare():
    """Directly compare two image paths with the full 2-layer pipeline."""
    import shutil
    from detailed_debug import run_detailed_debug

    print("\n--- SO SANH NHANH 2 ANH (2-LAYER INSPECTION) ---")
    p_default = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    r_default = os.path.join(PROJECT_ROOT, "images", "15.jpg")

    prod_path = input(f"Nhap duong dan anh Product [{p_default}]: ").strip()
    if not prod_path:
        prod_path = p_default

    ret_path = input(f"Nhap duong dan anh Return [{r_default}]: ").strip()
    if not ret_path:
        ret_path = r_default

    p_name = os.path.splitext(os.path.basename(prod_path))[0]
    r_name = os.path.splitext(os.path.basename(ret_path))[0]
    out_dir = os.path.join(V2_DIR, "debug", f"compare_{p_name}_vs_{r_name}")
    latest_dir = os.path.join(V2_DIR, "debug", "latest_run")

    print(f"\nDang xu ly: \n  Product: {prod_path}\n  Return:  {ret_path}\n  Folder:  {out_dir}")
    res = run_detailed_debug(prod_path, ret_path, base_debug_dir=out_dir)

    try:
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(out_dir, latest_dir)
    except Exception as e:
        pass

    print("\n" + "=" * 55)
    print("  KET QUA KIEM TRA CHI TIET 2 LAYER")
    print("=" * 55)
    print(f"  Tong so tam giac hoa van:          {res['triangles']}")
    print(f"  Layer 1 (Khong mau - Nut/Gay net): {res['l1_defects']} tam giac")
    print(f"  Layer 2 (Co mau - Troc men/Mat mau): {res['l2_defects']} tam giac")
    print(f"  Tong vi tri loi xac dinh:          {res['total_defects']} tam giac")
    print("=" * 55)
    print(f"  Ket qua da luu tai:")
    print(f"  --> {out_dir}")
    print(f"  --> {latest_dir}")
    print("=" * 55)


def menu():
    while True:
        print("\n=== Damage Detection System v2 ===")
        print("1. Add New Product (Database)")
        print("2. Analyze Return Product (Database)")
        print("3. Quick 2-Layer Compare (Nhap truc tiep 2 anh)")
        print("0. Exit")

        choice = input("Select option: ").strip()

        if choice == '1':
            from pipeline.product_pipeline import add_product
            add_product()
        elif choice == '2':
            from pipeline.return_pipeline import analyze_return_product
            analyze_return_product()
        elif choice == '3':
            quick_compare()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")


if __name__ == '__main__':
    menu()