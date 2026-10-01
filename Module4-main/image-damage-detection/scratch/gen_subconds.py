import sys, os
sys.path.insert(0, 'v2')

with open('scratch/test_refined_logic.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = "is_severe_standalone = bool("
replacement = """c_intrusive = bool(item["intrusive_len"] >= 12.0)
        c_crack = bool(item["dark_crack"] >= 12.0 and item["ori_corr"] < 0.85)
        c_broken = bool(item["broken_length"] >= 25.0 and item["ori_corr"] < 0.80)
        c_blob = bool(item["blob_area"] >= 60 and item["z_l1"] > 0.4)
        is_severe_standalone = c_intrusive or c_crack or c_broken or c_blob
        if is_severe_standalone and not is_l2_damage:
            reasons = []
            if c_intrusive: reasons.append('Intrusive')
            if c_crack: reasons.append(f"Crack({item['dark_crack']:.1f},ori={item['ori_corr']:.2f})")
            if c_broken: reasons.append(f"Broken({item['broken_length']:.1f})")
            if c_blob: reasons.append(f"Blob({item['blob_area']},z={item['z_l1']:.2f})")
            print(f"FP Tri #{item['id']:3d}: {', '.join(reasons)}")
        _unused = bool("""

text = text.replace(target, replacement, 1)
part1 = text.split('if __name__ == "__main__":')[0]
runner = """
if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    run_detailed_debug(p1, p2, base_debug_dir=os.path.join(V2_DIR, "debug", "01_undamaged_vase"))
"""
full = part1 + runner
with open('scratch/test_subconds.py', 'w', encoding='utf-8') as f:
    f.write(full)

print("Created scratch/test_subconds.py successfully.")
