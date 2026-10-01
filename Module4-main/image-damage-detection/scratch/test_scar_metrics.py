import sys, os

with open('v2/detailed_debug.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace("V2_DIR = os.path.dirname(os.path.abspath(__file__))", "V2_DIR = os.path.abspath('v2')")

hook_code = """
    # DEBUG PRINT HOOK
    target_ids = [22, 76, 98, 121, 123, 124, 299, 304, 341, 396, 431, 432, 433, 439, 454]
    print("\\n=== METRICS FOR SCAR & FLAGGED TRIANGLES ===")
    for item in patch_data:
        if item["id"] in target_ids:
            print(f"Tri #{item['id']:3d}: p_edge={item['p_edge_cnt']:2d}, r_edge={item['r_edge_cnt']:2d}, "
                  f"crack={item['dark_crack']:.1f}, scratch={item['white_scratch']:.1f}, "
                  f"intrusive={item['intrusive_len']:.1f}, broken={item['broken_length']:.1f}, "
                  f"blob={item['blob_area']}, z_l1={item.get('z_l1', 0.0):.2f}, "
                  f"has_l1={item['has_physical_l1']}")
"""

code = code.replace('triangle_details = []', hook_code + '\n    triangle_details = []')

with open('scratch/detailed_debug_hooked.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Hooked script written.")
